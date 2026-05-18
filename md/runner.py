"""JAX-MD simulation runner for trained CG protein force fields.

Runs NVT-Langevin or NVE dynamics using a CombinedModel loaded from PKL params.
All quantities in AKMA units (Å, kcal/mol, amu); 1 AKMA time unit ≈ 48.88 fs.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import simulate

from models.combined_model import CombinedModel
from utils.logging import md_logger

# Boltzmann constant in kcal/mol/K (AKMA unit system)
_kB_KCAL = 1.9872041e-3

# All observable keys in the default output order.
_ALL_OBSERVABLES = ["step", "T", "KE", "PE", "E_total", "pressure", "box_volume"]

# Energy-component keys returned by CombinedModel.compute_components that are
# always present (regardless of whether priors are enabled).
_DECOMP_ENERGY_KEYS = ["E_ml", "E_prior_total"]


class MDRunner:
    """Run NVT-Langevin or NVE dynamics with a trained CombinedModel.

    All simulation parameters use AKMA units:
      length  — Å
      energy  — kcal/mol
      mass    — amu (g/mol)
      time    — 1 AKMA ≈ 48.88 fs
      force   — kcal/mol/Å

    Typical values for 1 fs timestep at 300 K with 50 ps friction:
      dt    = 0.02045  (AKMA)
      kT    = 0.5961   (kcal/mol)
      gamma = 0.000977 (AKMA^-1)
    """

    def __init__(
        self,
        model: CombinedModel,
        params: Dict[str, Any],
        md_config: dict,
    ):
        self.model = model
        self.params = params

        self.integrator        = str(md_config.get("integrator", "nvt_langevin"))
        self.n_steps           = int(md_config.get("n_steps", 1000))
        self.dt                = float(md_config.get("dt", 0.02045))
        self.kT                = float(md_config.get("kT", 0.5961))
        self.gamma             = float(md_config.get("gamma", 0.000977))
        mass_raw = md_config.get("mass", 12.011)
        if isinstance(mass_raw, (list, tuple)):
            self.mass: Any = [float(m) for m in mass_raw]
        else:
            self.mass = float(mass_raw)
        self.output_every      = int(md_config.get("output_every", 10))
        self.continuous_output = bool(md_config.get("continuous_output", False))
        self.continuous_output_every = max(
            1, int(md_config.get("continuous_output_every", 10))
        )
        partial_output_path = md_config.get("_partial_output_path", None)
        self.partial_output_path = (
            Path(partial_output_path) if partial_output_path else None
        )

        # Equilibration
        self.equilibrate       = bool(md_config.get("equilibrate", False))
        self.n_equil_steps     = int(md_config.get("n_equil_steps", 0))

        # COM drift removal (equivalent to LAMMPS 'fix langevin ... zero yes').
        # Langevin noise kicks are independent per atom, so their sum is a
        # random force on the COM that causes diffusive drift — especially
        # visible for small molecules.  Default on.
        self.zero_com_velocity = bool(md_config.get("zero_com_velocity", True))

        # Scalar observables (logged to CSV, independent cadence from trajectory)
        self.observables_every = int(md_config.get("observables_every", self.output_every))
        raw_obs = md_config.get("observables", _ALL_OBSERVABLES)
        self.observables       = list(raw_obs)   # ordered list of keys to record

        # Force / energy decomposition mode: record E_ml, E_prior and their
        # force contributions (F_ml, F_prior) at output_every cadence.
        self.force_decomp       = bool(md_config.get("force_decomp", False))
        self.force_decomp_every = int(md_config.get("force_decomp_every", self.output_every))

        if self.integrator not in ("nvt_langevin", "nve"):
            raise ValueError(
                f"Unknown integrator {self.integrator!r}. Expected 'nvt_langevin' or 'nve'."
            )

        energy_fn = model.energy_fn_template(params)
        # Cache energy_fn so _potential_energy doesn't rebuild it each call.
        self._energy_fn = energy_fn

        shift_fn = model.ml_model.shift

        if self.integrator == "nvt_langevin":
            self._init_fn, _step_fn = simulate.nvt_langevin(
                energy_fn, shift_fn, self.dt, self.kT, self.gamma
            )
        else:
            self._init_fn, _step_fn = simulate.nve(energy_fn, shift_fn, self.dt)

        self._step_fn = jax.jit(_step_fn)

        if isinstance(self.mass, list):
            mass_str = "[" + ", ".join(f"{m:.3f}" for m in self.mass) + "] amu (per-species)"
        else:
            mass_str = f"{self.mass:.3f} amu"
        md_logger.info(
            f"MDRunner: integrator={self.integrator} n_steps={self.n_steps} "
            f"dt={self.dt:.5f} AKMA kT={self.kT:.4f} kcal/mol mass={mass_str}"
        )
        if self.integrator == "nvt_langevin":
            md_logger.info(f"  gamma={self.gamma:.6f} AKMA^-1 (τ ≈ {1/self.gamma:.1f} AKMA)")
        if self.equilibrate and self.n_equil_steps > 0:
            md_logger.info(f"  equilibration: {self.n_equil_steps} steps (not recorded)")
        md_logger.info(f"  zero_com_velocity: {self.zero_com_velocity}")
        md_logger.info(
            f"  observables: {self.observables}  every {self.observables_every} steps"
        )
        if self.force_decomp:
            md_logger.info(
                f"  force_decomp: ON  every {self.force_decomp_every} steps  "
                f"(E_ml/E_prior + F_ml/F_prior saved to trajectory NPZ)"
            )
        if self.continuous_output:
            if self.partial_output_path is None:
                md_logger.warning(
                    "  continuous_output requested, but no partial output path was provided."
                )
            else:
                md_logger.info(
                    f"  continuous output: {self.partial_output_path}  "
                    f"every {self.continuous_output_every} trajectory frame(s)"
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        R0: jax.Array,
        mask: jax.Array,
        species: jax.Array,
        rng_key: jax.Array,
    ) -> Dict[str, np.ndarray]:
        """Run equilibration (optional) + production and return trajectory + observables.

        Args:
            R0:      Initial positions, shape (N, 3).
            mask:    Validity mask, shape (N,).  1 = real atom, 0 = padding.
            species: Species IDs, shape (N,).
            rng_key: JAX PRNGKey for momentum initialisation.

        Returns:
            Dict with trajectory keys (R, F, T, KE, PE, step, box, species, mask)
            and observable keys prefixed with 'obs_' (obs_step, obs_T, obs_KE,
            obs_PE, obs_E_total, obs_pressure, obs_box_volume — subset determined
            by config 'observables' list).
        """
        valid_mask = jnp.asarray(mask > 0, dtype=jnp.bool_)
        n_valid    = int(jnp.sum(valid_mask))
        n_dof      = 3 * n_valid - 3
        if isinstance(self.mass, list):
            mass_table = jnp.array(self.mass, dtype=jnp.float32)  # (n_species,)
            mass_arr   = mass_table[species][..., None]            # (N, 1)
        else:
            mass_arr   = jnp.full((R0.shape[0], 1), self.mass)

        ml   = self.model.ml_model
        nbrs = ml.nbrs_init

        # ── Compile ────────────────────────────────────────────────────
        md_logger.info("Compiling integrator (first step may be slow) …")
        t0    = time.perf_counter()
        state = self._init_fn(
            rng_key, R0, mass=mass_arr,
            neighbor=nbrs, mask=valid_mask, species=species,
        )
        nbrs  = ml.nneigh_fn.update(state.position, nbrs, mask=valid_mask)
        state = self._step_fn(state, neighbor=nbrs, mask=valid_mask, species=species)
        jax.block_until_ready(state.position)
        md_logger.info(f"  compilation + first step: {time.perf_counter() - t0:.1f} s")

        # Re-init so we start fresh.
        rng_key, subkey = jax.random.split(rng_key)
        state = self._init_fn(
            subkey, R0, mass=mass_arr,
            neighbor=nbrs, mask=valid_mask, species=species,
        )

        # ── Equilibration (optional) ────────────────────────────────────
        if self.equilibrate and self.n_equil_steps > 0:
            md_logger.info(f"Equilibrating for {self.n_equil_steps} steps …")
            t_eq = time.perf_counter()
            for _ in range(self.n_equil_steps):
                nbrs  = ml.nneigh_fn.update(state.position, nbrs, mask=valid_mask)
                state = self._step_fn(state, neighbor=nbrs, mask=valid_mask, species=species)
                if self.zero_com_velocity:
                    com_p = (
                        jnp.sum(jnp.where(valid_mask[:, None], state.momentum, 0.0), axis=0)
                        / n_valid
                    )
                    state = state.set(
                        momentum=jnp.where(
                            valid_mask[:, None],
                            state.momentum - com_p[None, :],
                            state.momentum,
                        )
                    )
            jax.block_until_ready(state.position)
            md_logger.info(
                f"  equilibration done in {time.perf_counter() - t_eq:.1f} s"
            )

        # ── Pre-allocate trajectory arrays ──────────────────────────────
        n_traj_frames = self.n_steps // self.output_every + 1
        out_R    = np.zeros((n_traj_frames, R0.shape[0], 3), dtype=np.float32)
        out_F    = np.zeros_like(out_R)
        out_KE   = np.zeros(n_traj_frames, dtype=np.float32)
        out_PE   = np.zeros(n_traj_frames, dtype=np.float32)
        out_T    = np.zeros(n_traj_frames, dtype=np.float32)
        out_step = np.zeros(n_traj_frames, dtype=np.int32)

        # ── Force/energy decomposition arrays (optional) ─────────────────
        use_priors = self.model.use_priors
        if self.force_decomp:
            n_decomp_frames = self.n_steps // self.force_decomp_every + 1
            out_E_ml    = np.zeros(n_decomp_frames, dtype=np.float32)
            out_E_prior = np.zeros(n_decomp_frames, dtype=np.float32)
            out_F_ml    = np.zeros((n_decomp_frames, R0.shape[0], 3), dtype=np.float32)
            out_F_prior = np.zeros((n_decomp_frames, R0.shape[0], 3), dtype=np.float32)
            out_decomp_step = np.zeros(n_decomp_frames, dtype=np.int32)

        # ── Pre-allocate observable arrays ──────────────────────────────
        n_obs_frames  = self.n_steps // self.observables_every + 1
        obs_buf: Dict[str, np.ndarray] = {
            k: np.zeros(n_obs_frames, dtype=np.float32) for k in _ALL_OBSERVABLES
        }
        # step is integer
        obs_buf["step"] = np.zeros(n_obs_frames, dtype=np.int32)

        # ── Record helpers ──────────────────────────────────────────────
        def _ke(s):
            p2 = jnp.sum(s.momentum ** 2, axis=-1)
            return float(np.asarray(
                jnp.sum(jnp.asarray(0.5, dtype=p2.dtype) * p2 / s.mass[..., 0])
            ))

        def _pe(s, nb):
            return float(np.asarray(
                self._energy_fn(s.position, nb, mask=valid_mask, species=species)
            ))

        def _record_traj(idx, step, s, nb):
            ke = _ke(s)
            pe = _pe(s, nb)
            out_R[idx]    = np.asarray(s.position)
            out_F[idx]    = np.asarray(s.force)
            out_KE[idx]   = ke
            out_PE[idx]   = pe
            out_T[idx]    = 2.0 * ke / (n_dof * _kB_KCAL)
            out_step[idx] = step

        def _record_obs(idx, step, s, nb):
            ke = _ke(s)
            pe = _pe(s, nb)
            obs_buf["step"][idx]       = step
            obs_buf["KE"][idx]         = ke
            obs_buf["PE"][idx]         = pe
            obs_buf["E_total"][idx]    = ke + pe
            obs_buf["T"][idx]          = 2.0 * ke / (n_dof * _kB_KCAL)
            if "pressure" in self.observables or "box_volume" in self.observables:
                R_v = np.asarray(s.position)[np.asarray(valid_mask)]
                F_v = np.asarray(s.force)[np.asarray(valid_mask)]
                vol = self._box_volume_np(R_v)
                obs_buf["box_volume"][idx] = vol
                if "pressure" in self.observables:
                    virial = float(np.sum(R_v * F_v))
                    obs_buf["pressure"][idx] = (2.0 * ke + virial) / (3.0 * max(vol, 1e-10))

        def _record_decomp(idx, step, s, nb):
            """Compute and store per-component energies and forces via vjp."""
            R_now = jnp.asarray(s.position)

            # ── energy components ─────────────────────────────────────────
            comps = self.model.compute_components(
                self.params, R_now, valid_mask, species, neighbor=nb
            )
            e_ml    = float(np.asarray(comps.get("E_ml", 0.0)))
            e_prior = float(np.asarray(comps.get("E_prior_total", 0.0)))

            # ── force components via vjp (one forward pass, two backward) ─
            if use_priors:
                def _both_energies(R_):
                    c = self.model.compute_components(
                        self.params, R_, valid_mask, species, neighbor=nb
                    )
                    return c["E_ml"], c["E_prior_total"]
                _, vjp_fn = jax.vjp(_both_energies, R_now)
                f_ml    = np.asarray(-vjp_fn((1.0, 0.0))[0], dtype=np.float32)
                f_prior = np.asarray(-vjp_fn((0.0, 1.0))[0], dtype=np.float32)
            else:
                def _ml_energy(R_):
                    c = self.model.compute_components(
                        self.params, R_, valid_mask, species, neighbor=nb
                    )
                    return c["E_ml"]
                f_ml    = np.asarray(-jax.grad(_ml_energy)(R_now), dtype=np.float32)
                f_prior = np.zeros_like(f_ml)

            out_E_ml[idx]    = e_ml
            out_E_prior[idx] = e_prior
            out_F_ml[idx]    = f_ml
            out_F_prior[idx] = f_prior
            out_decomp_step[idx] = step

            # Log a brief summary for the first and then every 10th decomp frame.
            if idx == 0 or idx % 10 == 0:
                f_ml_rms    = float(np.sqrt(np.mean(f_ml[np.asarray(valid_mask)] ** 2)))
                f_prior_rms = float(np.sqrt(np.mean(f_prior[np.asarray(valid_mask)] ** 2))) if use_priors else 0.0
                md_logger.info(
                    f"  [decomp step={step}]  E_ml={e_ml:.3f}  E_prior={e_prior:.3f} kcal/mol"
                    f"  |F_ml|_rms={f_ml_rms:.3f}  |F_prior|_rms={f_prior_rms:.3f} kcal/mol/Å"
                )

        # ── Box (static for free-space) ─────────────────────────────────
        try:
            ref = np.asarray(ml.nbrs_init.reference_position)
            box = ref.max(axis=0) - ref.min(axis=0)
        except Exception:
            box = np.zeros(3, dtype=np.float32)

        def _build_traj(traj_count, obs_count, decomp_count=0, interrupted=False):
            traj = {
                "R":       out_R[:traj_count],
                "F":       out_F[:traj_count],
                "KE":      out_KE[:traj_count],
                "PE":      out_PE[:traj_count],
                "T":       out_T[:traj_count],
                "step":    out_step[:traj_count],
                "box":     box,
                "species": np.asarray(species, dtype=np.int32),
                "mask":    np.asarray(mask, dtype=np.float32),
                "complete": np.asarray(not interrupted, dtype=np.bool_),
            }
            if traj_count > 0:
                traj["last_step"] = np.asarray(out_step[traj_count - 1], dtype=np.int32)

            # Only include requested observables, prefixed with 'obs_'.
            for key in self.observables:
                if key in obs_buf:
                    traj[f"obs_{key}"] = obs_buf[key][:obs_count]

            # Force/energy decomposition (if enabled).
            if self.force_decomp and decomp_count > 0:
                traj["decomp_step"]    = out_decomp_step[:decomp_count]
                traj["decomp_E_ml"]    = out_E_ml[:decomp_count]
                traj["decomp_E_prior"] = out_E_prior[:decomp_count]
                traj["decomp_F_ml"]    = out_F_ml[:decomp_count]
                traj["decomp_F_prior"] = out_F_prior[:decomp_count]
            return traj

        def _write_partial(traj_count, obs_count, decomp_count=0, interrupted=False):
            if (
                not self.continuous_output
                or self.partial_output_path is None
                or traj_count <= 0
            ):
                return
            self.partial_output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.partial_output_path.with_name(
                self.partial_output_path.name + ".tmp.npz"
            )
            np.savez(
                str(tmp_path),
                **_build_traj(traj_count, obs_count, decomp_count, interrupted=interrupted),
            )
            os.replace(tmp_path, self.partial_output_path)

        # ── Production loop ─────────────────────────────────────────────
        # Record step 0.
        _record_traj(0, 0, state, nbrs)
        _record_obs(0, 0, state, nbrs)
        traj_idx   = 1
        obs_idx    = 1
        decomp_idx = 0
        if self.force_decomp:
            _record_decomp(0, 0, state, nbrs)
            decomp_idx = 1
        _write_partial(traj_idx, obs_idx, decomp_idx)

        t_loop = time.perf_counter()
        try:
            for step in range(1, self.n_steps + 1):
                nbrs  = ml.nneigh_fn.update(state.position, nbrs, mask=valid_mask)
                state = self._step_fn(state, neighbor=nbrs, mask=valid_mask, species=species)
                if self.zero_com_velocity:
                    com_p = (
                        jnp.sum(jnp.where(valid_mask[:, None], state.momentum, 0.0), axis=0)
                        / n_valid
                    )
                    state = state.set(
                        momentum=jnp.where(
                            valid_mask[:, None],
                            state.momentum - com_p[None, :],
                            state.momentum,
                        )
                    )

                wrote_traj = False
                if step % self.output_every == 0:
                    _record_traj(traj_idx, step, state, nbrs)
                    traj_idx += 1
                    wrote_traj = True

                if step % self.observables_every == 0:
                    _record_obs(obs_idx, step, state, nbrs)
                    obs_idx += 1

                if self.force_decomp and step % self.force_decomp_every == 0:
                    _record_decomp(decomp_idx, step, state, nbrs)
                    decomp_idx += 1

                if (
                    wrote_traj
                    and (traj_idx - 1) % self.continuous_output_every == 0
                ):
                    _write_partial(traj_idx, obs_idx, decomp_idx)
        except KeyboardInterrupt:
            jax.block_until_ready(state.position)
            _write_partial(traj_idx, obs_idx, decomp_idx, interrupted=True)
            if self.partial_output_path is not None:
                md_logger.info(
                    f"MD interrupted; partial trajectory saved: {self.partial_output_path} "
                    f"(frames={traj_idx}, last_step={int(out_step[traj_idx - 1])})"
                )
            raise

        elapsed = time.perf_counter() - t_loop
        md_logger.info(
            f"MD complete: {self.n_steps} steps in {elapsed:.1f} s "
            f"({elapsed / self.n_steps * 1000:.2f} ms/step)  "
            f"T_final={obs_buf['T'][obs_idx-1]:.1f} K  "
            f"PE_final={obs_buf['PE'][obs_idx-1]:.3f} kcal/mol"
        )
        if self.force_decomp and decomp_idx > 0:
            md_logger.info(
                f"  decomp: {decomp_idx} frames  "
                f"E_ml_mean={float(np.mean(out_E_ml[:decomp_idx])):.3f}  "
                f"E_prior_mean={float(np.mean(out_E_prior[:decomp_idx])):.3f} kcal/mol"
            )

        traj = _build_traj(traj_idx, obs_idx, decomp_idx, interrupted=False)
        _write_partial(traj_idx, obs_idx, decomp_idx, interrupted=False)
        return traj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _box_volume_np(R_valid: np.ndarray) -> float:
        """Bounding-box volume of valid atom positions (sss convention)."""
        if R_valid.shape[0] == 0:
            return 1.0
        lo  = R_valid.min(axis=0)
        hi  = R_valid.max(axis=0)
        ext = hi - lo
        # Guard against degenerate dimensions (all atoms in a plane).
        return float(np.prod(np.where(ext > 0, ext, 1.0)))
