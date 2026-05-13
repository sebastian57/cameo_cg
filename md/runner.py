"""JAX-MD simulation runner for trained CG protein force fields.

Runs NVT-Langevin or NVE dynamics using a CombinedModel loaded from PKL params.
All quantities in AKMA units (Å, kcal/mol, amu); 1 AKMA time unit ≈ 48.88 fs.
"""

import time
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
        self.mass              = float(md_config.get("mass", 12.011))
        self.output_every      = int(md_config.get("output_every", 10))

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

        md_logger.info(
            f"MDRunner: integrator={self.integrator} n_steps={self.n_steps} "
            f"dt={self.dt:.5f} AKMA kT={self.kT:.4f} kcal/mol mass={self.mass:.3f} amu"
        )
        if self.integrator == "nvt_langevin":
            md_logger.info(f"  gamma={self.gamma:.6f} AKMA^-1 (τ ≈ {1/self.gamma:.1f} AKMA)")
        if self.equilibrate and self.n_equil_steps > 0:
            md_logger.info(f"  equilibration: {self.n_equil_steps} steps (not recorded)")
        md_logger.info(f"  zero_com_velocity: {self.zero_com_velocity}")
        md_logger.info(
            f"  observables: {self.observables}  every {self.observables_every} steps"
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

        # ── Production loop ─────────────────────────────────────────────
        # Record step 0.
        _record_traj(0, 0, state, nbrs)
        _record_obs(0, 0, state, nbrs)
        traj_idx = 1
        obs_idx  = 1

        t_loop = time.perf_counter()
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

            if step % self.output_every == 0:
                _record_traj(traj_idx, step, state, nbrs)
                traj_idx += 1

            if step % self.observables_every == 0:
                _record_obs(obs_idx, step, state, nbrs)
                obs_idx += 1

        elapsed = time.perf_counter() - t_loop
        md_logger.info(
            f"MD complete: {self.n_steps} steps in {elapsed:.1f} s "
            f"({elapsed / self.n_steps * 1000:.2f} ms/step)  "
            f"T_final={obs_buf['T'][obs_idx-1]:.1f} K  "
            f"PE_final={obs_buf['PE'][obs_idx-1]:.3f} kcal/mol"
        )

        # ── Box (static for free-space) ─────────────────────────────────
        try:
            ref = np.asarray(ml.nbrs_init.reference_position)
            box = ref.max(axis=0) - ref.min(axis=0)
        except Exception:
            box = np.zeros(3, dtype=np.float32)

        # ── Build return dict ────────────────────────────────────────────
        traj = {
            "R":       out_R[:traj_idx],
            "F":       out_F[:traj_idx],
            "KE":      out_KE[:traj_idx],
            "PE":      out_PE[:traj_idx],
            "T":       out_T[:traj_idx],
            "step":    out_step[:traj_idx],
            "box":     box,
            "species": np.asarray(species, dtype=np.int32),
            "mask":    np.asarray(mask, dtype=np.float32),
        }

        # Only include requested observables, prefixed with 'obs_'.
        for key in self.observables:
            if key in obs_buf:
                traj[f"obs_{key}"] = obs_buf[key][:obs_idx]

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
