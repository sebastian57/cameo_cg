"""JAX-MD simulation runner for trained CG protein force fields.

Runs NVT-Langevin or NVE dynamics using a CombinedModel loaded from PKL params.
All quantities in AKMA units (Å, kcal/mol, amu); 1 AKMA time unit ≈ 48.88 fs.
"""

import time
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import simulate

from models.combined_model import CombinedModel
from utils.logging import md_logger

# Boltzmann constant in kcal/mol/K (AKMA unit system)
_kB_KCAL = 1.9872041e-3


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

        self.integrator   = str(md_config.get("integrator", "nvt_langevin"))
        self.n_steps      = int(md_config.get("n_steps", 1000))
        self.dt           = float(md_config.get("dt", 0.02045))
        self.kT           = float(md_config.get("kT", 0.5961))
        self.gamma        = float(md_config.get("gamma", 0.000977))
        self.mass         = float(md_config.get("mass", 12.011))
        self.output_every = int(md_config.get("output_every", 10))

        if self.integrator not in ("nvt_langevin", "nve"):
            raise ValueError(
                f"Unknown integrator {self.integrator!r}. Expected 'nvt_langevin' or 'nve'."
            )

        energy_fn = model.energy_fn_template(params)

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
        """Run the simulation and return a trajectory dictionary.

        Args:
            R0:      Initial positions, shape (N, 3).
            mask:    Validity mask, shape (N,).  1 = real atom, 0 = padding.
            species: Species IDs, shape (N,).
            rng_key: JAX PRNGKey for momentum initialisation (NVT) or NVE.

        Returns:
            Dict with keys:
              R       — positions          (n_frames, N, 3)
              F       — forces             (n_frames, N, 3)
              KE      — kinetic energy     (n_frames,)
              PE      — potential energy   (n_frames,)
              T       — temperature in K   (n_frames,)
              step    — step indices       (n_frames,)
              box     — simulation box     (3,)
              species — species IDs (0-indexed, constant across frames)  (N,)
              mask    — validity mask      (N,)
        """
        valid_mask = jnp.asarray(mask > 0, dtype=jnp.bool_)
        n_valid    = int(jnp.sum(valid_mask))
        n_dof      = 3 * n_valid - 3         # 3D, subtract COM translation
        mass_arr   = jnp.full((R0.shape[0], 1), self.mass)

        ml = self.model.ml_model   # AllegroModelCuEq (or equivalent)
        nbrs = ml.nbrs_init

        # Warm-up: init state and JIT-compile.
        md_logger.info("Compiling integrator (first step may be slow) …")
        t0 = time.perf_counter()
        state = self._init_fn(
            rng_key, R0,
            mass=mass_arr,
            neighbor=nbrs,
            mask=valid_mask,
            species=species,
        )
        # Force JIT compilation now, before the timed loop.
        nbrs = ml.nneigh_fn.update(state.position, nbrs, mask=valid_mask)
        state = self._step_fn(state, neighbor=nbrs, mask=valid_mask, species=species)
        jax.block_until_ready(state.position)
        md_logger.info(f"  compilation + first step: {time.perf_counter() - t0:.1f} s")

        # Re-init from scratch so step 0 is genuinely the initial frame.
        rng_key, subkey = jax.random.split(rng_key)
        state = self._init_fn(
            subkey, R0,
            mass=mass_arr,
            neighbor=nbrs,
            mask=valid_mask,
            species=species,
        )

        # Pre-allocate output lists.
        n_frames  = self.n_steps // self.output_every + 1
        out_R  = np.zeros((n_frames, R0.shape[0], 3), dtype=np.float32)
        out_F  = np.zeros_like(out_R)
        out_KE = np.zeros(n_frames, dtype=np.float32)
        out_PE = np.zeros(n_frames, dtype=np.float32)
        out_T  = np.zeros(n_frames, dtype=np.float32)
        out_step = np.zeros(n_frames, dtype=np.int32)

        def _record(idx, step, s, nbrs_):
            KE = float(np.asarray(self._kinetic_energy(s)))
            PE = float(np.asarray(self._potential_energy(s, nbrs_, valid_mask, species)))
            T  = 2.0 * KE / (n_dof * _kB_KCAL)
            out_R[idx]    = np.asarray(s.position)
            out_F[idx]    = np.asarray(s.force)
            out_KE[idx]   = KE
            out_PE[idx]   = PE
            out_T[idx]    = T
            out_step[idx] = step

        # Record frame 0 (initial state — before any steps).
        _record(0, 0, state, nbrs)
        frame_idx = 1

        t_loop = time.perf_counter()
        for step in range(1, self.n_steps + 1):
            nbrs  = ml.nneigh_fn.update(state.position, nbrs, mask=valid_mask)
            state = self._step_fn(state, neighbor=nbrs, mask=valid_mask, species=species)

            if step % self.output_every == 0:
                _record(frame_idx, step, state, nbrs)
                frame_idx += 1

        elapsed = time.perf_counter() - t_loop
        md_logger.info(
            f"MD complete: {self.n_steps} steps in {elapsed:.1f} s "
            f"({elapsed / self.n_steps * 1000:.2f} ms/step), "
            f"T_final={out_T[frame_idx - 1]:.1f} K  "
            f"PE_final={out_PE[frame_idx - 1]:.3f} kcal/mol"
        )

        # Box recorded from the model's neighbor list setup (constant for free space).
        try:
            ref = np.asarray(ml.nbrs_init.reference_position)
            box = ref.max(axis=0) - ref.min(axis=0)
        except Exception:
            box = np.zeros(3, dtype=np.float32)

        return {
            "R":       out_R[:frame_idx],
            "F":       out_F[:frame_idx],
            "KE":      out_KE[:frame_idx],
            "PE":      out_PE[:frame_idx],
            "T":       out_T[:frame_idx],
            "step":    out_step[:frame_idx],
            "box":     box,
            "species": np.asarray(species, dtype=np.int32),
            "mask":    np.asarray(mask, dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _kinetic_energy(self, state) -> jax.Array:
        """KE = sum_i p_i^2 / (2 m_i)."""
        p2 = jnp.sum(state.momentum ** 2, axis=-1)         # (N,)
        return jnp.sum(jnp.asarray(0.5, dtype=p2.dtype) * p2 / state.mass[..., 0])

    def _potential_energy(self, state, nbrs, mask, species) -> jax.Array:
        """One energy evaluation at the current positions."""
        energy_fn = self.model.energy_fn_template(self.params)
        return energy_fn(state.position, nbrs, mask=mask, species=species)
