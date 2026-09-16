"""Batched CG MD: vmap over replicas, scan over steps.

WHY
---
`md/runner.py` advances ONE replica per process, so R replicas means R OS
processes, each with its own JAX runtime, ~900-thread pool, its own compile
(measured 108-280 s), and its own host dispatch loop, all contending for one
GPU through MPS.

Measured on ala2 bb6 (v4 model, 2026-08-25), force evaluations per second:

    replicas      1       4      16      64     256
    vmap       1023    2811    7363   11579   13006      <- 11.3x at R=64
    tiled      1010    2823    6378   10210   11889      <- consistently ~10% worse

A single 6-bead force call costs 0.977 ms; at R=64 the same call serves 64
replicas in 5.5 ms, i.e. 0.086 ms each. The bottleneck is kernel-launch
overhead on a network far larger than the system, which batching amortises.
Saturation is at R~64; R=256 buys ~10% more for 4x the batch.

EQUIVALENCE
-----------
This mirrors MDRunner's initialisation EXACTLY, because anything else silently
changes the trajectory:
  * warm-up init+step, THEN `jax.random.split`, and the real init uses SUBKEY
  * mass-weighted COM removal (not an unweighted mean)
  * optional initial-temperature rescale with n_dof = 3*n_valid - 3
Verified against MDRunner by `scripts/test_vmap_equivalence.py`.

UNITS: md_config must already be AKMA (post `run_md.py::to_akma`), same
contract as MDRunner. A guard rejects obviously-unconverted input.
"""
from __future__ import annotations

import logging
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import simulate

md_logger = logging.getLogger("md")


class BatchedMDRunner:
    def __init__(self, model, params, md_config: dict):
        c = md_config
        self.integrator = str(c.get("integrator", "nvt_langevin"))
        if self.integrator != "nvt_langevin":
            raise NotImplementedError(
                f"BatchedMDRunner supports nvt_langevin only (got '{self.integrator}'); "
                "use md/runner.py otherwise.")
        if c.get("force_decomp", False):
            raise ValueError("force_decomp is incompatible with scan (vjp inside scan).")

        self.dt = float(c.get("dt", 0.02045))
        self.kT = float(c.get("kT", 0.5961))
        self.gamma = float(c.get("gamma", 0.000977))
        if self.dt > 0.5 or self.kT > 10.0:
            raise ValueError(
                f"md_config is not in AKMA units (dt={self.dt}, kT={self.kT}); expected "
                f"dt~0.02-0.09, kT~0.6. Pass it through run_md.py::to_akma() first.")

        self.n_steps = int(c["n_steps"])
        self.output_every = int(c.get("output_every", 500))
        self.chunk = int(c.get("scan_chunk_size", 5000) or 5000)
        self.zero_com = bool(c.get("zero_com_velocity", True))
        # NOTE: MDRunner defaults this to False (md/runner.py:96). Matching that
        # default is required for equivalence -- True here silently rescales
        # initial momenta and every trajectory diverges from the reference.
        self.rescale_T = bool(c.get("rescale_initial_temperature", False))
        if c.get("equilibrate", False):
            raise NotImplementedError("equilibrate is not implemented in the batched route.")
        # Continuous output is written at CHUNK boundaries, where `np.asarray(pos)` already
        # forces the device->host sync, so it adds no synchronisation -- only the bytes of the
        # new slice. This is deliberately NOT md/runner.py's scheme, which re-serialises the
        # whole trajectory from frame 0 on every flush (O(N^2/k) bytes); that quadratic
        # amplification is what made continuous output look expensive.
        self.continuous_output = bool(c.get("continuous_output", False))
        self.continuous_output_every = max(1, int(c.get("continuous_output_every", 10)))
        self.T_scale = float(c.get("initial_temperature_scale", 1.0))
        self.mass = c.get("mass", 1.0)
        if self.n_steps % self.chunk:
            raise ValueError(f"n_steps ({self.n_steps}) must be divisible by "
                             f"scan_chunk_size ({self.chunk}).")
        if self.chunk % self.output_every:
            raise ValueError(f"scan_chunk_size ({self.chunk}) must be divisible by "
                             f"output_every ({self.output_every}).")

        self.model, self.params = model, params
        self._ml = model.ml_model
        efn = model.energy_fn_template(params)
        self._energy_fn = efn
        self._init_fn, self._step_fn = simulate.nvt_langevin(
            efn, self._ml.shift, self.dt, self.kT, self.gamma)

    # ------------------------------------------------------------------
    def run(self, R0, mask, species, base_seed: int, n_replicas: int,
            out_stem: Path, frame_indices: Optional[Sequence[int]] = None) -> dict:
        ml = self._ml
        valid_mask = jnp.asarray(np.asarray(mask) > 0, dtype=jnp.bool_)
        n_valid = int(jnp.sum(valid_mask))
        n_dof = 3 * n_valid - 3
        species = jnp.asarray(species)

        R0 = np.asarray(R0, np.float32)
        R0b = jnp.asarray(np.repeat(R0[None], n_replicas, 0) if R0.ndim == 2 else R0)
        if R0b.shape[0] != n_replicas:
            raise ValueError(f"{R0b.shape[0]} start frames for {n_replicas} replicas.")

        if isinstance(self.mass, list):
            mass_arr = jnp.array(self.mass, dtype=jnp.float32)[species][..., None]
        else:
            mass_arr = jnp.full((R0b.shape[1], 1), float(self.mass))
        total_mass = jnp.sum(jnp.where(valid_mask[:, None], mass_arr, 0.0))

        def _com(s):                                    # mirrors runner.py:298
            total_p = jnp.sum(jnp.where(valid_mask[:, None], s.momentum, 0.0), axis=0)
            v = total_p / total_mass
            return s.set(momentum=jnp.where(
                valid_mask[:, None], s.momentum - s.mass * v[None, :], s.momentum))

        def _rescale(s):                                # mirrors runner.py:311
            p2 = jnp.sum(s.momentum ** 2, axis=-1)
            ke = jnp.sum(jnp.where(valid_mask, 0.5 * p2 / s.mass[..., 0], 0.0))
            tgt = 0.5 * n_dof * (self.kT * self.T_scale)
            sc = jnp.where((ke > 0.0) & (tgt > 0.0), jnp.sqrt(tgt / ke), 0.0)
            return s.set(momentum=jnp.where(valid_mask[:, None], s.momentum * sc, 0.0))

        def _init_one(key, R):
            # EXACT MDRunner order: warm init, split, real init from SUBKEY.
            _ = self._init_fn(key, R, mass=mass_arr, neighbor=ml.nbrs_init,
                              mask=valid_mask, species=species)
            _, sub = jax.random.split(key)
            s = self._init_fn(sub, R, mass=mass_arr, neighbor=ml.nbrs_init,
                              mask=valid_mask, species=species)
            if self.zero_com:
                s = _com(s)
            if self.rescale_T:
                s = _rescale(s)
            return s

        keys = jnp.stack([jax.random.PRNGKey(base_seed + i) for i in range(n_replicas)])
        t0 = time.perf_counter()
        state = jax.vmap(_init_one)(keys, R0b)
        nbrs = jax.vmap(lambda R: ml.nneigh_fn.update(R, ml.nbrs_init, mask=valid_mask))(
            state.position)

        def _body(carry, _):
            s, nb = carry
            nb = jax.vmap(lambda R, n: ml.nneigh_fn.update(R, n, mask=valid_mask))(
                s.position, nb)
            s = jax.vmap(lambda st, n: self._step_fn(
                st, neighbor=n, mask=valid_mask, species=species))(s, nb)
            if self.zero_com:
                s = jax.vmap(_com)(s)
            return (s, nb), s.position

        @jax.jit
        def _chunk(s, nb):
            (s, nb), pos = jax.lax.scan(_body, (s, nb), None, length=self.chunk)
            return (s, nb), pos[self.output_every - 1::self.output_every]

        n_chunks = self.n_steps // self.chunk
        per = self.chunk // self.output_every
        # Frame 0 is the INITIAL structure, matching md/runner.py:491 (
        # n_steps//output_every + 1) and :697 (_record_traj(0, 0, ...)).
        # Without it the batched route is one frame short and every downstream
        # comparison/analysis silently misaligns.
        out_stem = Path(out_stem)
        out_stem.parent.mkdir(parents=True, exist_ok=True)
        shape = (n_chunks * per + 1, n_replicas, R0b.shape[1], 3)
        partial = out_stem.with_name(out_stem.stem + ".partial.npy") if self.continuous_output else None
        if partial is not None:
            from numpy.lib.format import open_memmap
            out = open_memmap(partial, mode="w+", dtype=np.float32, shape=shape)
            md_logger.info(f"  continuous output -> {partial} (flush every "
                           f"{self.continuous_output_every} chunk(s))")
        else:
            out = np.zeros(shape, np.float32)
        out[0] = np.asarray(state.position)

        def _flush(n_frames_written):
            """Persist what exists so far. O(new bytes), not O(trajectory)."""
            if partial is None:
                return
            out.flush()
            partial.with_suffix(".json").write_text(json.dumps(
                {"frames_written": int(n_frames_written), "frames_total": int(shape[0]),
                 "n_replicas": int(n_replicas), "output_every": int(self.output_every),
                 "complete": bool(n_frames_written >= shape[0])}) + "\n")

        _flush(1)
        md_logger.info(
            f"BatchedMDRunner: {n_replicas} replicas x {self.n_steps} steps, "
            f"chunk={self.chunk} -> {n_chunks} dispatches for ALL replicas "
            f"(per-process route would be {self.n_steps * n_replicas})")
        for ci in range(n_chunks):
            (state, nbrs), pos = _chunk(state, nbrs)
            out[1 + ci * per:1 + (ci + 1) * per] = np.asarray(pos)
            if self.continuous_output and (ci + 1) % self.continuous_output_every == 0:
                _flush(1 + (ci + 1) * per)
            if n_chunks >= 10 and (ci + 1) % (n_chunks // 10) == 0:
                md_logger.info(f"  chunk {ci+1}/{n_chunks}  ({time.perf_counter()-t0:.0f} s)")
        md_logger.info(f"  batched MD complete in {time.perf_counter()-t0:.1f} s")

        _flush(shape[0])
        steps = np.arange(out.shape[0], dtype=np.int64) * self.output_every
        files = []
        for i in range(n_replicas):
            p = out_stem.with_name(f"{out_stem.stem}_rep{i:02d}.npz")
            np.savez_compressed(
                p, R=out[:, i], step=steps, species=np.asarray(species),
                mask=np.asarray(mask), seed=base_seed + i,
                initial_frame_idx=(-1 if not frame_indices else
                                   frame_indices[min(i, len(frame_indices) - 1)]))
            files.append(str(p))
        n_out = int(out.shape[0])
        if partial is not None:
            del out                                  # release the memmap before unlinking
            partial.unlink(missing_ok=True)
            partial.with_suffix(".json").unlink(missing_ok=True)
        md_logger.info(f"wrote {len(files)} trajectories, {n_out} frames each")
        return {"files": files, "n_frames": n_out}
