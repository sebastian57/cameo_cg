"""
Shared JAX setup utilities.

Must be imported and called BEFORE any JAX-dependent module imports
in scripts that use jax_md or chemtrain.
"""

import os
import logging
import jax


def apply_jax_compat_shims():
    """Runtime compatibility shims for jax_md/chemtrain with newer JAX releases.

    No-ops on older JAX versions where the patched symbols still exist natively.
    Safe to call multiple times.
    """
    if not hasattr(jax.random, "KeyArray"):
        jax.random.KeyArray = jax.Array

    for name in ("tree_map", "tree_leaves", "tree_flatten", "tree_unflatten"):
        if not hasattr(jax, name):
            setattr(jax, name, getattr(jax.tree_util, name))

    if not hasattr(jax.lib, "xla_bridge"):
        from jax._src import xla_bridge as _xla_bridge
        jax.lib.xla_bridge = _xla_bridge


def apply_numpy_dataloader_patch():
    """Patch NumpyDataLoader so ``cache_size`` is never 0 (chemtrain bug)."""
    from jax_sgmc.data.numpy_loader import NumpyDataLoader as _NDL

    _orig_get_indices = _NDL._get_indices

    def _patched_get_indices(self, chain_id: int):
        chain = self._chains[chain_id]
        if chain.get("cache_size", 0) <= 0:
            chain["cache_size"] = 1
        return _orig_get_indices(self, chain_id)

    _NDL._get_indices = _patched_get_indices
    logging.info("[Patch] Applied NumpyDataLoader cache_size fix")


def assert_gpu_when_allocated(context: str = "job") -> None:
    """Fail fast if a SLURM job with an allocated GPU is silently running on CPU.

    On this cluster a faulty node can return ``cuInit(0) failed: CUDA_ERROR_UNKNOWN``.
    JAX then falls back to CPU and the job runs ~20x slower to the wall limit, producing
    nothing, with only a traceback buried in stdout to show for it. Observed twice on
    2026-07-31 (jobs 1137428_0 and 1139036, both on jpbo-001-48) at a cost of ~10 GPU-hours.

    Raises RuntimeError when running under SLURM with a GPU in the allocation but no GPU
    device visible to JAX. Outside SLURM, or when the user explicitly asked for CPU via
    JAX_PLATFORMS, this is a no-op.
    """
    if not os.environ.get("SLURM_JOB_ID"):
        return
    if "cpu" in os.environ.get("JAX_PLATFORMS", "").lower():
        return  # explicit opt-in to CPU
    gres = (os.environ.get("SLURM_JOB_GRES", "")
            or os.environ.get("SLURM_STEP_GRES", "")
            or os.environ.get("SLURM_GPUS_PER_TASK", "")
            or os.environ.get("SLURM_GPUS", ""))
    if "gpu" not in gres.lower() and not gres.strip().isdigit():
        return  # no GPU was requested; CPU is legitimate

    platforms = {d.platform for d in jax.devices()}
    if not platforms & {"gpu", "cuda", "rocm"}:
        raise RuntimeError(
            f"[{context}] SLURM allocated a GPU (SLURM_JOB_GRES={gres!r}) but JAX sees only "
            f"{sorted(platforms)} on node {os.environ.get('SLURMD_NODENAME', '?')}.\n"
            "This is the silent CUDA-init fallback: the run would proceed on CPU at roughly "
            "1/20th speed and hit the wall limit having produced nothing.\n"
            "Check the log above for 'cuInit(0) failed'. Resubmit excluding this node:\n"
            f"    sbatch --exclude={os.environ.get('SLURMD_NODENAME', '<node>')} ...\n"
            "Set JAX_PLATFORMS=cpu to run on CPU deliberately."
        )
    logging.info("[jax_setup] GPU confirmed: %s", sorted(platforms))
