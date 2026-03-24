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
