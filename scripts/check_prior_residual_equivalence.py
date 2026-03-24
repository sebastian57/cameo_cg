#!/usr/bin/env python3
"""
Check equivalence of baseline vs residual-prior force objectives.

Validates:
    ||(F_ml + F_prior) - F_ref||^2 == ||F_ml - (F_ref - F_prior)||^2
on a small fixed frame subset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def _apply_jax_compat_shims() -> None:
    if not hasattr(jax.random, "KeyArray"):
        jax.random.KeyArray = jax.Array
    if not hasattr(jax, "tree_map"):
        jax.tree_map = jax.tree_util.tree_map
    if not hasattr(jax, "tree_leaves"):
        jax.tree_leaves = jax.tree_util.tree_leaves
    if not hasattr(jax, "tree_flatten"):
        jax.tree_flatten = jax.tree_util.tree_flatten
    if not hasattr(jax, "tree_unflatten"):
        jax.tree_unflatten = jax.tree_util.tree_unflatten
    if not hasattr(jax.lib, "xla_bridge"):
        from jax._src import xla_bridge as _xla_bridge
        jax.lib.xla_bridge = _xla_bridge


_apply_jax_compat_shims()
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.manager import ConfigManager
from data.loader import DatasetLoader
from data.preprocessor import CoordinatePreprocessor
from models.combined_model import CombinedModel
from training.prior_residual import apply_prior_force_residual_targets


def _masked_mse(pred: jax.Array, target: jax.Array, mask: jax.Array) -> jax.Array:
    w = mask[..., None]
    num = jnp.sum(jnp.square(pred - target) * w)
    den = jnp.maximum(jnp.sum(w), 1.0)
    return num / den


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config.")
    parser.add_argument("--n-frames", type=int, default=4, help="Number of frames to test.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    config = ConfigManager(args.config)
    config._config.setdefault("model", {})
    config._config["model"]["use_priors"] = False
    config._config["model"]["train_priors"] = False

    config._config.setdefault("training", {})
    config._config["training"]["prior_residual"] = {
        "enabled": True,
        "cache_enabled": False,
        "cache_path": None,
        "force_recompute": True,
        "compute_batch_size": 64,
    }

    data_path = Path(config.get_data_path())
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / data_path

    loader = DatasetLoader(
        str(data_path),
        max_frames=config.get_max_frames(),
        seed=config.get_seed(),
    )
    preprocessor = CoordinatePreprocessor(
        cutoff=config.get_cutoff(),
        buffer_multiplier=config.get_buffer_multiplier(),
        park_multiplier=config.get_park_multiplier(),
    )
    extent, r_shift = preprocessor.compute_box_extent(loader.R, loader.mask)
    R_all = preprocessor.center_and_park(loader.R, loader.mask, extent, r_shift)

    n = min(int(args.n_frames), int(R_all.shape[0]))
    subset = {
        "R": np.asarray(R_all[:n], dtype=np.float32),
        "F": np.asarray(loader.F[:n], dtype=np.float32),
        "mask": np.asarray(loader.mask[:n], dtype=np.float32),
        "species": np.asarray(loader.species[:n], dtype=np.int32),
    }
    F_ref = np.asarray(subset["F"], dtype=np.float32)

    apply_prior_force_residual_targets(
        config=config,
        dataset=subset,
        dataset_path=data_path,
        dataset_tag=f"{data_path.stem}_residual_equiv",
        id_to_aa=loader.id_to_aa,
        project_root=Path(__file__).parent.parent,
        seed=config.get_seed(),
        max_frames=config.get_max_frames(),
        cutoff=config.get_cutoff(),
        buffer_multiplier=config.get_buffer_multiplier(),
        park_multiplier=config.get_park_multiplier(),
    )
    F_residual = np.asarray(subset["F"], dtype=np.float32)
    F_prior = F_ref - F_residual

    n_species = int(np.max(loader.species)) + 1
    model = CombinedModel(
        config=config,
        R0=jnp.asarray(subset["R"][0]),
        box=jnp.asarray(extent, dtype=jnp.float32),
        species=jnp.asarray(subset["species"][0]),
        N_max=int(subset["R"].shape[1]),
        n_species_override=n_species,
        id_to_aa=loader.id_to_aa,
        prior_only=False,
    )
    params = model.initialize_params(jax.random.PRNGKey(args.seed))

    R = jnp.asarray(subset["R"], dtype=jnp.float32)
    mask = jnp.asarray(subset["mask"], dtype=jnp.float32)
    species = jnp.asarray(subset["species"], dtype=jnp.int32)

    def _single_ml_force(R_i: jax.Array, m_i: jax.Array, s_i: jax.Array) -> jax.Array:
        return -jax.grad(lambda R_var: model.compute_energy(params, R_var, m_i, s_i))(R_i)

    F_ml = jax.jit(jax.vmap(_single_ml_force, in_axes=(0, 0, 0)))(R, mask, species)
    F_ml = np.asarray(F_ml, dtype=np.float32)

    baseline = float(
        _masked_mse(
            jnp.asarray(F_ml + F_prior),
            jnp.asarray(F_ref),
            jnp.asarray(subset["mask"]),
        )
    )
    residual = float(
        _masked_mse(
            jnp.asarray(F_ml),
            jnp.asarray(F_residual),
            jnp.asarray(subset["mask"]),
        )
    )
    diff = abs(baseline - residual)

    print("=== Prior Residual Equivalence ===")
    print(f"frames={n} atoms={subset['R'].shape[1]}")
    print(f"baseline_loss={baseline:.8e}")
    print(f"residual_loss={residual:.8e}")
    print(f"abs_diff={diff:.8e}")

    if not np.isclose(baseline, residual, rtol=args.rtol, atol=args.atol):
        raise SystemExit(
            "Residual equivalence check failed: "
            f"baseline={baseline:.6e} residual={residual:.6e} diff={diff:.6e}"
        )


if __name__ == "__main__":
    main()

