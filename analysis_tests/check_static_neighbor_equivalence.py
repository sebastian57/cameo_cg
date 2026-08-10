#!/usr/bin/env python3
"""
Static-vs-dynamic neighbor-list equivalence check for tiled force matching.

Validates that `data.static_neighbors.enabled: true` reproduces the dynamic
JAX-MD path exactly, through the real chemtrain evaluation stack rather than by
comparing edge sets alone:

1) directed edge sets of the static graph vs `masked_neighbor_list`
2) per-tile energies via `chemtrain.learn.force_matching.init_model`
3) per-node forces from the same path
4) one optimizer update from identical parameters and data

Reference tolerance from the existing t1024 all-pairs-vs-cell-list benchmark:
max-abs 1.19e-7, relative L2 1.07e-9 (see DESIGN/STATIC_TILED_NEIGHBOR_LISTS.md).

Runs on CPU with the `allegro` backbone; the cuEq backbones share the identical
`neighbor is None / error is None` branch in `compute_energy`.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp


def _apply_jax_compat_shims() -> None:
    """Runtime compatibility shims for jax_md/chemtrain with newer JAX."""
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

import yaml

from config.manager import ConfigManager
from data.loader import build_tiled_dataset
from models.combined_model import CombinedModel

from chemtrain.learn import force_matching

CUTOFF = 5.5
DR_THRESHOLD = 1.0

# Reference targets from the t1024 all-pairs vs cell-list benchmark.
MAX_ABS_TARGET = 1.19e-7
REL_L2_TARGET = 1.07e-9


def _minimal_config(tmpdir: Path, static: bool) -> ConfigManager:
    """Smallest config that instantiates a CPU Allegro model in tiled mode."""
    cfg = {
        "data": {
            "batch_mode": "tiled",
            "static_neighbors": {"enabled": static, "backend": "kdtree"},
        },
        "model": {
            "ml_model": "allegro",
            "cutoff": CUTOFF,
            "dr_threshold": DR_THRESHOLD,
            "use_priors": False,
            "pbc": False,
            # Match the production graph format: the static shell is Sparse.
            "neighbor_list_format": "sparse",
            "neighbor_disable_cell_list": True,
            "allegro": {
                "num_layers": 1,
                "mlp_n_hidden": 16,
                "mlp_n_layers": 1,
                "max_ell": 1,
                "avg_num_neighbors": 8,
                "neighbor_disable_cell_list": True,
            },
        },
        "training": {"gammas": {"F": 1.0, "U": 0.0}},
        "optimizer": {"name": "adam", "lr": 1e-3},
    }
    path = tmpdir / f"config_static_{static}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return ConfigManager(path)


def _toy_tiled(config: ConfigManager, n_structures: int, n_atoms: int, seed: int):
    rng = np.random.RandomState(seed)
    R = rng.uniform(0.0, 6.0, size=(n_structures, n_atoms, 3)).astype(np.float32)
    F = rng.normal(size=R.shape).astype(np.float32)
    mask = np.ones(R.shape[:2], dtype=np.float32)
    species = np.ones(R.shape[:2], dtype=np.int32)
    return build_tiled_dataset(
        R,
        F,
        mask,
        species,
        target_beads=n_structures * n_atoms,
        spatial_separation=True,
        spatial_layout="grid_3d",
        structure_gap=30.0,
        static_neighbors=config.get_static_neighbors_config(),
    )


def _edge_set(idx: np.ndarray, n_atoms: int) -> set:
    receivers, senders = np.asarray(idx)
    keep = (receivers < n_atoms) & (senders < n_atoms)
    return set(zip(senders[keep].tolist(), receivers[keep].tolist()))


def _predict(model, params, tiled, static: bool):
    """Run the chemtrain force-matching model exactly as the trainer does."""
    from jax_md_mod import custom_quantity

    quantities = {
        "F": custom_quantity.force_wrapper(None),
        "U": custom_quantity.energy_wrapper(None),
    }
    feature_fns = {
        "energy_and_force": custom_quantity.energy_force_wrapper(
            model.energy_fn_template
        )
    }
    fm_model = force_matching.init_model(
        None if static else model.initial_neighbors,
        quantities,
        feature_extract_fns=feature_fns,
    )

    # `init_model` drops a quantity whose target is absent from the
    # observations, so a dummy U target is needed to keep energies predicted.
    observations = {
        "R": jnp.asarray(tiled["R"]),
        "F": jnp.asarray(tiled["F"]),
        "U": jnp.zeros((tiled["R"].shape[0],), dtype=jnp.float32),
        "mask": jnp.asarray(tiled["mask"]),
        "species": jnp.asarray(tiled["species"]),
        "segment_id": jnp.asarray(tiled["segment_id"]),
    }
    if static:
        observations["neighbor_idx"] = jnp.asarray(tiled["neighbor_idx"])
    return fm_model(params, observations)


def _compare(name: str, a: np.ndarray, b: np.ndarray) -> tuple:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    max_abs = float(np.max(np.abs(a - b))) if a.size else 0.0
    denom = float(np.linalg.norm(b))
    rel_l2 = float(np.linalg.norm(a - b) / denom) if denom > 0 else 0.0
    print(f"  {name:<28} max_abs={max_abs:.3e}  rel_L2={rel_l2:.3e}")
    return max_abs, rel_l2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-structures", type=int, default=8)
    parser.add_argument("--n-atoms", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tol", type=float, default=1e-5,
        help="Max absolute tolerance for energies/forces/params (float32 slack).",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        config_static = _minimal_config(tmpdir, static=True)
        config_dynamic = _minimal_config(tmpdir, static=False)

        tiled = _toy_tiled(config_static, args.n_structures, args.n_atoms, args.seed)
        n_tiles, n_atoms = tiled["R"].shape[0], tiled["R"].shape[1]
        print(
            f"Tiled dataset: {n_tiles} tile(s) x {n_atoms} beads, "
            f"{int(tiled['neighbor_n_edges'].max())} max directed edges, "
            f"capacity {int(tiled['neighbor_capacity'][0])}"
        )

        # ---------------------------------------------------------------- #
        # 1) Edge-set equality against the dynamic JAX-MD path
        # ---------------------------------------------------------------- #
        from jax_md_mod import custom_partition
        from jax_md import partition, space

        displacement, _ = space.free()
        box = float(np.max(np.abs(tiled["R"])) + CUTOFF + DR_THRESHOLD + 1.0)
        neighbor_fn = custom_partition.masked_neighbor_list(
            displacement,
            box=jnp.asarray(box, dtype=jnp.float32),
            r_cutoff=CUTOFF,
            dr_threshold=DR_THRESHOLD,
            capacity_multiplier=1.25,
            fractional_coordinates=False,
            disable_cell_list=True,
            format=partition.Sparse,
        )
        print("\n[1] Edge sets")
        mismatched = 0
        for t in range(n_tiles):
            valid = jnp.asarray(tiled["mask"][t] > 0, dtype=jnp.bool_)
            nbrs = neighbor_fn.allocate(
                jnp.asarray(tiled["R"][t], dtype=jnp.float32),
                extra_capacity=10,
                mask=valid,
            )
            nbrs = custom_partition.mask_neighbor_list(
                nbrs,
                mask=valid,
                segment_id=jnp.asarray(tiled["segment_id"][t], dtype=jnp.int32),
            )
            dynamic = _edge_set(np.asarray(jax.device_get(nbrs.idx)), n_atoms)
            static = _edge_set(tiled["neighbor_idx"][t], n_atoms)
            if dynamic != static:
                mismatched += 1
                print(
                    f"  tile {t}: MISMATCH "
                    f"(static-only={len(static - dynamic)}, "
                    f"dynamic-only={len(dynamic - static)})"
                )
        if mismatched == 0:
            print(f"  all {n_tiles} tile(s) identical")

        # ---------------------------------------------------------------- #
        # 2/3) Energies and forces through chemtrain's force-matching model
        # ---------------------------------------------------------------- #
        model = CombinedModel(
            config=config_dynamic,
            R0=jnp.asarray(tiled["R"][0]),
            box=box,
            species=jnp.asarray(tiled["species"][0]),
            N_max=n_atoms,
            prior_only=False,
        )
        params = model.initialize_params(jax.random.PRNGKey(args.seed))

        pred_dynamic = _predict(model, params, tiled, static=False)
        pred_static = _predict(model, params, tiled, static=True)

        print("\n[2/3] Predictions (static vs dynamic)")
        e_max, e_rel = _compare(
            "energy", pred_static["U"], pred_dynamic["U"]
        )
        f_max, f_rel = _compare(
            "forces", pred_static["F"], pred_dynamic["F"]
        )

        # ---------------------------------------------------------------- #
        # 4) One optimizer update from identical parameters
        # ---------------------------------------------------------------- #
        import optax

        def loss_fn(p, static: bool):
            pred = _predict(model, p, tiled, static=static)
            diff = pred["F"] - jnp.asarray(tiled["F"])
            weight = jnp.asarray(tiled["mask"])[..., None]
            return jnp.sum(jnp.square(diff) * weight) / jnp.maximum(jnp.sum(weight), 1.0)

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        updated = {}
        for label, static in (("dynamic", False), ("static", True)):
            grads = jax.grad(loss_fn)(params, static)
            updates, _ = optimizer.update(grads, opt_state, params)
            updated[label] = optax.apply_updates(params, updates)

        flat_static = np.concatenate(
            [np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(updated["static"])]
        )
        flat_dynamic = np.concatenate(
            [np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(updated["dynamic"])]
        )
        print("\n[4] One optimizer update")
        p_max, p_rel = _compare("params after 1 step", flat_static, flat_dynamic)

        # ---------------------------------------------------------------- #
        print(
            f"\nReference benchmark (all-pairs vs cell-list, t1024): "
            f"max_abs={MAX_ABS_TARGET:.2e} rel_L2={REL_L2_TARGET:.2e}"
        )
        failures = []
        if mismatched:
            failures.append(f"{mismatched} tile(s) with mismatched edge sets")
        for label, value in (
            ("energy", e_max), ("forces", f_max), ("params", p_max)
        ):
            if not np.isfinite(value) or value > args.tol:
                failures.append(f"{label} max_abs={value:.3e} > tol={args.tol:.1e}")

        if failures:
            print("\nFAIL:")
            for item in failures:
                print(f"  - {item}")
            return 1
        print("\nPASS: static neighbor graphs reproduce the dynamic path.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
