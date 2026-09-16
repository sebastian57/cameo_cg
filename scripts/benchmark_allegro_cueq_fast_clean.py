"""Compare Allegro cuEq backends over increasing edge counts.

Run this on a CUDA compute node from the repository root.  The default model
settings match the approved 2026-08-30 wide160 production configuration.
"""

from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax_md import partition, space

from models import allegro_cueq_fast_1103 as original_backend
from models import allegro_cueq_fast_clean as clean_backend


MODEL_KWARGS = dict(
    max_ell=2,
    hidden_irreps="160x0e + 160x0o + 160x1o + 160x1e + 160x2e",
    output_irreps="0e",
    mlp_activation=jax.nn.silu,
    mlp_output_activation=jax.nn.tanh,
    mlp_n_hidden=192,
    mlp_n_layers=4,
    embed_n_hidden=(96, 160),
    species_embed=16,
    envelope_p=6,
    n_radial_basis=20,
    num_layers=5,
    tp_batch_strategy="nested_vmap",
    tp_left_mode="node_agg",
    tp_left_norm=None,
    tp_method="naive",
    tp_mode="mixed_naive",
)


def _make_inputs(n_nodes: int, avg_neighbors: int = 5):
    displacement, _shift = space.free()
    senders = jnp.repeat(jnp.arange(n_nodes, dtype=jnp.int32), avg_neighbors)
    offsets = jnp.tile(jnp.arange(1, avg_neighbors + 1, dtype=jnp.int32), n_nodes)
    receivers = (senders + offsets) % n_nodes
    neighbor = SimpleNamespace(
        format=partition.Sparse,
        idx=(receivers, senders),
    )
    node_ids = jnp.arange(n_nodes, dtype=jnp.float32)
    position = jnp.stack(
        [
            0.25 * jnp.sin(node_ids * 0.17),
            0.25 * jnp.cos(node_ids * 0.11),
            0.25 * jnp.sin(node_ids * 0.07 + 0.3),
        ],
        axis=-1,
    )
    species = jnp.mod(jnp.arange(n_nodes), 4).astype(jnp.int32)
    mask = jnp.ones((n_nodes,), dtype=jnp.bool_)
    return displacement, position, neighbor, species, mask


def _factory(backend, displacement, *, optimized: bool = False, per_particle: bool = False):
    kwargs = dict(MODEL_KWARGS)
    kwargs["tp_backend"] = "fused_sp" if optimized else "baseline_mixed"
    return backend.allegro_neighborlist_pp(
        displacement=displacement,
        r_cutoff=5.5,
        n_species=4,
        avg_num_neighbors=5.0,
        mode="energy",
        per_particle=per_particle,
        edge_distance_gate=None,
        **kwargs,
    )


def _block(value: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf,
        value,
    )


def _seconds(fn: Callable, args: tuple[Any, ...], repeats: int) -> tuple[float, float, Any]:
    start = time.perf_counter()
    first = _block(fn(*args))
    compile_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(repeats):
        _block(fn(*args))
    steady_seconds = (time.perf_counter() - start) / repeats
    return compile_seconds, steady_seconds, first


def _max_abs(a: Any, b: Any) -> float:
    return float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))


def _shape_signature(tree: Any):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return treedef, [(tuple(leaf.shape), str(leaf.dtype)) for leaf in leaves]


def benchmark_size(n_nodes: int, repeats: int) -> dict[str, Any]:
    displacement, position, neighbor, species, mask = _make_inputs(n_nodes)
    args = (position, neighbor, species, mask)

    variants = {
        "original_baseline": (original_backend, False),
        "clean_baseline": (clean_backend, False),
        "original_optimized": (original_backend, True),
    }
    built = {}
    result: dict[str, Any] = {"n_nodes": n_nodes, "n_edges": int(neighbor.idx[0].shape[0])}

    for name, (backend, optimized) in variants.items():
        try:
            init_fn, apply_fn = _factory(
                backend,
                displacement,
                optimized=optimized,
            )
            init_seconds, _unused, params = _seconds(
                init_fn,
                (jax.random.PRNGKey(0),) + args,
                repeats=1,
            )
            built[name] = (apply_fn, params)
            result.setdefault("init_seconds", {})[name] = init_seconds
        except Exception as exc:
            result.setdefault("errors", {})[name] = f"{type(exc).__name__}: {exc}"

    if "original_baseline" not in built or "clean_baseline" not in built:
        return result

    original_apply, original_params = built["original_baseline"]
    clean_apply, clean_params = built["clean_baseline"]
    original_signature = _shape_signature(original_params)
    clean_signature = _shape_signature(clean_params)
    result["parameter_signature_equal"] = original_signature == clean_signature
    if not result["parameter_signature_equal"]:
        result["errors"] = result.get("errors", {})
        result["errors"]["parameter_tree"] = "original and clean parameter trees differ"
        return result

    variants_to_time = {
        "original_baseline": (original_apply, original_params),
        "clean_baseline": (clean_apply, original_params),
    }
    if "original_optimized" in built:
        variants_to_time["original_optimized"] = (built["original_optimized"][0], original_params)

    values = {}
    for name, (apply_fn, params) in variants_to_time.items():
        try:
            compile_seconds, steady_seconds, value = _seconds(
                apply_fn,
                (params,) + args,
                repeats,
            )
            values[name] = value
            result.setdefault("energy_seconds", {})[name] = {
                "compile": compile_seconds,
                "steady": steady_seconds,
            }
        except Exception as exc:
            result.setdefault("errors", {})[name] = f"{type(exc).__name__}: {exc}"

    if "original_baseline" in values and "clean_baseline" in values:
        result["energy_max_abs_original_vs_clean"] = _max_abs(
            values["original_baseline"], values["clean_baseline"]
        )

    force_fns = {}
    param_grad_fns = {}
    for name, (apply_fn, params) in variants_to_time.items():
        force_fns[name] = jax.jit(
            jax.grad(lambda p, position_: apply_fn(p, position_, neighbor, species, mask), argnums=1)
        )
        param_grad_fns[name] = jax.jit(
            jax.grad(lambda p: apply_fn(p, position, neighbor, species, mask))
        )

    force_values = {}
    param_grad_values = {}
    for name, (apply_fn, params) in variants_to_time.items():
        try:
            force_compile, force_steady, force_value = _seconds(
                force_fns[name],
                (params, position),
                repeats,
            )
            param_compile, param_steady, param_value = _seconds(
                param_grad_fns[name],
                (params,),
                repeats,
            )
            force_values[name] = force_value
            param_grad_values[name] = param_value
            result.setdefault("force_seconds", {})[name] = {
                "compile": force_compile,
                "steady": force_steady,
            }
            result.setdefault("param_grad_seconds", {})[name] = {
                "compile": param_compile,
                "steady": param_steady,
            }
        except Exception as exc:
            result.setdefault("errors", {})[name] = f"{type(exc).__name__}: {exc}"

    if "original_baseline" in force_values and "clean_baseline" in force_values:
        result["force_max_abs_original_vs_clean"] = _max_abs(
            force_values["original_baseline"], force_values["clean_baseline"]
        )
    if "original_baseline" in param_grad_values and "clean_baseline" in param_grad_values:
        result["param_grad_max_abs_original_vs_clean"] = _max_abs(
            param_grad_values["original_baseline"], param_grad_values["clean_baseline"]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="32,256,1024")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    print(json.dumps({"devices": [str(device) for device in jax.devices()]}))
    for size in (int(value) for value in args.sizes.split(",") if value.strip()):
        print(json.dumps(benchmark_size(size, args.repeats), sort_keys=True))


if __name__ == "__main__":
    main()
