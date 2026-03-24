#!/usr/bin/env python3
"""
Analyze residual prior targets for a training config.

This mirrors the data-loading and preprocessing stage from ``scripts/train.py``
for a single dataset, then computes:

    F_prior
    F_residual = F_ref - F_prior

using the priors configured in the YAML file. It can optionally run the same
LBFGS prior pretraining used by training prior-residual mode, then recompute
and compare the residual targets after updating the prior parameters.

Examples:
    python scripts/analyze_prior_residuals.py configs_testing/exploss_respriors.yaml
    python scripts/analyze_prior_residuals.py configs_testing/exploss_respriors.yaml --frames 256
    python scripts/analyze_prior_residuals.py config.yaml --pretrain-priors
    python scripts/analyze_prior_residuals.py config.yaml --pretrain-priors --compare-pretrain
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np


def _apply_jax_compat_shims() -> None:
    """Runtime compatibility for older jax_md/chemtrain imports."""
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.manager import ConfigManager
from data.loader import DatasetLoader
from data.preprocessor import CoordinatePreprocessor
from models.prior_energy import PriorEnergy
from models.topology import TopologyBuilder
from training.prior_residual import (
    apply_prior_force_residual_targets,
    pretrain_prior_for_residual,
)


TERM_ORDER = [
    "E_bond",
    "E_angle",
    "E_repulsive",
    "E_dihedral",
    "E_excluded_volume",
    "E_dh",
    "E_stickiness",
    "E_salt_bridge",
]

METRIC_ORDER = [
    ("prior_ref_rms_ratio", "prior/reference RMS ratio"),
    ("residual_ref_rms_ratio", "residual/reference RMS ratio"),
    ("rms_reduction", "RMS reduction after subtracting priors"),
    ("cosine", "cos(F_prior, F_ref)"),
    ("projection", "projection onto reference"),
]


def _resolve_data_path(config: ConfigManager) -> Path:
    """Match scripts/train.py relative data-path resolution."""
    data_path = Path(config.get_data_path())
    if data_path.is_absolute():
        return data_path
    return PROJECT_ROOT / data_path


def _copy_config(config: ConfigManager) -> ConfigManager:
    """Deep-copy ConfigManager so analysis variants can diverge safely."""
    copied = ConfigManager(config.config_path)
    copied._config = copy.deepcopy(config._config)  # deep copy for override isolation
    return copied


def _resolve_spline_path_if_needed(config: ConfigManager) -> None:
    """Resolve spline-file paths before PriorEnergy tries to load them."""
    if not config.use_spline_priors_enabled():
        return

    spline_path_str = config.get_spline_file_path()
    if not spline_path_str:
        return

    spline_path = Path(spline_path_str)
    if not spline_path.is_absolute():
        candidates = [
            Path.cwd() / spline_path_str,
            PROJECT_ROOT / spline_path_str,
            config.config_path.parent / spline_path_str,
        ]
        for candidate in candidates:
            if candidate.exists():
                spline_path = candidate
                break

    if not spline_path.exists():
        raise FileNotFoundError(f"Spline prior file not found: {spline_path_str}")

    config.set("model", "priors", "spline_file", str(spline_path.resolve()))


def _ensure_residual_analysis_enabled(
    config: ConfigManager,
    *,
    no_cache: bool,
    force_recompute: bool,
) -> None:
    """Allow analysis even when residual mode is not enabled in YAML."""
    if config.get("model", "priors", default=None) is None:
        raise ValueError("Config must define model.priors to analyze prior residuals.")

    residual_cfg = config.get("training", "prior_residual", default={})
    if not residual_cfg:
        residual_cfg = {}
        config.set("training", "prior_residual", residual_cfg)

    if not bool(residual_cfg.get("enabled", False)):
        residual_cfg["enabled"] = True
        print("[note] Enabled training.prior_residual for this analysis run only.")

    if no_cache:
        residual_cfg["cache_enabled"] = False
    if force_recompute:
        residual_cfg["force_recompute"] = True


def _slice_dataset(dataset: Dict[str, Any], n_frames: int) -> Dict[str, Any]:
    """Slice frame-wise arrays consistently."""
    sliced: Dict[str, Any] = {}
    total_frames = int(dataset["R"].shape[0])
    for key, value in dataset.items():
        if hasattr(value, "shape") and len(value.shape) > 0 and int(value.shape[0]) == total_frames:
            sliced[key] = np.asarray(value[:n_frames])
        else:
            sliced[key] = value
    return sliced


def _clone_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Deep copy array-valued dataset fields for isolated analysis passes."""
    cloned: Dict[str, Any] = {}
    for key, value in dataset.items():
        if isinstance(value, np.ndarray):
            cloned[key] = np.array(value, copy=True)
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _valid_vectors(F: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return valid force vectors with shape (n_valid, 3)."""
    valid = np.asarray(mask) > 0
    return np.asarray(F, dtype=np.float32)[valid]


def _component_rms(F: np.ndarray, mask: np.ndarray) -> float:
    """RMS over valid Cartesian force components."""
    F = np.asarray(F, dtype=np.float32)
    w = np.asarray(mask, dtype=np.float32)[..., None]
    denom = float(np.sum(w) * 3.0)
    if denom <= 0.0:
        return 0.0
    return float(np.sqrt(np.sum(np.square(F) * w) / denom))


def _norm_summary(F: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Summary of per-particle vector norms over valid particles."""
    vecs = _valid_vectors(F, mask)
    if vecs.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}

    norms = np.linalg.norm(vecs, axis=-1)
    return {
        "mean": float(np.mean(norms)),
        "median": float(np.median(norms)),
        "p90": float(np.quantile(norms, 0.9)),
        "max": float(np.max(norms)),
    }


def _global_cosine(A: np.ndarray, B: np.ndarray, mask: np.ndarray) -> float:
    """Cosine similarity over all valid force components."""
    a = _valid_vectors(A, mask).reshape(-1)
    b = _valid_vectors(B, mask).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _projection_fraction(prior: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    """Projected fraction of reference force power captured by priors."""
    p = _valid_vectors(prior, mask).reshape(-1)
    r = _valid_vectors(ref, mask).reshape(-1)
    denom = float(np.dot(r, r))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(p, r) / denom)


def _residual_norm_ratio(residual: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Distribution of |F_residual| / |F_ref| over valid particles."""
    res_norm = np.linalg.norm(_valid_vectors(residual, mask), axis=-1)
    ref_norm = np.linalg.norm(_valid_vectors(ref, mask), axis=-1)
    ratio = res_norm / np.maximum(ref_norm, 1e-8)
    return {
        "median": float(np.median(ratio)),
        "p90": float(np.quantile(ratio, 0.9)),
        "mean": float(np.mean(ratio)),
    }


def _format_float(value: float) -> str:
    """Readable floating-point formatting for console output."""
    if math.isnan(value):
        return "nan"
    if abs(value) >= 1e3 or (0.0 < abs(value) < 1e-2):
        return f"{value:.3e}"
    return f"{value:.4f}"


def _build_prior(config: ConfigManager, n_atoms: int, id_to_aa: Optional[Dict[int, str]]) -> PriorEnergy:
    """Construct PriorEnergy directly without building an ML model."""
    topology = TopologyBuilder(N_max=n_atoms, min_repulsive_sep=6)
    displacement = lambda Ra, Rb: Ra - Rb
    return PriorEnergy(config, topology, displacement, id_to_aa=id_to_aa)


def _compute_term_breakdown(
    *,
    config: ConfigManager,
    dataset: Dict[str, np.ndarray],
    id_to_aa: Optional[Dict[int, str]],
    fitted_params: Optional[Dict[str, np.ndarray]],
    n_frames: int,
) -> Dict[str, Dict[str, float]]:
    """
    Compute weighted prior term means on a small frame subset.

    This is intentionally limited to a few frames because per-term force
    decomposition requires autodiff through each energy component.
    """
    n_frames = min(int(n_frames), int(dataset["R"].shape[0]))
    if n_frames <= 0:
        return {}

    subset = _slice_dataset(dataset, n_frames)
    prior = _build_prior(config, n_atoms=int(subset["R"].shape[1]), id_to_aa=id_to_aa)
    params = None
    if fitted_params is not None:
        params = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in fitted_params.items()}

    R = jnp.asarray(subset["R"], dtype=jnp.float32)
    mask = jnp.asarray(subset["mask"], dtype=jnp.float32)
    species = jnp.asarray(subset["species"], dtype=jnp.int32)

    def single_energy_components(
        R_i: jax.Array,
        mask_i: jax.Array,
        species_i: jax.Array,
    ) -> Dict[str, jax.Array]:
        R_detached = jax.lax.stop_gradient(R_i)
        R_masked = jnp.where(mask_i[:, None] > 0, R_i, R_detached)
        return prior.compute_energy(R_masked, mask_i, species=species_i, params=params)

    energy_fn = jax.jit(jax.vmap(single_energy_components, in_axes=(0, 0, 0)))
    energy_components = jax.device_get(energy_fn(R, mask, species))

    summaries: Dict[str, Dict[str, float]] = {}
    mask_np = np.asarray(subset["mask"], dtype=np.float32)

    for term_name in TERM_ORDER:
        def single_force(
            R_i: jax.Array,
            mask_i: jax.Array,
            species_i: jax.Array,
            *,
            energy_key: str = term_name,
        ) -> jax.Array:
            def energy_of_R(R_var: jax.Array) -> jax.Array:
                R_detached = jax.lax.stop_gradient(R_var)
                R_masked = jnp.where(mask_i[:, None] > 0, R_var, R_detached)
                comps = prior.compute_energy(R_masked, mask_i, species=species_i, params=params)
                return comps[energy_key]

            return -jax.grad(energy_of_R)(R_i)

        force_fn = jax.jit(jax.vmap(single_force, in_axes=(0, 0, 0)))
        F_term = np.asarray(force_fn(R, mask, species), dtype=np.float32)

        summaries[term_name] = {
            "mean_energy": float(np.mean(np.asarray(energy_components[term_name], dtype=np.float32))),
            "force_rms": _component_rms(F_term, mask_np),
            "force_mean_norm": _norm_summary(F_term, mask_np)["mean"],
        }

    return summaries


def _compute_metrics(F_ref: np.ndarray, F_prior: np.ndarray, F_residual: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """Aggregate scalar diagnostics for one residual-analysis pass."""
    ref_rms = _component_rms(F_ref, mask)
    prior_rms = _component_rms(F_prior, mask)
    residual_rms = _component_rms(F_residual, mask)
    ratio_stats = _residual_norm_ratio(F_residual, F_ref, mask)

    return {
        "ref_rms": ref_rms,
        "prior_rms": prior_rms,
        "residual_rms": residual_rms,
        "prior_ref_rms_ratio": prior_rms / ref_rms if ref_rms > 0.0 else float("nan"),
        "residual_ref_rms_ratio": residual_rms / ref_rms if ref_rms > 0.0 else float("nan"),
        "rms_reduction": 1.0 - (residual_rms / ref_rms if ref_rms > 0.0 else float("nan")),
        "cosine": _global_cosine(F_prior, F_ref, mask),
        "projection": _projection_fraction(F_prior, F_ref, mask),
        "residual_ratio": ratio_stats,
    }


def _run_analysis(
    *,
    label: str,
    config: ConfigManager,
    dataset_source: Dict[str, np.ndarray],
    dataset_path: Path,
    dataset_tag: str,
    id_to_aa: Optional[Dict[int, str]],
    component_frames: int,
) -> Dict[str, Any]:
    """Run one prior residual analysis pass on a cloned dataset."""
    dataset = _clone_dataset(dataset_source)
    F_ref = np.asarray(dataset["F"], dtype=np.float32).copy()

    fitted_prior_params = pretrain_prior_for_residual(config, dataset, id_to_aa)
    stats = apply_prior_force_residual_targets(
        config=config,
        dataset=dataset,
        dataset_path=dataset_path,
        dataset_tag=dataset_tag,
        id_to_aa=id_to_aa,
        project_root=PROJECT_ROOT,
        seed=config.get_seed(),
        max_frames=config.get_max_frames(),
        cutoff=config.get_cutoff(),
        buffer_multiplier=config.get_buffer_multiplier(),
        park_multiplier=config.get_park_multiplier(),
        fitted_params=fitted_prior_params,
    )

    F_residual = np.asarray(dataset["F"], dtype=np.float32)
    F_prior = F_ref - F_residual
    mask = np.asarray(dataset["mask"], dtype=np.float32)
    metrics = _compute_metrics(F_ref, F_prior, F_residual, mask)

    component_summaries = {}
    if component_frames > 0:
        component_summaries = _compute_term_breakdown(
            config=config,
            dataset=dataset,
            id_to_aa=id_to_aa,
            fitted_params=fitted_prior_params,
            n_frames=component_frames,
        )

    return {
        "label": label,
        "config": config,
        "dataset": dataset,
        "F_ref": F_ref,
        "F_prior": F_prior,
        "F_residual": F_residual,
        "mask": mask,
        "stats": stats,
        "metrics": metrics,
        "fitted_prior_params": fitted_prior_params,
        "component_summaries": component_summaries,
    }


def _print_force_summary(title: str, F: np.ndarray, mask: np.ndarray) -> None:
    """Pretty-print vector-norm and RMS summaries."""
    stats = _norm_summary(F, mask)
    rms = _component_rms(F, mask)
    print(
        f"{title:<12} "
        f"rms={_format_float(rms)}  "
        f"mean|F|={_format_float(stats['mean'])}  "
        f"p90|F|={_format_float(stats['p90'])}  "
        f"max|F|={_format_float(stats['max'])}"
    )


def _print_term_breakdown(result: Dict[str, Any], component_frames: int, analyze_frames: int) -> None:
    """Print sorted per-term prior summaries."""
    component_summaries = result["component_summaries"]
    if not component_summaries:
        return

    print()
    print(f"Prior Term Breakdown [{result['label']}] (first {min(component_frames, analyze_frames)} frames)")
    print("-" * 80)
    ranked = sorted(
        component_summaries.items(),
        key=lambda item: item[1]["force_rms"],
        reverse=True,
    )
    for term_name, summary in ranked:
        print(
            f"{term_name:<18} "
            f"mean_E/frame={_format_float(summary['mean_energy'])}  "
            f"force_rms={_format_float(summary['force_rms'])}  "
            f"mean|F|={_format_float(summary['force_mean_norm'])}"
        )


def _print_analysis(result: Dict[str, Any]) -> None:
    """Print one residual-analysis section."""
    metrics = result["metrics"]
    stats = result["stats"]
    print()
    print(result["label"])
    print("=" * 80)
    print(f"pretrained_priors:  {result['fitted_prior_params'] is not None}")
    print(f"cache_hit:          {bool(stats.get('cache_hit', False))}")
    print(f"cache_path:         {stats.get('cache_path')}")
    print()
    print("Force Magnitude Summary")
    print("-" * 80)
    _print_force_summary("reference", result["F_ref"], result["mask"])
    _print_force_summary("prior", result["F_prior"], result["mask"])
    _print_force_summary("residual", result["F_residual"], result["mask"])
    print()
    print("Residual Impact")
    print("-" * 80)
    print(f"prior/reference RMS ratio:          {_format_float(metrics['prior_ref_rms_ratio'])}")
    print(f"residual/reference RMS ratio:       {_format_float(metrics['residual_ref_rms_ratio'])}")
    print(f"RMS reduction after subtracting priors: {_format_float(metrics['rms_reduction'])}")
    print(f"cos(F_prior, F_ref):                {_format_float(metrics['cosine'])}")
    print(f"projection onto reference:          {_format_float(metrics['projection'])}")
    print(
        "per-particle |F_res|/|F_ref|:      "
        f"mean={_format_float(metrics['residual_ratio']['mean'])}  "
        f"median={_format_float(metrics['residual_ratio']['median'])}  "
        f"p90={_format_float(metrics['residual_ratio']['p90'])}"
    )


def _print_comparison(before: Dict[str, Any], after: Dict[str, Any]) -> None:
    """Print metric deltas for baseline vs post-pretraining analyses."""
    print()
    print("Pretraining Delta")
    print("-" * 80)
    for key, label in METRIC_ORDER:
        b = before["metrics"][key]
        a = after["metrics"][key]
        delta = a - b
        print(
            f"{label:<36} before={_format_float(b)}  after={_format_float(a)}  delta={_format_float(delta)}"
        )


def _save_results_npz(output_path: Path, results: list[Dict[str, Any]]) -> None:
    """Save one or more analysis passes to an NPZ file."""
    payload: Dict[str, Any] = {}
    primary = results[0]
    payload["R"] = np.asarray(primary["dataset"]["R"], dtype=np.float32)
    payload["mask"] = np.asarray(primary["mask"], dtype=np.float32)
    payload["species"] = np.asarray(primary["dataset"]["species"], dtype=np.int32)
    payload["F_ref"] = np.asarray(primary["F_ref"], dtype=np.float32)

    for result in results:
        suffix = result["label"].lower().replace(" ", "_").replace("[", "").replace("]", "")
        payload[f"F_prior_{suffix}"] = np.asarray(result["F_prior"], dtype=np.float32)
        payload[f"F_residual_{suffix}"] = np.asarray(result["F_residual"], dtype=np.float32)
        payload[f"pretrained_{suffix}"] = np.asarray(bool(result["fitted_prior_params"] is not None))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the impact of prior residual targets F_ref - F_prior."
    )
    parser.add_argument("config", type=str, help="Path to YAML config.")
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Analyze only the first N preprocessed frames after DatasetLoader sampling.",
    )
    parser.add_argument(
        "--component-frames",
        type=int,
        default=16,
        help="Frames to use for the expensive per-term prior-force breakdown (0 disables it).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable residual prior cache for this analysis run.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of prior forces even if cache is available.",
    )
    parser.add_argument(
        "--pretrain-priors",
        action="store_true",
        help="Run the same LBFGS prior pretraining used by scripts/train.py before recomputing residuals.",
    )
    parser.add_argument(
        "--compare-pretrain",
        action="store_true",
        help="Run and print both baseline and post-pretraining residual analyses side by side.",
    )
    parser.add_argument(
        "--save-npz",
        type=str,
        default=None,
        help="Optional output .npz containing F_ref plus one or more prior/residual analysis results.",
    )
    args = parser.parse_args()

    base_config = ConfigManager(args.config)
    _resolve_spline_path_if_needed(base_config)
    _ensure_residual_analysis_enabled(
        base_config,
        no_cache=bool(args.no_cache),
        force_recompute=bool(args.force_recompute),
    )

    want_pretrain = bool(args.pretrain_priors or base_config.pretrain_prior_enabled())
    compare_pretrain = bool(args.compare_pretrain)

    if base_config.use_spline_priors_enabled() and want_pretrain:
        print("[note] Prior pretraining is only available for parametric priors; skipping LBFGS because spline priors are enabled.")
        want_pretrain = False

    data_path = _resolve_data_path(base_config)
    loader = DatasetLoader(
        str(data_path),
        max_frames=base_config.get_max_frames(),
        seed=base_config.get_seed(),
    )

    preprocessor = CoordinatePreprocessor(
        cutoff=base_config.get_cutoff(),
        buffer_multiplier=base_config.get_buffer_multiplier(),
        park_multiplier=base_config.get_park_multiplier(),
    )
    extent, r_shift = preprocessor.compute_box_extent(loader.R, loader.mask)

    dataset = loader.get_all()
    dataset["R"] = preprocessor.center_and_park(dataset["R"], dataset["mask"], extent, r_shift)

    loaded_frames = int(dataset["R"].shape[0])
    if args.frames is not None:
        if args.frames <= 0:
            raise ValueError("--frames must be > 0 when provided.")
        analyze_frames = min(int(args.frames), loaded_frames)
        if analyze_frames < loaded_frames:
            dataset = _slice_dataset(dataset, analyze_frames)
    else:
        analyze_frames = loaded_frames

    dataset_tag = data_path.stem
    if analyze_frames != loaded_frames:
        dataset_tag = f"{dataset_tag}_first{analyze_frames}"

    print("=" * 80)
    print("Prior Residual Analysis")
    print("=" * 80)
    print(f"config:             {Path(args.config).resolve()}")
    print(f"dataset:            {data_path.resolve()}")
    print(f"loaded_frames:      {loaded_frames}")
    print(f"analyzed_frames:    {analyze_frames}")
    print(f"n_atoms:            {dataset['R'].shape[1]}")
    print(f"use_spline_priors:  {base_config.use_spline_priors_enabled()}")
    print(f"pretrain_requested: {want_pretrain}")
    print(f"compare_pretrain:   {compare_pretrain}")

    results: list[Dict[str, Any]] = []

    run_baseline = compare_pretrain or not want_pretrain
    if run_baseline:
        baseline_config = _copy_config(base_config)
        baseline_config.set_pretrain_prior_enabled(False)
        results.append(
            _run_analysis(
                label="Baseline Analysis",
                config=baseline_config,
                dataset_source=dataset,
                dataset_path=data_path,
                dataset_tag=f"{dataset_tag}_baseline" if want_pretrain else dataset_tag,
                id_to_aa=loader.id_to_aa,
                component_frames=int(args.component_frames),
            )
        )

    if want_pretrain:
        pretrained_config = _copy_config(base_config)
        pretrained_config.set_pretrain_prior_enabled(True)
        results.append(
            _run_analysis(
                label="Post-LBFGS Pretraining Analysis",
                config=pretrained_config,
                dataset_source=dataset,
                dataset_path=data_path,
                dataset_tag=f"{dataset_tag}_pretrained",
                id_to_aa=loader.id_to_aa,
                component_frames=int(args.component_frames),
            )
        )

    for result in results:
        _print_analysis(result)
        _print_term_breakdown(result, int(args.component_frames), analyze_frames)

    if compare_pretrain and len(results) == 2:
        _print_comparison(results[0], results[1])

    if args.save_npz:
        output_path = Path(args.save_npz)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        _save_results_npz(output_path, results)
        print()
        print(f"saved_npz:          {output_path}")


if __name__ == "__main__":
    main()
