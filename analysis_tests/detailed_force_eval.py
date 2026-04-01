#!/usr/bin/env python3
"""Detailed held-out force evaluation with baseline comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.manager import ConfigManager


def _resolve_data_path(config: ConfigManager) -> Path:
    data_path = Path(config.get_data_path())
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    return data_path


def _split_indices(n_frames: int, *, val_fraction: float, min_val_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_frames, dtype=np.int32)
    if n_frames > 1:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_frames)

    n_train = int(np.round(n_frames * (1.0 - val_fraction)))
    n_val = n_frames - n_train
    if n_val < min_val_samples:
        n_train = n_frames
        n_val = 0
    return indices[:n_train], indices[n_train:n_train + n_val]


from utils.jax_setup import apply_jax_compat_shims as _apply_jax_compat_shims_global


def _resolve_spline_path_if_needed(config: ConfigManager) -> None:
    if not config.use_spline_priors_enabled():
        return
    spline_path = Path(config.get_spline_file_path())
    if spline_path.is_absolute():
        resolved = spline_path
    else:
        candidates = [
            Path.cwd() / spline_path,
            PROJECT_ROOT / spline_path,
            config.config_path.parent / spline_path,
        ]
        resolved = spline_path
        for candidate in candidates:
            if candidate.exists():
                resolved = candidate
                break
    config.set("model", "priors", "spline_file", str(resolved.resolve()))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_center = x - np.mean(x)
    y_center = y - np.mean(y)
    denom = np.linalg.norm(x_center) * np.linalg.norm(y_center)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(x_center, y_center) / denom)


def _mean_cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    dots = np.sum(y_true * y_pred, axis=1)
    norms = np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1)
    valid = norms > 1e-12
    if not np.any(valid):
        return float("nan")
    cos = np.zeros_like(dots, dtype=np.float64)
    cos[valid] = dots[valid] / norms[valid]
    return float(np.mean(cos[valid]))


def _r2_vector(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_mean = np.mean(y_true, axis=0, keepdims=True)
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - y_mean) ** 2))
    if sst <= 1e-12:
        return float("nan")
    return float(1.0 - sse / sst)


def _variance_ratio(y_true_valid: np.ndarray, y_pred_valid: np.ndarray) -> float:
    var_true = float(np.var(y_true_valid))
    if var_true <= 1e-12:
        return float("nan")
    return float(np.var(y_pred_valid) / var_true)


def _masked_flatten(F: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (F * mask[..., None]).reshape(F.shape[0], -1)


def _valid_component_mask(mask: np.ndarray) -> np.ndarray:
    return np.repeat(mask[..., None] > 0, 3, axis=-1).reshape(mask.shape[0], -1)


def compute_detailed_force_eval(
    config_path: Path,
    params_path: Path,
    output_dir: Path,
    *,
    devices_per_run: int = 4,
    shuffle_repeats: int = 8,
    shuffle_seed: int = 42,
    max_val_frames: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    import jax
    import jax.numpy as jnp

    _apply_jax_compat_shims_global()

    from data.loader import DatasetLoader
    from data.preprocessor import CoordinatePreprocessor
    from models.combined_model import CombinedModel

    config = ConfigManager(str(config_path))
    _resolve_spline_path_if_needed(config)

    data_path = _resolve_data_path(config)
    loader = DatasetLoader(str(data_path), max_frames=config.get_max_frames(), seed=config.get_seed())
    preprocessor = CoordinatePreprocessor(
        cutoff=config.get_cutoff(),
        buffer_multiplier=config.get_buffer_multiplier(),
        park_multiplier=config.get_park_multiplier(),
    )
    extent, r_shift = preprocessor.compute_box_extent(loader.R, loader.mask)
    dataset = loader.get_all()
    dataset["R"] = preprocessor.center_and_park(dataset["R"], dataset["mask"], extent, r_shift)
    box = np.asarray(extent, dtype=np.float32)

    train_idx, val_idx = _split_indices(
        dataset["R"].shape[0],
        val_fraction=float(config.get_val_fraction()),
        min_val_samples=int(config.get_batch_per_device()) * int(devices_per_run),
        seed=int(config.get_seed()),
    )
    if max_val_frames is not None:
        val_idx = val_idx[: int(max_val_frames)]
    if val_idx.size == 0:
        raise ValueError("No held-out validation samples available for detailed force evaluation.")

    config.set("model", "use_priors", bool(config.export_combined_ml_priors_enabled()))
    config.set("model", "train_priors", False)

    with open(params_path, "rb") as f:
        payload = pickle.load(f)

    params = payload
    if isinstance(payload, dict):
        if isinstance(payload.get("params"), dict):
            params = payload["params"]
        elif isinstance(payload.get("trainer_state"), dict) and isinstance(payload["trainer_state"].get("params"), dict):
            params = payload["trainer_state"]["params"]

    if isinstance(params, dict) and "ml" not in params and "allegro" in params:
        params = dict(params)
        params["ml"] = params["allegro"]

    ref_idx = train_idx[0] if train_idx.size else val_idx[0]
    model = CombinedModel(
        config=config,
        R0=np.asarray(dataset["R"][ref_idx], dtype=np.float32),
        box=box,
        species=np.asarray(dataset["species"][ref_idx], dtype=np.int32),
        N_max=loader.N_max,
        id_to_aa=loader.id_to_aa,
        prior_only=False,
    )

    F_train_ref = np.asarray(dataset["F"][train_idx], dtype=np.float32)
    mask_train = np.asarray(dataset["mask"][train_idx], dtype=np.float32)
    F_val_ref = np.asarray(dataset["F"][val_idx], dtype=np.float32)
    mask_val = np.asarray(dataset["mask"][val_idx], dtype=np.float32)

    # Build a vmapped, JIT-compiled force function over the batch dimension.
    # All frames share the same padded shape so vmap compiles once.
    def _forces_single(R, mask, species):
        def energy_fn(R_):
            return model.compute_energy(params, R_, mask, species, None)
        return -jax.grad(energy_fn)(R)

    batched_forces_fn = jax.jit(jax.vmap(_forces_single))

    R_val_jax = jnp.asarray(dataset["R"][val_idx], dtype=jnp.float32)
    mask_val_jax = jnp.asarray(dataset["mask"][val_idx], dtype=jnp.float32)
    species_val_jax = jnp.asarray(dataset["species"][val_idx], dtype=jnp.int32)

    n_val = val_idx.size
    chunk_size = n_val if batch_size is None else int(batch_size)
    chunks = []
    for start in range(0, n_val, chunk_size):
        end = min(start + chunk_size, n_val)
        F_chunk = batched_forces_fn(
            R_val_jax[start:end],
            mask_val_jax[start:end],
            species_val_jax[start:end],
        )
        chunks.append(np.asarray(F_chunk, dtype=np.float32))
        print(f"  [detailed_force_eval] evaluated {end}/{n_val} frames", flush=True)
    F_val_pred = np.concatenate(chunks, axis=0)

    Y_train = _masked_flatten(F_train_ref, mask_train)
    Y_ref = _masked_flatten(F_val_ref, mask_val)
    Y_pred = _masked_flatten(F_val_pred, mask_val)

    valid_components = _valid_component_mask(mask_val)
    y_ref_valid = Y_ref[valid_components]
    y_pred_valid = Y_pred[valid_components]

    mean_force_train = np.mean(Y_train, axis=0, keepdims=True) if Y_train.size else np.zeros((1, Y_ref.shape[1]), dtype=np.float32)
    Y_zero = np.zeros_like(Y_ref)
    Y_mean = np.repeat(mean_force_train, Y_ref.shape[0], axis=0)
    y_zero_valid = Y_zero[valid_components]
    y_mean_valid = Y_mean[valid_components]

    # Baseline RMSEs should ignore padded components just like training loss and
    # the lightweight force eval do; otherwise matching zeros in padded slots
    # artificially deflate the reported RMSE.
    rmse_model = _rmse(y_ref_valid, y_pred_valid)
    rmse_zero = _rmse(y_ref_valid, y_zero_valid)
    rmse_mean = _rmse(y_ref_valid, y_mean_valid)

    rng = np.random.RandomState(shuffle_seed)
    shuffle_rmses = []
    for _ in range(int(shuffle_repeats)):
        perm = rng.permutation(Y_pred.shape[0])
        shuffle_rmses.append(_rmse(y_ref_valid, Y_pred[perm][valid_components]))
    shuffle_rmses = np.asarray(shuffle_rmses, dtype=np.float64)

    metrics: Dict[str, Any] = {
        "split_name": "validation",
        "n_train_samples": int(train_idx.size),
        "n_eval_samples": int(val_idx.size),
        "shuffle_repeats": int(shuffle_repeats),
        "rmse_model": rmse_model,
        "rmse_zero": rmse_zero,
        "rmse_mean": rmse_mean,
        "rmse_shuffle_mean": float(np.mean(shuffle_rmses)),
        "rmse_shuffle_std": float(np.std(shuffle_rmses)),
        "shuffle_gap_rmse": float(np.mean(shuffle_rmses) - rmse_model),
        "pearson_global": _safe_pearson(y_ref_valid, y_pred_valid),
        "mean_cosine_similarity": _mean_cosine_similarity(Y_ref, Y_pred),
        "r2_explained_variance": _r2_vector(Y_ref, Y_pred),
        "variance_ratio_pred_to_ref": _variance_ratio(y_ref_valid, y_pred_valid),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = output_dir / "metrics.json"
    metrics_csv = output_dir / "metrics.csv"
    shuffle_csv = output_dir / "shuffle_rmses.csv"
    baseline_plot = output_dir / "baseline_rmse_comparison.png"
    shuffle_plot = output_dir / "shuffle_rmse_distribution.png"
    cosine_plot = output_dir / "cosine_similarity_hist.png"

    metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    with open(shuffle_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["shuffle_idx", "rmse"])
        for i, value in enumerate(shuffle_rmses):
            writer.writerow([i, float(value)])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["model", "zero", "mean", "shuffle"]
    values = [metrics["rmse_model"], metrics["rmse_zero"], metrics["rmse_mean"], metrics["rmse_shuffle_mean"]]
    errors = [0.0, 0.0, 0.0, metrics["rmse_shuffle_std"]]
    ax.bar(labels, values, yerr=errors, color=["#2a9d8f", "#8d99ae", "#457b9d", "#e76f51"], alpha=0.9, capsize=4)
    ax.set_ylabel("RMSE")
    ax.set_title("Held-out Baseline RMSE Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(baseline_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(shuffle_rmses, bins=min(10, max(5, len(shuffle_rmses))), alpha=0.75, color="#e76f51", edgecolor="black")
    ax.axvline(metrics["rmse_model"], color="#2a9d8f", linestyle="--", linewidth=2.0, label=f"Model RMSE = {metrics['rmse_model']:.4f}")
    ax.axvline(metrics["rmse_shuffle_mean"], color="#264653", linestyle="-", linewidth=2.0, label=f"Shuffle mean = {metrics['rmse_shuffle_mean']:.4f}")
    ax.set_xlabel("Shuffle RMSE")
    ax.set_ylabel("Count")
    ax.set_title("Shuffled Baseline RMSE Distribution")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(shuffle_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)

    dots = np.sum(Y_ref * Y_pred, axis=1)
    norms = np.linalg.norm(Y_ref, axis=1) * np.linalg.norm(Y_pred, axis=1)
    valid = norms > 1e-12
    cos_values = dots[valid] / norms[valid] if np.any(valid) else np.asarray([], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5))
    if cos_values.size > 0:
        ax.hist(cos_values, bins=40, alpha=0.75, color="#457b9d", edgecolor="black")
        ax.axvline(metrics["mean_cosine_similarity"], color="#d62828", linestyle="--", linewidth=2.0, label=f"Mean = {metrics['mean_cosine_similarity']:.4f}")
        ax.legend(loc="best")
    ax.set_xlabel("Per-sample cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title("Held-out Cosine Similarity Distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(cosine_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)

    metrics.update({
        "metrics_json_path": str(metrics_json),
        "metrics_csv_path": str(metrics_csv),
        "shuffle_csv_path": str(shuffle_csv),
        "baseline_plot_path": str(baseline_plot),
        "shuffle_plot_path": str(shuffle_plot),
        "cosine_plot_path": str(cosine_plot),
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed held-out force evaluation")
    parser.add_argument("config", type=str)
    parser.add_argument("params", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--devices-per-run", type=int, default=4)
    parser.add_argument("--shuffle-repeats", type=int, default=8)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--max-val-frames", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Frames per vmap chunk. Defaults to all val frames at once.")
    args = parser.parse_args()

    metrics = compute_detailed_force_eval(
        Path(args.config).resolve(),
        Path(args.params).resolve(),
        Path(args.output_dir).resolve(),
        devices_per_run=args.devices_per_run,
        shuffle_repeats=args.shuffle_repeats,
        shuffle_seed=args.shuffle_seed,
        max_val_frames=args.max_val_frames,
        batch_size=args.batch_size,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
