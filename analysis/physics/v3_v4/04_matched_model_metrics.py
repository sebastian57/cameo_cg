"""Compare trained models on one explicitly aligned held-out state set."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from analysis.physics.v3_v4.common import bootstrap_mean, join_labels_by_state, load_label_table, load_state_table, write_json


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(np.asarray(arrays[0]).shape, dtype=bool)
    for array in arrays:
        mask &= np.isfinite(np.asarray(array))
    return mask


def evaluate_force_metrics(
    predicted_force: np.ndarray,
    measured_force: np.ndarray,
    measured_se: np.ndarray,
) -> dict[str, float]:
    """Return force errors, including a label-noise-normalized error."""
    predicted = np.asarray(predicted_force, dtype=np.float64)
    measured = np.asarray(measured_force, dtype=np.float64)
    se = np.asarray(measured_se, dtype=np.float64)
    if predicted.shape != measured.shape or measured.shape != se.shape:
        raise ValueError("predicted force, measured force, and SE must have equal shapes")
    diff = predicted - measured
    valid = _finite_mask(diff, se) & (se > 0)
    if not np.any(valid):
        raise ValueError("no finite force/SE entries available for metric evaluation")
    d = diff[valid]
    z = np.abs(d / se[valid])
    return {
        "rmse": float(np.sqrt(np.mean(d * d))),
        "mae": float(np.mean(np.abs(d))),
        "median_abs_z": float(np.median(z)),
        "mean_abs_z": float(np.mean(z)),
        "bias_norm": float(np.linalg.norm(np.mean(d.reshape(-1, 3), axis=0))),
        "n_components": int(d.size),
    }


def evaluate_hvp_metrics(hvp_pred: np.ndarray, hvp_measured: np.ndarray) -> dict[str, float]:
    """Compare Hessian-vector products with global and per-row relative errors."""
    predicted = np.asarray(hvp_pred, dtype=np.float64)
    measured = np.asarray(hvp_measured, dtype=np.float64)
    if predicted.shape != measured.shape:
        raise ValueError("predicted and measured HVP arrays must have equal shapes")
    diff = predicted - measured
    denom = np.linalg.norm(measured.reshape(len(measured), -1), axis=1)
    numer = np.linalg.norm(diff.reshape(len(diff), -1), axis=1)
    good = np.isfinite(denom) & np.isfinite(numer)
    if not np.any(good):
        raise ValueError("no finite HVP rows available")
    safe = np.maximum(denom[good], np.finfo(float).eps)
    cos = np.sum(
        predicted.reshape(len(predicted), -1)[good]
        * measured.reshape(len(measured), -1)[good],
        axis=1,
    ) / np.maximum(
        np.linalg.norm(predicted.reshape(len(predicted), -1)[good], axis=1) * safe,
        np.finfo(float).eps,
    )
    return {
        "relative_error": float(np.linalg.norm(diff) / max(np.linalg.norm(measured), np.finfo(float).eps)),
        "median_relative_error": float(np.median(numer[good] / safe)),
        "median_cosine": float(np.median(cos)),
        "rmse": float(np.sqrt(np.mean(diff[good] ** 2))),
        "n_rows": int(np.sum(good)),
    }


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("models", data.get("entries", data))
    if not isinstance(data, list) or not data:
        raise ValueError("model manifest must be a non-empty JSON list")
    out = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"model manifest entry {i} is not an object")
        row = dict(entry)
        row.setdefault("name", f"model_{i}")
        if "config" not in row or "params" not in row:
            raise ValueError(f"model manifest entry {row['name']!r} needs config and params")
        out.append(row)
    return out


def _load_model_entry(entry: dict[str, Any], dataset: Path, n_beads: int):
    from analysis.md.analyze_model_residuals_by_region import _load_model

    return _load_model(str(entry["config"]), str(entry["params"]), str(dataset), n_beads)


def _predict_forces(entry: dict[str, Any], R: np.ndarray, dataset: Path) -> tuple[np.ndarray, np.ndarray]:
    import jax
    import jax.numpy as jnp

    model, params, mask0, species0 = _load_model_entry(entry, dataset, R.shape[1])

    def energy(r):
        return model.compute_energy(params, r, mask0, species0)

    value_grad = jax.jit(jax.value_and_grad(energy))
    values = np.empty(len(R), dtype=np.float64)
    forces = np.empty_like(R, dtype=np.float64)
    for i, frame in enumerate(R):
        value, grad = value_grad(jnp.asarray(frame, dtype=jnp.float32))
        values[i] = float(value)
        forces[i] = -np.asarray(grad, dtype=np.float64)
    return values, forces


def _select_rows(states: dict[str, np.ndarray], n_anchors: int, seed: int) -> np.ndarray:
    n = len(states["R"])
    if "anchor" not in states or np.all(np.asarray(states["direction"]) < 0):
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n, min(n_anchors, n), replace=False))
    anchor = np.asarray(states["anchor"])
    direction = np.asarray(states.get("direction", np.full(n, -1)))
    candidates = np.unique(anchor[direction < 0])
    if not len(candidates):
        candidates = np.unique(anchor)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates, min(n_anchors, len(candidates)), replace=False)
    return np.flatnonzero(np.isin(anchor, chosen))


def _collect_hvp_pairs(states: dict[str, np.ndarray], forces: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor = np.asarray(states["anchor"], dtype=np.int64)
    direction = np.asarray(states["direction"], dtype=np.int64)
    multiplier = np.asarray(states.get("multiplier", np.zeros(len(anchor))), dtype=np.float64)
    eps = np.asarray(states.get("eps_state", np.zeros(len(anchor))), dtype=np.float64)
    R = np.asarray(states["R"], dtype=np.float64)
    selected = set(int(i) for i in np.asarray(rows, dtype=np.int64))
    positive: dict[tuple[int, int, float], int] = {}
    negative: dict[tuple[int, int, float], int] = {}
    for i in selected:
        if direction[i] < 0 or multiplier[i] == 0 or eps[i] <= 0:
            continue
        key = (int(anchor[i]), int(direction[i]), round(abs(float(multiplier[i])), 7))
        (positive if multiplier[i] > 0 else negative).setdefault(key, i)
    centers, tangents, targets = [], [], []
    for key in sorted(set(positive) & set(negative)):
        ip, im = positive[key], negative[key]
        step = abs(float(multiplier[ip])) * abs(float(eps[ip]))
        if step <= 0:
            continue
        centers.append(0.5 * (R[ip] + R[im]))
        tangents.append((R[ip] - R[im]) / (2.0 * step))
        targets.append(-(forces[ip] - forces[im]) / (2.0 * step))
    if not centers:
        return (np.empty((0,) + R.shape[1:]), np.empty((0,) + R.shape[1:]),
                np.empty((0,) + R.shape[1:]))
    return np.asarray(centers), np.asarray(tangents), np.asarray(targets)


def _predict_hvp(entry: dict[str, Any], centers: np.ndarray, tangents: np.ndarray, dataset: Path) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    model, params, mask0, species0 = _load_model_entry(entry, dataset, centers.shape[1])

    def energy(r):
        return model.compute_energy(params, r, mask0, species0)

    grad = jax.grad(energy)
    hvp = jax.jit(lambda r, v: jax.jvp(grad, (r,), (v,))[1])
    return np.asarray([
        np.asarray(hvp(jnp.asarray(r, dtype=jnp.float32), jnp.asarray(v, dtype=jnp.float32)))
        for r, v in zip(centers, tangents)
    ], dtype=np.float64)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True, help="held-out state NPZ")
    parser.add_argument("--labels", type=Path, required=True, help="matching mean-force NPZ")
    parser.add_argument("--models", type=Path, required=True, help="JSON model manifest")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--model-reference", type=Path, default=None,
                        help="dataset containing species/mask for model construction; defaults to states")
    parser.add_argument("--n-anchors", type=int, default=500)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--mapping", default="ala2_backbone_cb_6")
    args = parser.parse_args()

    states = load_state_table(args.states)
    labels = load_label_table(args.labels)
    joined = join_labels_by_state(states, labels)
    rows = _select_rows(states, args.n_anchors, args.seed)
    R = np.asarray(states["R"][rows], dtype=np.float64)
    F = np.asarray(joined["F"][rows], dtype=np.float64)
    SE = np.asarray(joined.get("SE", np.ones_like(F)), dtype=np.float64)
    manifest = _load_manifest(args.models)
    model_reference = args.model_reference or args.states

    report: dict[str, Any] = {
        "states": str(args.states),
        "labels": str(args.labels),
        "models": str(args.models),
        "evaluation_indices": rows.tolist(),
        "n_evaluation_states": int(len(rows)),
        "seed": args.seed,
        "model_results": [],
    }
    csv_rows: list[dict[str, Any]] = []
    for entry in manifest:
        started = time.time()
        energies, predicted = _predict_forces(entry, R, model_reference)
        metrics = evaluate_force_metrics(predicted, F, SE)
        centers, tangents, hvp_target = _collect_hvp_pairs(states, joined["F"], rows)
        if len(centers):
            metrics["hvp"] = evaluate_hvp_metrics(
                _predict_hvp(entry, centers, tangents, model_reference), hvp_target
            )
        else:
            metrics["hvp"] = {"n_rows": 0, "note": "no complete +/- pairs in selected states"}
        per_state = np.sqrt(np.mean((predicted - F) ** 2, axis=(1, 2)))
        metrics["rmse_boot_lo"], metrics["rmse_boot_hi"] = bootstrap_mean(
            per_state, args.n_boot, args.seed
        )[1:]
        result = {"name": entry["name"], **metrics,
                  "energy_mean": float(np.mean(energies)),
                  "runtime_s": time.time() - started}
        report["model_results"].append(result)
        csv_rows.append(result)

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_json(args.outdir / "metrics.json", report)
    _write_rows(args.outdir / "metrics.csv", csv_rows)
    np.savez_compressed(args.outdir / "metrics_raw.npz", indices=rows, R=R, F=F, SE=SE)
    print(json.dumps(report["model_results"], indent=2))


if __name__ == "__main__":
    main()
