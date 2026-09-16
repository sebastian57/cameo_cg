"""Measure translation/rotation invariance and Cartesian gauge curvature."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import numpy as np

from analysis.physics.v3_v4.common import (
    jsonable_summary,
    rigid_tangent,
    rotate_about_centroid,
    write_json,
)


AXES = np.eye(3, dtype=np.float64)


def rotation_second_derivative(R: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Second derivative of exact centroid rotation at angle zero."""
    tangent = rigid_tangent(R, axis)
    unit_axis = np.asarray(axis, dtype=np.float64).ravel()
    unit_axis /= np.linalg.norm(unit_axis)
    return np.cross(unit_axis[None, :], tangent)


def translation_tangent(R: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).ravel()
    axis /= np.linalg.norm(axis)
    return np.broadcast_to(axis, np.asarray(R).shape).copy()


def rotation_invariance_error(
    energy_fn: Callable[[np.ndarray], float], R: np.ndarray, axis: np.ndarray, angle: float
) -> float:
    base = float(energy_fn(np.asarray(R, dtype=np.float64)))
    moved = float(energy_fn(rotate_about_centroid(R, axis, angle)))
    return abs(moved - base)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).ravel()
    return {
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def analyze_gauge(
    energy_fn: Callable[[np.ndarray], float],
    gradient_fn: Callable[[np.ndarray], np.ndarray],
    hessian_fn: Callable[[np.ndarray], np.ndarray],
    anchors: np.ndarray,
    angles: tuple[float, ...] = (1e-3, 0.1, 0.5),
) -> dict[str, np.ndarray | dict[str, float]]:
    """Return raw gauge residuals and corrected/uncorrected rotational curvatures."""
    anchors = np.asarray(anchors, dtype=np.float64)
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    cart_rotation: list[float] = []
    correction: list[float] = []
    covariant_rotation: list[float] = []
    translation_curvature: list[float] = []

    for R in anchors:
        E0 = float(energy_fn(R))
        H = np.asarray(hessian_fn(R), dtype=np.float64).reshape(R.size, R.size)
        H = 0.5 * (H + H.T)
        g = np.asarray(gradient_fn(R), dtype=np.float64).reshape(-1)
        for axis in AXES:
            moved = R + translation_tangent(R, axis) * 1.0
            translation_errors.append(abs(float(energy_fn(moved)) - E0))
            q_t = translation_tangent(R, axis).reshape(-1)
            q_t /= np.linalg.norm(q_t)
            translation_curvature.append(float(q_t @ H @ q_t))

            moved_angles = [rotation_invariance_error(energy_fn, R, axis, angle) for angle in angles]
            rotation_errors.append(max(moved_angles))
            q_raw = rigid_tangent(R, axis).reshape(-1)
            q_norm = np.linalg.norm(q_raw)
            if q_norm == 0:
                continue
            q = q_raw / q_norm
            a2 = rotation_second_derivative(R, axis).reshape(-1) / (q_norm**2)
            h_cart = float(q @ H @ q)
            force_coordinate_term = float(g @ a2)
            cart_rotation.append(h_cart)
            correction.append(force_coordinate_term)
            covariant_rotation.append(h_cart + force_coordinate_term)

    return {
        "translation_energy_error": np.asarray(translation_errors),
        "rotation_energy_error": np.asarray(rotation_errors),
        "translation_cartesian_curvature": np.asarray(translation_curvature),
        "rotation_cartesian_curvature": np.asarray(cart_rotation),
        "rotation_force_coordinate_term": np.asarray(correction),
        "rotation_covariant_curvature": np.asarray(covariant_rotation),
    }


def _summary(raw: dict[str, np.ndarray]) -> dict[str, object]:
    return {key: _quantiles(value) for key, value in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--model-params", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-anchors", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--mapping", default="ala2_backbone_cb_6")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    from analysis.md.analyze_model_residuals_by_region import _load_model
    from sampling.mapping import get_mapping

    R_all = np.asarray(np.load(args.reference)["R"], dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(len(R_all), min(args.n_anchors, len(R_all)), replace=False))
    anchors = R_all[indices]
    mapping = get_mapping(args.mapping)
    model, params, mask0, species0 = _load_model(
        args.model_config, args.model_params, str(args.reference), mapping.n_beads
    )
    energy_jax = lambda r: model.compute_energy(params, r, mask0, species0)
    grad_jax = jax.grad(energy_jax)
    hess_jax = jax.jacfwd(grad_jax)
    energy_fn = lambda r: float(energy_jax(jnp.asarray(r, dtype=jnp.float32)))
    gradient_fn = lambda r: np.asarray(grad_jax(jnp.asarray(r, dtype=jnp.float32)))
    hessian_fn = lambda r: np.asarray(hess_jax(jnp.asarray(r, dtype=jnp.float32)))

    t0 = time.time()
    raw = analyze_gauge(energy_fn, gradient_fn, hessian_fn, anchors)
    args.outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.outdir / "gauge_raw.npz", **raw, indices=indices)
    summary = {
        "model_config": str(args.model_config),
        "model_params": str(args.model_params),
        "reference": str(args.reference),
        "n_anchors": len(anchors),
        "seed": args.seed,
        "summary": _summary(raw),
        "runtime_s": time.time() - t0,
    }
    write_json(args.outdir / "gauge_summary.json", summary)
    print(jsonable_summary(summary)["summary"])


if __name__ == "__main__":
    main()
