"""Compare model curvature along measured soft/stiff and CV-tangent directions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.common.cli import (
    add_output_argument, add_project_root_argument, configure_cpu_default, resolve_cli_output
)
from analysis.common.paths import repo_root, resolve_input
from analysis.common.provenance import write_manifest
from analysis.physics.stencil_hessian import (
    build_stencil_pairs,
    choose_anchor_indices,
    compute_measured_hvp,
    evaluate_model_hvp,
    load_npz_arrays,
    output_key,
    parse_model_spec,
    summarize_hvp,
)


@dataclass(frozen=True)
class SoftDirection:
    direction: np.ndarray
    measured_hvp: np.ndarray
    retained_norm: np.ndarray
    eigenvalues: np.ndarray | None = None


def dihedral_gradient(coordinates: np.ndarray, indices, step: float = 1e-4) -> np.ndarray:
    """Finite-difference gradient of one mapping dihedral in degrees per Angstrom."""

    from sampling.mapping import dihedral_deg, wrap_deg

    coordinates = np.asarray(coordinates, dtype=np.float64)
    gradient = np.zeros_like(coordinates)
    for bead in range(coordinates.shape[1]):
        for xyz in range(3):
            plus = coordinates.copy()
            minus = coordinates.copy()
            plus[:, bead, xyz] += step
            minus[:, bead, xyz] -= step
            delta = wrap_deg(dihedral_deg(plus, indices) - dihedral_deg(minus, indices))
            gradient[:, bead, xyz] = delta / (2.0 * step)
    return gradient


def build_soft_directions(
    coordinates: np.ndarray, v: np.ndarray, measured_hvp: np.ndarray, mapping
) -> dict[str, SoftDirection]:
    """Construct generalized-eigenvector and projected CV-tangent directions."""

    v_flat = np.asarray(v, dtype=np.float64).reshape(v.shape[0], v.shape[1], -1)
    h_flat = np.asarray(measured_hvp, dtype=np.float64).reshape(measured_hvp.shape[0], measured_hvp.shape[1], -1)
    m = np.einsum("aik,ajk->aij", v_flat, h_flat)
    m = 0.5 * (m + m.transpose(0, 2, 1))
    g = np.einsum("aik,ajk->aij", v_flat, v_flat)

    from scipy.linalg import eigh

    eigenvalues = []
    stiff_directions = []
    stiff_hvp = []
    soft_directions = []
    soft_hvp = []
    for anchor in range(v_flat.shape[0]):
        values, vectors = eigh(m[anchor], g[anchor])
        eigenvalues.append(values)
        for coefficients, target_direction, target_hvp in (
            (vectors[:, -1], stiff_directions, stiff_hvp),
            (vectors[:, 0], soft_directions, soft_hvp),
        ):
            direction = coefficients @ v_flat[anchor]
            norm = np.linalg.norm(direction)
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("generalized-eigenvector direction has zero or non-finite norm")
            target_direction.append(direction / norm)
            target_hvp.append((coefficients @ h_flat[anchor]) / norm)

    result = {
        "stiffest_eigenvector": SoftDirection(
            np.asarray(stiff_directions), np.asarray(stiff_hvp), np.ones(v_flat.shape[0]), np.asarray(eigenvalues)
        ),
        "softest_eigenvector": SoftDirection(
            np.asarray(soft_directions), np.asarray(soft_hvp), np.ones(v_flat.shape[0]), np.asarray(eigenvalues)
        ),
    }

    for cv_name in ("phi", "psi"):
        raw_gradient = dihedral_gradient(coordinates, mapping.cvs[cv_name].bead_indices)
        raw_flat = raw_gradient.reshape(raw_gradient.shape[0], -1)
        raw_norm = np.linalg.norm(raw_flat, axis=1)
        if np.any(~np.isfinite(raw_norm)) or np.any(raw_norm <= 0):
            raise ValueError(f"{cv_name} tangent has a zero or non-finite raw gradient")
        unit_gradient = raw_flat / raw_norm[:, None]
        projected_coefficients = np.stack(
            [np.linalg.solve(g[anchor], v_flat[anchor] @ unit_gradient[anchor]) for anchor in range(v_flat.shape[0])]
        )
        projected = np.einsum("ai,aik->ak", projected_coefficients, v_flat)
        retained = np.linalg.norm(projected, axis=1)
        if np.any(~np.isfinite(retained)) or np.any(retained <= 0):
            raise ValueError(f"{cv_name} tangent projection has zero or non-finite norm")
        result[f"{cv_name}_tangent"] = SoftDirection(
            projected / retained[:, None],
            np.einsum("ai,aik->ak", projected_coefficients, h_flat) / retained[:, None],
            retained,
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meanforce", type=Path, required=True, help="NPZ containing R and F arrays")
    parser.add_argument("--stencil", type=Path, required=True, help="NPZ containing anchor, direction, and multiplier")
    parser.add_argument("--reference", type=Path, required=True, help="reference dataset used to construct the model")
    parser.add_argument(
        "--model", action="append", required=True, metavar="LABEL=CONFIG:PARAMS",
        help="model specification; repeat for each trained model",
    )
    parser.add_argument("--mapping", default="ala2_backbone_cb_6", help="registered mapping containing phi and psi CVs")
    parser.add_argument("--layer", type=float, default=1.0, help="absolute stencil multiplier for +/- pairs")
    parser.add_argument("--n-anchors", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260820)
    add_output_argument(parser)
    add_project_root_argument(parser)
    return parser


def run(args: argparse.Namespace) -> dict:
    project_root = repo_root(args.project_root)
    outdir = resolve_cli_output(args, label="soft_subspace")
    meanforce_path = resolve_input(args.meanforce, base=project_root, label="mean-force labels")
    stencil_path = resolve_input(args.stencil, base=project_root, label="stencil states")
    reference_path = resolve_input(args.reference, base=project_root, label="reference dataset")

    meanforce = load_npz_arrays(meanforce_path)
    stencil = load_npz_arrays(stencil_path)
    pairs = build_stencil_pairs(meanforce, stencil, layer=args.layer)
    selected = choose_anchor_indices(pairs.n_anchors, args.n_anchors, args.seed)
    measured = compute_measured_hvp(meanforce, pairs, selected)

    from sampling.mapping import get_mapping

    mapping = get_mapping(args.mapping)
    directions = build_soft_directions(measured.r0, measured.v, measured.hvp, mapping)

    model_specs = []
    labels = set()
    keys = set()
    for raw_spec in args.model:
        label, config, params = parse_model_spec(raw_spec)
        key = output_key(label)
        if label in labels or key in keys:
            raise ValueError(f"model labels must be unique after NPZ sanitization: {label!r}")
        labels.add(label)
        keys.add(key)
        model_specs.append((label, key, resolve_input(config, base=project_root, label=f"{label} config"),
                            resolve_input(params, base=project_root, label=f"{label} parameters")))

    from analysis.md.analyze_model_residuals_by_region import _load_model

    arrays = {"anchor_ids": measured.anchor_ids, "anchor_rows": measured.anchor_rows, "directions": pairs.directions,
              "r0": measured.r0}
    result = {
        "n_anchors": int(measured.anchor_ids.size),
        "n_directions": pairs.n_directions,
        "layer": float(args.layer),
        "seed": int(args.seed),
        "mapping": args.mapping,
        "direction_families": {},
        "models": {},
    }
    for family, data in directions.items():
        family_key = output_key(family)
        arrays[f"direction_{family_key}"] = data.direction.reshape(data.direction.shape[0], -1)
        arrays[f"hvp_measured_{family_key}"] = data.measured_hvp
        arrays[f"retained_norm_{family_key}"] = data.retained_norm
        if data.eigenvalues is not None:
            arrays[f"eigenvalues_{family_key}"] = data.eigenvalues
        result["direction_families"][family] = {
            "measured_hvp_norm_median": float(np.median(np.linalg.norm(data.measured_hvp, axis=-1))),
            "retained_norm_median": float(np.median(data.retained_norm)),
        }

    for label, key, config_path, params_path in model_specs:
        model, params, mask, species = _load_model(
            str(config_path), str(params_path), str(reference_path), measured.r0.shape[1]
        )
        model_result = {}
        for family, data in directions.items():
            model_direction = data.direction.reshape(
                data.direction.shape[0], 1, *measured.r0.shape[1:]
            )
            predicted = evaluate_model_hvp(
                model, params, measured.r0, model_direction, mask, species
            )[:, 0]
            family_key = output_key(family)
            arrays[f"hvp_model_{key}_{family_key}"] = predicted
            model_result[family] = summarize_hvp(data.measured_hvp, predicted)
        result["models"][label] = model_result

    (outdir / "soft_subspace.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(outdir / "soft_subspace.npz", **arrays)
    write_manifest(
        outdir,
        inputs={
            "meanforce": meanforce_path,
            "stencil": stencil_path,
            "reference": reference_path,
            "models": {label: {"config": config, "params": params} for label, _, config, params in model_specs},
        },
        parameters={"mapping": args.mapping, "layer": args.layer, "n_anchors": args.n_anchors, "seed": args.seed},
        module="analysis.physics.soft_subspace",
        extra={"result_json": outdir / "soft_subspace.json", "result_npz": outdir / "soft_subspace.npz"},
    )
    return result


def main() -> None:
    configure_cpu_default()
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
