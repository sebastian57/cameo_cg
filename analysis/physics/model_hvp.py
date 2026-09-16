"""Compare measured stencil HVPs with one or more trained models.

Inputs are explicit so the command can be reused for any compatible force
stencil and model set::

    python -m analysis.physics.model_hvp \
        --meanforce labels.npz --stencil stencil_states.npz \
        --reference reference.npz --model v51=config.yaml:params.pkl \
        --outdir local_work/analysis/model_hvp

The measured product uses the realized plus/minus displacement.  The model
product is the directional derivative of the model energy gradient.
"""

from __future__ import annotations

import argparse
import json
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meanforce", type=Path, required=True, help="NPZ containing R and F arrays")
    parser.add_argument("--stencil", type=Path, required=True, help="NPZ containing anchor, direction, and multiplier")
    parser.add_argument("--reference", type=Path, required=True, help="reference dataset used to construct the model")
    parser.add_argument(
        "--model", action="append", required=True, metavar="LABEL=CONFIG:PARAMS",
        help="model specification; repeat for each trained model",
    )
    parser.add_argument("--layer", type=float, default=1.0, help="absolute stencil multiplier for +/- pairs")
    parser.add_argument("--n-anchors", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260820)
    add_output_argument(parser)
    add_project_root_argument(parser)
    return parser


def run(args: argparse.Namespace) -> dict:
    project_root = repo_root(args.project_root)
    outdir = resolve_cli_output(args, label="model_hvp")
    meanforce_path = resolve_input(args.meanforce, base=project_root, label="mean-force labels")
    stencil_path = resolve_input(args.stencil, base=project_root, label="stencil states")
    reference_path = resolve_input(args.reference, base=project_root, label="reference dataset")

    meanforce = load_npz_arrays(meanforce_path)
    stencil = load_npz_arrays(stencil_path)
    pairs = build_stencil_pairs(meanforce, stencil, layer=args.layer)
    selected = choose_anchor_indices(pairs.n_anchors, args.n_anchors, args.seed)
    measured = compute_measured_hvp(meanforce, pairs, selected)

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

    arrays = {
        "anchor_ids": measured.anchor_ids,
        "anchor_rows": measured.anchor_rows,
        "directions": pairs.directions,
        "r0": measured.r0,
        "v": measured.v,
        "hvp_measured": measured.hvp,
        "realized_displacement": measured.realized_displacement,
    }
    if measured.noise is not None:
        arrays["hvp_noise"] = measured.noise

    summaries = {
        "n_anchors": int(measured.anchor_ids.size),
        "n_directions": pairs.n_directions,
        "layer": float(args.layer),
        "seed": int(args.seed),
        "measured_hvp_norm_median": float(np.median(np.linalg.norm(measured.hvp, axis=(-2, -1)))),
    }
    if measured.noise is not None:
        summaries["measured_noise_norm_median"] = float(
            np.median(np.linalg.norm(measured.noise, axis=(-2, -1)))
        )

    model_results = {}
    for label, key, config_path, params_path in model_specs:
        model, params, mask, species = _load_model(
            str(config_path), str(params_path), str(reference_path), measured.r0.shape[1]
        )
        predicted = evaluate_model_hvp(model, params, measured.r0, measured.v, mask, species)
        arrays[f"hvp_model_{key}"] = predicted
        model_results[label] = summarize_hvp(measured.hvp, predicted)

    summaries["models"] = model_results
    (outdir / "model_hvp.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(outdir / "model_hvp.npz", **arrays)
    write_manifest(
        outdir,
        inputs={
            "meanforce": meanforce_path,
            "stencil": stencil_path,
            "reference": reference_path,
            "models": {label: {"config": config, "params": params} for label, _, config, params in model_specs},
        },
        parameters={"layer": args.layer, "n_anchors": args.n_anchors, "seed": args.seed},
        module="analysis.physics.model_hvp",
        extra={"result_json": outdir / "model_hvp.json", "result_npz": outdir / "model_hvp.npz"},
    )
    return summaries


def main() -> None:
    configure_cpu_default()
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
