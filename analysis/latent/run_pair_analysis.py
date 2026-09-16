#!/usr/bin/env python3
"""Run the canonical latent diagnostic suite for arbitrary MD ensembles.

The driver launches the repository modules for diagnosis, sharpness, figure 7, and
optional ridge identity.  All source data and output locations are explicit, and each
child analysis writes its own provenance manifest.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from analysis.common.cli import add_project_root_argument
from analysis.common.paths import repo_root, resolve_input, resolve_output


def _pairs(values: list[str], flag: str) -> dict[str, str]:
    result = {}
    for value in values:
        label, sep, payload = value.partition("=")
        if not sep or not label or not payload:
            raise SystemExit(f"{flag} must be LABEL=VALUE")
        result[label] = payload
    return result


def _common(a, project_root: Path) -> list[str]:
    result = [
        "--reference", str(resolve_input(a.reference, base=project_root, label="reference")),
        "--bias-npz", str(resolve_input(a.bias_npz, base=project_root, label="bias artifact")),
        "--flow-dir", str(resolve_input(a.flow_dir, base=project_root, label="flow directory")),
        "--mapping", a.mapping, "--discard-frac", str(a.discard_frac),
        "--max-bond", str(a.max_bond), "--grid-lim", str(a.grid_lim),
        "--bins", str(a.bins), "--project-root", str(project_root),
    ]
    if a.flow_primary:
        result += ["--flow-primary", a.flow_primary]
    if a.flow_crosscheck:
        result += ["--flow-crosscheck", a.flow_crosscheck]
    for seed in a.flow_seed or []:
        result += ["--flow-seed", seed]
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--bias-npz", type=Path, required=True)
    ap.add_argument("--flow-dir", type=Path, required=True)
    ap.add_argument("--run", action="append", required=True, help="LABEL=trajectory glob")
    ap.add_argument("--trainset", action="append", default=[], help="LABEL=dataset NPZ")
    ap.add_argument("--flow-primary", default=None)
    ap.add_argument("--flow-seed", action="append", default=None)
    ap.add_argument("--flow-crosscheck", default=None)
    ap.add_argument("--mapping", default="ala2_backbone_cb_6")
    ap.add_argument("--discard-frac", type=float, default=0.20)
    ap.add_argument("--max-bond", type=float, default=3.0)
    ap.add_argument("--grid-lim", type=float, default=4.5)
    ap.add_argument("--bins", type=int, default=80)
    add_project_root_argument(ap)
    a = ap.parse_args(argv)
    project_root = repo_root(a.project_root)
    outdir = resolve_output(a.outdir, base=project_root)
    runs = _pairs(a.run, "--run")
    trains = _pairs(a.trainset, "--trainset")
    missing = sorted(set(trains) - set(runs))
    if missing:
        raise SystemExit(f"training sets have no matching runs: {missing}")

    common = _common(a, project_root)
    run_args = []
    for label, pattern in runs.items():
        run_args += ["--ensemble", f"{label}={pattern}"]

    def launch(module: str, extra: list[str]) -> None:
        command = [sys.executable, "-m", module] + extra
        print("running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=project_root, check=True)

    launch("analysis.latent.diagnosis",
           ["--outdir", str(outdir / "diagnosis")] + common + run_args)
    launch("analysis.latent.sharpness",
           ["--outdir", str(outdir / "sharpness")] + common + run_args)
    launch("analysis.latent.fig7_sharpness",
           ["--outdir", str(outdir / "fig7")] + common + run_args)

    for label, dataset in trains.items():
        launch("analysis.latent.ridge_identity", [
            "--outdir", str(outdir / f"ridge_{label}"),
            "--reference", common[1],
            "--bias-npz", common[3],
            "--flow-dir", common[5],
            "--ensemble", f"{label}={runs[label]}",
            "--training-dataset", str(resolve_input(dataset, base=project_root,
                                                    label=f"training dataset {label}")),
            "--mapping", a.mapping, "--discard-frac", str(a.discard_frac),
            "--max-bond", str(a.max_bond), "--grid-lim", str(a.grid_lim),
            "--bins", str(a.bins), "--project-root", str(project_root),
        ] + ([ "--flow-primary", a.flow_primary] if a.flow_primary else [])
          + sum((["--flow-seed", seed] for seed in (a.flow_seed or [])), [])
    )
    print(f"all latent outputs: {outdir}")


if __name__ == "__main__":
    main()
