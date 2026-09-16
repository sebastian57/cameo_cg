#!/usr/bin/env python3
"""Measure latent over-sharpening against an analytic N(0,I) reference.

This ports the historical latent-ensemble sharpness diagnostic into the repository
analysis namespace. Every ensemble, reference, bias, flow directory, and output
directory is explicit; no result is written beside the source code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.common.cli import add_project_root_argument
from analysis.common.paths import repo_root, resolve_glob, resolve_input, resolve_output
from analysis.common.provenance import write_manifest
from analysis.latent.diagnosis import assign_basins, load_replicas

H_GAUSS_2D = float(np.log(2.0 * np.pi * np.e))


def entropy_2d(u: np.ndarray, edges: np.ndarray) -> float:
    H, _, _ = np.histogram2d(u[:, 0], u[:, 1], bins=[edges, edges])
    p = H / H.sum()
    area = (edges[1] - edges[0]) ** 2
    m = p > 0
    return float(-np.sum(p[m] * np.log(p[m] / area)))


def _split_ensemble(values: list[str], project_root: Path) -> tuple[dict[str, list[Path]], dict[str, str]]:
    files: dict[str, list[Path]] = {}
    patterns: dict[str, str] = {}
    for value in values:
        label, sep, pattern = value.partition("=")
        if not sep or not label or not pattern:
            raise SystemExit("--ensemble must be LABEL=trajectory-glob")
        files[label] = resolve_glob(pattern, base=project_root, label=f"ensemble {label}")
        patterns[label] = pattern
    return files, patterns


def _flow_tags(flow_dir: Path, primary: str | None, seeds: list[str] | None,
                crosscheck: list[str]) -> list[str]:
    requested = list(seeds or [])
    if primary and primary not in requested:
        requested.insert(0, primary)
    if not requested:
        requested = sorted(p.stem.removeprefix("flow_")
                           for p in flow_dir.glob("flow_*.npz"))
    for tag in crosscheck:
        if tag not in requested:
            requested.append(tag)
    tags = [tag for tag in requested if (flow_dir / f"flow_{tag}.npz").exists()]
    if not tags:
        raise SystemExit(f"no requested flow_*.npz files found in {flow_dir}")
    return tags


def collect_metrics(*, reference: Path, bias_npz: Path, flow_dir: Path,
                    ensemble_files: dict[str, list[Path]], flow_primary: str | None,
                    flow_seeds: list[str] | None, flow_crosscheck: list[str],
                    mapping_name: str, discard_frac: float, max_bond: float,
                    grid_lim: float, bins: int, axis_bins: int) -> dict:
    from sampling.biases.tica_regional import SmoothTICABias
    from sampling.flow_density import load_flow, to_latent
    from sampling.mapping import get_mapping

    mapping = get_mapping(mapping_name)
    bias = SmoothTICABias.load(bias_npz)
    kT = float(bias.kbt_kcal_mol)
    R = {"reference": np.asarray(np.load(reference, allow_pickle=False)["R"], np.float64)}
    for label, paths in ensemble_files.items():
        R[label] = load_replicas(paths, discard_frac, max_bond)[0]
    reg_ref, _, _ = assign_basins(R["reference"], mapping)

    edges = np.linspace(-grid_lim, grid_lim, bins + 1)
    s_edges = np.linspace(-grid_lim, grid_lim, axis_bins + 1)
    s_c = 0.5 * (s_edges[1:] + s_edges[:-1])
    phi_s = np.exp(-0.5 * s_c ** 2) / np.sqrt(2.0 * np.pi)
    tags = _flow_tags(flow_dir, flow_primary, flow_seeds, flow_crosscheck)

    per_flow = {label: [] for label in R}
    dfmin = {label: [] for label in ensemble_files}
    dfloc = {label: [] for label in ensemble_files}
    dfwid = {label: [] for label in ensemble_files}
    ridge = {label: [] for label in ensemble_files}

    for tag in tags:
        params, cfg = load_flow(flow_dir / f"flow_{tag}.npz")
        d = cfg.n_dims
        u = {label: np.asarray(to_latent(
            params, cfg, np.asarray(bias.projection.transform(Rl), np.float64)[:, :d]
        ), np.float64) for label, Rl in R.items()}
        for label in R:
            per_flow[label].append(entropy_2d(u[label], edges))

        mu_b = u["reference"][reg_ref == "beta"].mean(0)
        mu_a = u["reference"][reg_ref == "alphaR"].mean(0)
        e_ax = (mu_a - mu_b) / np.linalg.norm(mu_a - mu_b)

        Href, _, _ = np.histogram2d(u["reference"][:, 0], u["reference"][:, 1],
                                    bins=[edges, edges])
        pref = Href / Href.sum()
        for label in ensemble_files:
            h, _ = np.histogram(u[label] @ e_ax, bins=s_edges, density=True)
            dF = -kT * np.log(np.maximum(h, 1e-12) / phi_s)
            core = np.abs(s_c) < 2.5
            i = int(np.argmin(np.where(core, dF, np.inf)))
            dfmin[label].append(float(dF[i]))
            dfloc[label].append(float(s_c[i]))
            dfwid[label].append(float(((dF < -0.5 * kT) & core).sum()
                                      * (s_c[1] - s_c[0])))

            Hm, _, _ = np.histogram2d(u[label][:, 0], u[label][:, 1], bins=[edges, edges])
            pm = Hm / Hm.sum()
            hot = (pm > 2.0 * pref) & (pm > 1e-5)
            ridge[label].append((float(pm[hot].sum()), float(pref[hot].sum())))

    def ms(values: list[float]) -> tuple[float, float]:
        values = np.asarray(values, float)
        return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0

    out = {"n_flows": len(tags), "flows": tags, "kT": kT,
           "analytic_entropy_2d_nats": H_GAUSS_2D, "entropy": {},
           "free_energy_error": {}, "ridge": {}, "axis_centers": s_c.tolist()}
    H_ref_m, _ = ms(per_flow["reference"])
    for label in R:
        m, s = ms(per_flow[label])
        out["entropy"][label] = {
            "H_mean": m, "H_sd": s, "A_eff": float(np.exp(m)),
            "A_rel_to_reference": float(np.exp(m - H_ref_m)),
        }
    for label in ensemble_files:
        a, b = ms(dfmin[label])
        c, d = ms(dfloc[label])
        e, f = ms(dfwid[label])
        out["free_energy_error"][label] = {
            "min_dF": a, "min_dF_sd": b, "at_s": c, "at_s_sd": d,
            "width_below_half_kT": e, "width_sd": f,
        }
        pm = np.array([x[0] for x in ridge[label]])
        pr = np.array([x[1] for x in ridge[label]])
        out["ridge"][label] = {
            "model_mass": float(pm.mean()),
            "model_mass_sd": float(pm.std(ddof=1)) if len(pm) > 1 else 0.0,
            "reference_mass": float(pr.mean()),
            "concentration": float((pm / np.maximum(pr, 1e-9)).mean()),
        }
    out["_internal"] = {"per_flow": per_flow, "dfmin": dfmin, "dfloc": dfloc,
                        "dfwid": dfwid, "ridge": ridge, "s_centers": s_c}
    return out


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--bias-npz", type=Path, required=True)
    ap.add_argument("--flow-dir", type=Path, required=True)
    ap.add_argument("--ensemble", action="append", required=True,
                    help="LABEL=trajectory glob; repeatable")
    ap.add_argument("--flow-primary", default=None)
    ap.add_argument("--flow-seed", action="append", default=None)
    ap.add_argument("--flow-crosscheck", action="append", default=[])
    ap.add_argument("--mapping", default="ala2_backbone_cb_6")
    ap.add_argument("--discard-frac", type=float, default=0.20)
    ap.add_argument("--max-bond", type=float, default=3.0)
    ap.add_argument("--grid-lim", type=float, default=4.5)
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--axis-bins", type=int, default=120)
    add_project_root_argument(ap)
    return ap


def main(argv: list[str] | None = None) -> None:
    ap = _parser()
    a = ap.parse_args(argv)
    project_root = repo_root(a.project_root)
    outdir = resolve_output(a.outdir, base=project_root)
    reference = resolve_input(a.reference, base=project_root, label="reference")
    bias_npz = resolve_input(a.bias_npz, base=project_root, label="bias artifact")
    flow_dir = resolve_input(a.flow_dir, base=project_root, label="flow directory")
    ensemble_files, patterns = _split_ensemble(a.ensemble, project_root)
    result = collect_metrics(
        reference=reference, bias_npz=bias_npz, flow_dir=flow_dir,
        ensemble_files=ensemble_files, flow_primary=a.flow_primary,
        flow_seeds=a.flow_seed, flow_crosscheck=a.flow_crosscheck,
        mapping_name=a.mapping, discard_frac=a.discard_frac, max_bond=a.max_bond,
        grid_lim=a.grid_lim, bins=a.bins, axis_bins=a.axis_bins,
    )
    result.pop("_internal")
    print(f"flows: {result['flows']}")
    print("=== EFFECTIVE VOLUME ===")
    for label, row in result["entropy"].items():
        print(f"  {label:16s} H={row['H_mean']:.4f} +/- {row['H_sd']:.4f} "
              f"A_rel={100*row['A_rel_to_reference']:.1f}%")
    print("=== FREE-ENERGY ERROR ===")
    for label, row in result["free_energy_error"].items():
        print(f"  {label:16s} dF_min={row['min_dF']:+.3f} +/- {row['min_dF_sd']:.3f} "
              f"at s={row['at_s']:+.3f}")
    print("=== RIDGE MASS ===")
    for label, row in result["ridge"].items():
        print(f"  {label:16s} model={100*row['model_mass']:.2f}% "
              f"reference={100*row['reference_mass']:.2f}%")
    summary_path = outdir / "sharpness.json"
    summary_path.write_text(json.dumps(result, indent=2, default=float) + "\n")
    manifest = write_manifest(
        outdir,
        inputs={"reference": reference, "bias_npz": bias_npz, "flow_dir": flow_dir,
                "ensembles": {label: paths for label, paths in ensemble_files.items()}},
        parameters=vars(a),
        module="analysis.latent.sharpness",
        extra={"summary": str(summary_path), "patterns": patterns},
    )
    print(f"wrote {summary_path} and {manifest}")


if __name__ == "__main__":
    main()
