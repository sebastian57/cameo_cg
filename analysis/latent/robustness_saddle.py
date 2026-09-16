#!/usr/bin/env python3
"""Check the historical saddle-shift diagnostic across flow checkpoints.

The calculation is intentionally parameterized by one explicit ensemble.  The original
script used a fixed v2 trajectory and machine-local input constants; this entry point
records all inputs and writes its JSON result and manifest under the requested outdir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.common.cli import add_project_root_argument
from analysis.common.paths import repo_root, resolve_glob, resolve_input, resolve_output
from analysis.common.provenance import write_manifest
from analysis.latent.diagnosis import (
    assign_basins, cov_stats, find_saddle, free_energy, gaussian_2d_on_grid,
    hist2d_density, kl_hist, load_replicas, profile_along,
)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--bias-npz", type=Path, required=True)
    ap.add_argument("--flow-dir", type=Path, required=True)
    ap.add_argument("--ensemble", required=True, help="LABEL=trajectory glob")
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
    a = ap.parse_args(argv)

    project_root = repo_root(a.project_root)
    outdir = resolve_output(a.outdir, base=project_root)
    reference = resolve_input(a.reference, base=project_root, label="reference")
    bias_npz = resolve_input(a.bias_npz, base=project_root, label="bias artifact")
    flow_dir = resolve_input(a.flow_dir, base=project_root, label="flow directory")
    label, separator, pattern = str(a.ensemble).partition("=")
    if not separator or not label or not pattern:
        raise SystemExit("--ensemble must be LABEL=trajectory-glob")
    model_files = resolve_glob(pattern, base=project_root, label=f"ensemble {label}")

    from sampling.biases.tica_regional import SmoothTICABias
    from sampling.flow_density import load_flow, to_latent
    from sampling.mapping import get_mapping
    from analysis.latent.sharpness import _flow_tags

    mapping = get_mapping(a.mapping)
    bias = SmoothTICABias.load(bias_npz)
    kT = float(bias.kbt_kcal_mol)
    R_ref = np.asarray(np.load(reference, allow_pickle=False)["R"], np.float64)
    reg_ref, _, _ = assign_basins(R_ref, mapping)
    R_mod, _ = load_replicas(model_files, a.discard_frac, a.max_bond)
    reg_mod, _, _ = assign_basins(R_mod, mapping)
    z_ref = np.asarray(bias.projection.transform(R_ref), np.float64)
    z_mod = np.asarray(bias.projection.transform(R_mod), np.float64)

    s_edges = np.linspace(-a.grid_lim, a.grid_lim, a.axis_bins + 1)
    s_centers = 0.5 * (s_edges[1:] + s_edges[:-1])
    edges = np.linspace(-a.grid_lim, a.grid_lim, a.bins + 1)
    area = (edges[1] - edges[0]) ** 2
    p_gauss = gaussian_2d_on_grid(edges)
    tags = _flow_tags(flow_dir, a.flow_primary, a.flow_seed, a.flow_crosscheck)

    rows = []
    for tag in tags:
        params, cfg = load_flow(flow_dir / f"flow_{tag}.npz")
        d = cfg.n_dims
        u_ref = np.asarray(to_latent(params, cfg, z_ref[:, :d]), np.float64)
        u_mod = np.asarray(to_latent(params, cfg, z_mod[:, :d]), np.float64)
        mu_b = u_ref[reg_ref == "beta"].mean(axis=0)
        mu_a = u_ref[reg_ref == "alphaR"].mean(axis=0)
        sep = float(np.linalg.norm(mu_a - mu_b))
        e_axis = (mu_a - mu_b) / sep
        s_b, s_a = float(mu_b @ e_axis), float(mu_a @ e_axis)
        _, h_r = profile_along(u_ref, e_axis, s_edges)
        _, h_m = profile_along(u_mod, e_axis, s_edges)
        F_r, F_m = free_energy(h_r, kT), free_energy(h_m, kT)
        sr, Fr = find_saddle(s_centers, F_r, min(s_b, s_a), max(s_b, s_a))
        sm, Fm = find_saddle(s_centers, F_m, min(s_b, s_a), max(s_b, s_a))
        gv_b = (cov_stats(u_mod[reg_mod == "beta"])["generalized_variance"]
                / cov_stats(u_ref[reg_ref == "beta"])["generalized_variance"])
        gv_a = (cov_stats(u_mod[reg_mod == "alphaR"])["generalized_variance"]
                / cov_stats(u_ref[reg_ref == "alphaR"])["generalized_variance"])
        shift_b = float(np.linalg.norm(u_mod[reg_mod == "beta"].mean(0)
                                       - u_ref[reg_ref == "beta"].mean(0)))
        shift_a = float(np.linalg.norm(u_mod[reg_mod == "alphaR"].mean(0)
                                       - u_ref[reg_ref == "alphaR"].mean(0)))
        rows.append(dict(
            flow=tag, n_dims=int(d), separation=sep, saddle_ref=sr, saddle_mod=sm,
            saddle_shift=sm - sr, shift_frac_of_separation=(sm - sr) / sep,
            barrier_ref=Fr, barrier_mod=Fm, barrier_ratio=Fm / max(Fr, 1e-9),
            centroid_shift_beta=shift_b, centroid_shift_alphaR=shift_a,
            gv_ratio_beta=gv_b, gv_ratio_alphaR=gv_a,
            kl_ref_floor=kl_hist(hist2d_density(u_ref, edges), p_gauss, area),
        ))

    if not rows:
        raise SystemExit("no valid flow checkpoints were selected")
    arr = lambda key: np.array([row[key] for row in rows], float)
    small = [row for row in rows if row["flow"].startswith("small_seed")]
    small_arr = lambda key: np.array([row[key] for row in small], float)
    def mean_sd(values):
        values = np.asarray(values, float)
        return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0
    stats = {
        "n_flows": len(rows),
        "small_seeds": {},
        "all_flows": {
            "saddle_shift_min": float(arr("saddle_shift").min()),
            "saddle_shift_max": float(arr("saddle_shift").max()),
            "all_negative": bool((arr("saddle_shift") < 0).all()),
            "centroid_shift_beta_max": float(arr("centroid_shift_beta").max()),
            "centroid_shift_alphaR_max": float(arr("centroid_shift_alphaR").max()),
        },
        "rows": rows,
    }
    for key in ("saddle_shift", "shift_frac_of_separation", "barrier_ratio"):
        stats["small_seeds"][f"{key}_mean"], stats["small_seeds"][f"{key}_sd"] = (
            mean_sd(small_arr(key)) if small else (float("nan"), float("nan")))
    for row in rows:
        print(f"{row['flow']:18s} saddle {row['saddle_ref']:+.3f} -> "
              f"{row['saddle_mod']:+.3f}, shift {row['saddle_shift']:+.3f}, "
              f"barrier ratio {row['barrier_ratio']:.2f}")
    summary_path = outdir / "robustness_saddle.json"
    summary_path.write_text(json.dumps(stats, indent=2, default=float) + "\n")
    manifest = write_manifest(
        outdir,
        inputs={"reference": reference, "bias_npz": bias_npz, "flow_dir": flow_dir,
                "ensemble": model_files},
        parameters=vars(a),
        module="analysis.latent.robustness_saddle",
        extra={"summary": str(summary_path), "ensemble_label": label},
    )
    print(f"wrote {summary_path} and {manifest}")


if __name__ == "__main__":
    main()
