#!/usr/bin/env python3
"""Plot the corrected latent over-sharpening diagnostic with flow-checkpoint bands.

Inputs and output location match analysis.latent.sharpness.  The figure reports the
free-energy error relative to the analytic N(0,1) axis baseline and effective volume
relative to the AA reference.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis.common.paths import repo_root, resolve_input, resolve_output
from analysis.common.provenance import write_manifest
from analysis.latent.diagnosis import BASIN_COLOR, GRID, INK, INK2, MUTED, SURFACE
from analysis.latent.sharpness import _flow_tags, _parser, _split_ensemble, entropy_2d


def main(argv: list[str] | None = None) -> None:
    ap = _parser()
    a = ap.parse_args(argv)
    project_root = repo_root(a.project_root)
    outdir = resolve_output(a.outdir, base=project_root)
    reference = resolve_input(a.reference, base=project_root, label="reference")
    bias_npz = resolve_input(a.bias_npz, base=project_root, label="bias artifact")
    flow_dir = resolve_input(a.flow_dir, base=project_root, label="flow directory")
    ensemble_files, patterns = _split_ensemble(a.ensemble, project_root)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sampling.biases.tica_regional import SmoothTICABias
    from sampling.flow_density import load_flow, to_latent
    from sampling.mapping import get_mapping
    from analysis.latent.diagnosis import assign_basins, load_replicas

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
        "axes.grid": True, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.dpi": 150})

    mapping = get_mapping(a.mapping)
    bias = SmoothTICABias.load(bias_npz)
    kT = float(bias.kbt_kcal_mol)
    R = {"reference": np.asarray(np.load(reference, allow_pickle=False)["R"], np.float64)}
    for label, paths in ensemble_files.items():
        R[label] = load_replicas(paths, a.discard_frac, a.max_bond)[0]
    reg_ref, _, _ = assign_basins(R["reference"], mapping)
    edges = np.linspace(-a.grid_lim, a.grid_lim, a.bins + 1)
    s_edges = np.linspace(-a.grid_lim, a.grid_lim, a.axis_bins + 1)
    s_c = 0.5 * (s_edges[1:] + s_edges[:-1])
    phi_s = np.exp(-0.5 * s_c ** 2) / np.sqrt(2.0 * np.pi)
    tags = _flow_tags(flow_dir, a.flow_primary, a.flow_seed, a.flow_crosscheck)

    curves = {label: [] for label in R}
    ent = {label: [] for label in R}
    for tag in tags:
        params, cfg = load_flow(flow_dir / f"flow_{tag}.npz")
        d = cfg.n_dims
        u = {label: np.asarray(to_latent(
            params, cfg, np.asarray(bias.projection.transform(Rl), np.float64)[:, :d]
        ), np.float64) for label, Rl in R.items()}
        for label in R:
            ent[label].append(entropy_2d(u[label], edges))
        mu_b = u["reference"][reg_ref == "beta"].mean(0)
        mu_a = u["reference"][reg_ref == "alphaR"].mean(0)
        e_ax = (mu_a - mu_b) / np.linalg.norm(mu_a - mu_b)
        for label in R:
            h, _ = np.histogram(u[label] @ e_ax, bins=s_edges, density=True)
            curves[label].append(-kT * np.log(np.maximum(h, 1e-12) / phi_s))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.2, 4.3), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax1.axhline(0, color=INK, lw=1.6, ls="--", label="analytic N(0,1)")
    palette = [BASIN_COLOR["beta"], BASIN_COLOR["alphaR"],
               BASIN_COLOR["alphaL"], "#4c78a8", "#7f7f7f"]
    colors = {"reference": INK2}
    colors.update({label: palette[i % len(palette)]
                   for i, label in enumerate(ensemble_files)})
    for label in R:
        A = np.asarray(curves[label])
        color = colors[label]
        ax1.fill_between(s_c, A.min(0), A.max(0), color=color, alpha=0.18, lw=0)
        ax1.plot(s_c, A.mean(0), color=color, lw=2.0,
                 label="AA reference (flow spread)" if label == "reference" else label)
    ax1.axhspan(-0.5 * kT, 0.5 * kT, color=MUTED, alpha=0.10, lw=0)
    ax1.set_xlim(-3, 3)
    ax1.set_xlabel("$s = u\\cdot e_{\\beta\\rightarrow\\alpha_R}$")
    ax1.set_ylabel("$\\Delta F$  (kcal/mol)")
    ax1.set_title("Free-energy error vs analytic reference")
    ax1.legend(fontsize=8, loc="lower left")

    labs = ["reference"] + list(ensemble_files)
    H_ref = float(np.mean(ent["reference"]))
    vals = [100 * np.exp(np.mean(ent[label]) - H_ref) for label in labs]
    errs = [100 * np.exp(np.mean(ent[label]) - H_ref)
            * (np.std(ent[label], ddof=1) if len(ent[label]) > 1 else 0.0)
            for label in labs]
    ax2.bar(range(len(labs)), vals, 0.6, color=[colors[label] for label in labs],
            yerr=errs, ecolor=MUTED, capsize=4)
    ax2.axhline(100, color=INK, lw=1.6, ls="--")
    ax2.set_xticks(range(len(labs)))
    ax2.set_xticklabels(labs, rotation=25, ha="right")
    ax2.set_ylim(0, max(115, max(vals) * 1.2))
    ax2.set_ylabel("effective volume, % of reference")
    ax2.set_title("Effective volume exp(H(u))\n(over-sharpening = less volume)")
    fig_path = outdir / "fig7_sharpness.png"
    fig.savefig(fig_path, bbox_inches="tight")

    summary_path = outdir / "fig7_sharpness.json"
    summary_path.write_text(json.dumps({
        "flows": tags,
        "effective_volume_percent": dict(zip(labs, vals)),
        "effective_volume_percent_sd": dict(zip(labs, errs)),
        "kT": kT,
    }, indent=2) + "\n")
    manifest = write_manifest(
        outdir,
        inputs={"reference": reference, "bias_npz": bias_npz, "flow_dir": flow_dir,
                "ensembles": {label: paths for label, paths in ensemble_files.items()}},
        parameters=vars(a),
        module="analysis.latent.fig7_sharpness",
        extra={"figure": str(fig_path), "summary": str(summary_path), "patterns": patterns},
    )
    print(f"wrote {fig_path}, {summary_path}, and {manifest}")


if __name__ == "__main__":
    main()
