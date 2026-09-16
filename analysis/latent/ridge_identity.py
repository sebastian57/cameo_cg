#!/usr/bin/env python3
"""What IS the ridge, physically -- and did the training set put it there?

TWO QUESTIONS, BOTH DECIDABLE FROM DATA

  Q1. Is the over-populated ridge the beta/alphaR BORDER, or is it inside a basin?
      Answered by taking the latent cells where the FM model exceeds the reference by >2x and
      reading off (a) which Ramachandran basin the model frames in those cells belong to, and
      (b) which basin the REFERENCE frames in the SAME cells belong to. A border ridge is a
      mixture / transition-labelled; an in-basin ridge is dominated by one label.

  Q2. Did the ENHANCED SAMPLING put the ridge there?
      The v2 discover stage chose 42,000 states by farthest-point over the DISCOVERED region,
      which deliberately puts them in reference-density regions ~41.5x lower than typical
      reference frames. If those training states are themselves concentrated on the ridge, then
      the model is sharp exactly where it was over-supervised, and the causal story closes:
          over-represented region -> capacity allocated there -> artificial well -> trapping.
      Measured as an ENRICHMENT: (fraction of training states on the ridge) / (fraction of
      reference mass on the ridge). Enrichment >> 1 supports the story; ~1 refutes it.

WHY THE ANSWER MATTERS FOR WHAT TO DO NEXT
      If the ridge is over-supervised already, adding MORE labels there makes it worse, not
      better -- the intuition "sample the barrier so structures don't get stuck" would be
      exactly backwards. The fix would be to balance, reweight or smooth, not to add.
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
    BASIN_COLOR, BASIN_MARKER, BASINS, GRID, INK, INK2, MUTED, SEQ_STEPS,
    SURFACE, assign_basins, load_replicas,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--bias-npz", type=Path, required=True)
    ap.add_argument("--flow-dir", type=Path, required=True)
    ap.add_argument("--ensemble", required=True, help="LABEL=trajectory glob")
    ap.add_argument("--training-dataset", type=Path, required=True)
    ap.add_argument("--flow-primary", default="small_seed0")
    ap.add_argument("--flow-seed", action="append", default=None)
    ap.add_argument("--discard-frac", type=float, default=0.20)
    ap.add_argument("--max-bond", type=float, default=3.0)
    ap.add_argument("--mapping", default="ala2_backbone_cb_6")
    ap.add_argument("--grid-lim", type=float, default=4.5)
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--ridge-threshold", type=float, default=2.0)
    add_project_root_argument(ap)
    a = ap.parse_args()

    project_root = repo_root(a.project_root)
    a.outdir = resolve_output(a.outdir, base=project_root)
    reference = resolve_input(a.reference, base=project_root, label="reference")
    bias_npz = resolve_input(a.bias_npz, base=project_root, label="bias artifact")
    flow_dir = resolve_input(a.flow_dir, base=project_root, label="flow directory")
    training_dataset = resolve_input(a.training_dataset, base=project_root, label="training dataset")
    label, separator, pattern = str(a.ensemble).partition("=")
    if not separator or not label or not pattern:
        raise SystemExit("--ensemble must be LABEL=trajectory-glob")
    model_files = resolve_glob(pattern, base=project_root, label=f"ensemble {label}")
    flow_primary = str(a.flow_primary)
    flow_seeds = list(a.flow_seed or [flow_primary])
    if flow_primary not in flow_seeds:
        flow_seeds.insert(0, flow_primary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from sampling.biases.tica_regional import SmoothTICABias
    from sampling.flow_density import load_flow, to_latent
    from sampling.mapping import get_mapping

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

    R_ref = np.asarray(np.load(reference, allow_pickle=False)["R"], np.float64)
    R_mod, _ = load_replicas(model_files, a.discard_frac, a.max_bond)
    R_tr = np.asarray(np.load(training_dataset, allow_pickle=False)["R"], np.float64)
    reg_ref, phi_ref, psi_ref = assign_basins(R_ref, mapping)
    reg_mod, phi_mod, psi_mod = assign_basins(R_mod, mapping)
    reg_tr, phi_tr, psi_tr = assign_basins(R_tr, mapping)

    edges = np.linspace(-a.grid_lim, a.grid_lim, a.bins + 1)
    tags = [t for t in flow_seeds if (flow_dir / f"flow_{t}.npz").exists()]

    agg = {"per_flow": [], "n_train": int(len(R_tr))}
    keep = None
    for tag in tags:
        params, cfg = load_flow(flow_dir / f"flow_{tag}.npz")
        d = cfg.n_dims
        lat = lambda Rx: np.asarray(to_latent(params, cfg,
                                              np.asarray(bias.projection.transform(Rx),
                                                         np.float64)[:, :d]), np.float64)
        u_ref, u_mod, u_tr = lat(R_ref), lat(R_mod), lat(R_tr)

        Hr, _, _ = np.histogram2d(u_ref[:, 0], u_ref[:, 1], bins=[edges, edges])
        Hm, _, _ = np.histogram2d(u_mod[:, 0], u_mod[:, 1], bins=[edges, edges])
        pr, pm = Hr / Hr.sum(), Hm / Hm.sum()
        hot = (pm > a.ridge_threshold * pr) & (pm > 1e-5)

        ix = lambda u: (np.clip(np.digitize(u[:, 0], edges) - 1, 0, len(edges) - 2),
                        np.clip(np.digitize(u[:, 1], edges) - 1, 0, len(edges) - 2))
        on = lambda u: hot[ix(u)]
        on_mod, on_ref, on_tr = on(u_mod), on(u_ref), on(u_tr)

        f_tr, f_ref = float(on_tr.mean()), float(on_ref.mean())
        row = dict(flow=tag,
                   model_mass=float(on_mod.mean()), reference_mass=f_ref,
                   train_frac=f_tr, enrichment=f_tr / max(f_ref, 1e-12),
                   basins_model_on_ridge={b: float((reg_mod[on_mod] == b).mean())
                                          for b in BASINS},
                   basins_reference_on_ridge={b: float((reg_ref[on_ref] == b).mean())
                                              for b in BASINS},
                   basins_train_on_ridge={b: float((reg_tr[on_tr] == b).mean())
                                          for b in BASINS})
        agg["per_flow"].append(row)
        if tag == flow_primary:
            keep = dict(hot=hot, u_ref=u_ref, u_mod=u_mod, u_tr=u_tr,
                        on_mod=on_mod, on_ref=on_ref, on_tr=on_tr, pr=pr, pm=pm)

    ms = lambda k: (float(np.mean([r[k] for r in agg["per_flow"]])),
                    float(np.std([r[k] for r in agg["per_flow"]], ddof=1)))
    print(f"flows: {tags}\n")
    print("=== Q1. WHAT IS THE RIDGE? (cells where FM model > 2x reference) ===")
    m, s = ms("model_mass"); print(f"  holds {100*m:.2f} +/- {100*s:.2f}% of MODEL mass")
    m, s = ms("reference_mass"); print(f"  holds {100*m:.2f} +/- {100*s:.2f}% of REFERENCE mass")
    print(f"\n  {'basin':10s} {'model on ridge':>15s} {'reference on ridge':>19s} {'ref overall':>12s}")
    for b in BASINS:
        mm = np.mean([r["basins_model_on_ridge"][b] for r in agg["per_flow"]])
        rr = np.mean([r["basins_reference_on_ridge"][b] for r in agg["per_flow"]])
        ov = float((reg_ref == b).mean())
        print(f"  {b:10s} {100*mm:14.1f}% {100*rr:18.1f}% {100*ov:11.1f}%")

    print("\n=== Q2. DID THE TRAINING SET PUT IT THERE? ===")
    m, s = ms("train_frac")
    e, es = ms("enrichment")
    print(f"  {100*m:.2f} +/- {100*s:.2f}% of the {agg['n_train']:,} training states lie ON the ridge")
    print(f"  the ridge holds only {100*ms('reference_mass')[0]:.2f}% of reference mass")
    print(f"  --> TRAINING-SET ENRICHMENT ON THE RIDGE = {e:.2f} +/- {es:.2f} x")

    agg["summary"] = {k: dict(zip(("mean", "sd"), ms(k)))
                      for k in ("model_mass", "reference_mass", "train_frac", "enrichment")}

    # ---------------- figure ----------------
    seq = LinearSegmentedColormap.from_list("seq", SEQ_STEPS)
    K = keep
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.5), constrained_layout=True)

    ax = axes[0]
    ax.pcolormesh(edges, edges, np.where(K["hot"], 1.0, np.nan).T, cmap=seq, vmin=0, vmax=1.4,
                  rasterized=True)
    sub = np.random.default_rng(0).choice(len(K["u_tr"]), min(6000, len(K["u_tr"])), replace=False)
    ax.plot(K["u_tr"][sub, 0], K["u_tr"][sub, 1], ".", ms=1.4, color="#eb6834", alpha=0.45,
            label=f"{len(R_tr):,} training states")
    for r in (1, 2, 3):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ec=MUTED, lw=0.8, ls=":"))
    ax.set_aspect("equal"); ax.set_xlim(-a.grid_lim, a.grid_lim); ax.set_ylim(-a.grid_lim, a.grid_lim)
    ax.set_xlabel("$u_1$"); ax.set_ylabel("$u_2$")
    ax.set_title("The ridge (blue) and where the\nmean-force labels actually sit")
    ax.legend(fontsize=8, loc="upper left", markerscale=6)

    ax = axes[1]
    for b in BASINS:
        m_ = (reg_ref == b)
        ax.plot(phi_ref[m_][::40], psi_ref[m_][::40], BASIN_MARKER[b], ms=1.2,
                color=BASIN_COLOR[b], alpha=0.30, label=f"ref {b}")
    m_ = K["on_mod"]
    ax.plot(phi_mod[m_][::12], psi_mod[m_][::12], "o", ms=1.8, color=INK, alpha=0.55,
            label="MODEL frames on ridge")
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180); ax.set_aspect("equal")
    ax.set_xlabel("$\\phi$ (deg)"); ax.set_ylabel("$\\psi$ (deg)")
    ax.set_title("Where the ridge lives in Ramachandran space")
    ax.legend(fontsize=7, loc="lower left", markerscale=5, ncol=2)

    ax = axes[2]
    lab = ["model\non ridge", "reference\non ridge", "training set\non ridge"]
    src = [K["on_mod"], K["on_ref"], K["on_tr"]]
    regs = [reg_mod, reg_ref, reg_tr]
    bot = np.zeros(3)
    for b in BASINS:
        v = np.array([100 * (rg[s_] == b).mean() for rg, s_ in zip(regs, src)])
        ax.bar(range(3), v, 0.6, bottom=bot, color=BASIN_COLOR[b], label=b,
               edgecolor=SURFACE, linewidth=1.5)
        bot += v
    ax.set_xticks(range(3)); ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylabel("composition (%)"); ax.set_ylim(0, 100)
    ax.set_title("Basin composition of the ridge"); ax.legend(fontsize=8)

    e_, es_ = ms("enrichment")
    # Caption is data-driven: which basins the ridge is made of varies by model, so state the
    # measured composition rather than asserting the beta/alphaR corridor (true for v2 FM only).
    top = max(BASINS, key=lambda b: np.mean([r["basins_model_on_ridge"][b]
                                             for r in agg["per_flow"]]))
    top_pct = 100 * np.mean([r["basins_model_on_ridge"][top] for r in agg["per_flow"]])
    fig.suptitle(f"The ridge holds {100*ms('model_mass')[0]:.2f}% of model mass "
                 f"({top} {top_pct:.0f}%-dominated) — training states are "
                 f"{e_:.1f}$\\times$ enriched on it.", fontsize=10.5)
    figure_path = a.outdir / "fig8_ridge_identity.png"
    fig.savefig(figure_path, bbox_inches="tight")
    summary_path = a.outdir / "ridge_identity.json"
    summary_path.write_text(json.dumps(agg, indent=2, default=float) + "\n")
    manifest = write_manifest(
        a.outdir,
        inputs={"reference": reference, "bias_npz": bias_npz,
                "flow_dir": flow_dir, "training_dataset": training_dataset,
                "ensemble": [str(item) for item in model_files]},
        parameters=vars(a),
        module="analysis.latent.ridge_identity",
        extra={"summary": str(summary_path), "figure": str(figure_path)},
    )
    print(f"\nwrote {figure_path}, {summary_path}, and {manifest}")


if __name__ == "__main__":
    main()
