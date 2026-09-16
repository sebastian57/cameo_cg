#!/usr/bin/env python3
"""Where do the stencil LABELS sit, in Ramachandran and TICA space, vs the AA reference?

This maps the DATASET, not a model's MD. The question it answers: the v2 42k label set was
4.37x over-enriched on the beta/alphaR ridge and only 23.5% beta against a truth of 65.2%; did
the v3 anchor selection (50% p_ref importance draw + 50% pair-distance farthest point) fix it,
and where exactly do the remaining labels pile up?
"""
import argparse
from pathlib import Path

import numpy as np

from analysis.common.cli import add_project_root_argument
from analysis.common.paths import repo_root, resolve_input, resolve_output
from analysis.common.provenance import write_manifest

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_LO, DIV_MID, DIV_HI = "#2a78d6", "#f2f1ec", "#eb6834"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["acquisition", "subsets", "tight"], default="acquisition")
    ap.add_argument("--meanforce", type=Path, required=True)
    ap.add_argument("--stencil", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--bias-npz", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--v2-dataset", type=Path)
    ap.add_argument("--refmatched", type=Path)
    ap.add_argument("--randomctrl", type=Path)
    ap.add_argument("--tight-dataset", type=Path)
    add_project_root_argument(ap)
    a = ap.parse_args()
    project_root = repo_root(a.project_root)
    outdir = resolve_output(a.outdir, base=project_root)
    meanforce = resolve_input(a.meanforce, base=project_root, label="mean-force labels")
    stencil = resolve_input(a.stencil, base=project_root, label="stencil states")
    reference = resolve_input(a.reference, base=project_root, label="reference frames")
    bias_npz = resolve_input(a.bias_npz, base=project_root, label="bias artifact")
    from sampling.biases.tica_regional import SmoothTICABias
    from sampling.mapping import dihedral_deg, get_mapping, wrap_deg
    bias = SmoothTICABias.load(bias_npz); mp = get_mapping("ala2_backbone_cb_6")
    mm = np.load(meanforce, allow_pickle=False)
    st = np.load(stencil, allow_pickle=False)
    ref_R = np.asarray(np.load(reference, allow_pickle=False)["R"], np.float64)
    if a.mode == "acquisition" and a.v2_dataset is None:
        raise SystemExit("--v2-dataset is required for acquisition mode")
    if a.mode == "subsets" and (a.refmatched is None or a.randomctrl is None):
        raise SystemExit("--refmatched and --randomctrl are required for subsets mode")
    if a.mode == "tight" and a.tight_dataset is None:
        raise SystemExit("--tight-dataset is required for tight mode")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cv = lambda R, n: wrap_deg(dihedral_deg(R, mp.cvs[n].bead_indices) + mp.cvs[n].shift_deg)

    isanc = st["direction"][mm["state"]] == -1
    REF = ("AA reference", ref_R)
    mode = a.mode
    if mode == "subsets":
        # the two 139k arms of the transition-region control experiment, against the full set
        sets = [REF,
                ("v3 full (250k)", mm["R"].astype(np.float64)),
                ("v3 refmatched (139k)",
                 np.asarray(np.load(resolve_input(a.refmatched, base=project_root,
                                                   label="refmatched dataset"),
                                      allow_pickle=False)["R"], np.float64)),
                ("v3 random ctrl (139k)",
                 np.asarray(np.load(resolve_input(a.randomctrl, base=project_root,
                                                   label="random control dataset"),
                                      allow_pickle=False)["R"], np.float64))]
        tag = "subset"
    elif mode == "tight":
        # progression: unmatched -> density-matched -> hard-masked + gap-filled
        sets = [REF,
                ("v3 full (250k)", mm["R"].astype(np.float64)),
                ("v3 refmatched (139k)",
                 np.asarray(np.load(resolve_input(a.refmatched, base=project_root,
                                                   label="refmatched dataset"),
                                      allow_pickle=False)["R"], np.float64)),
                ("v3 tight+fill (117k)",
                 np.asarray(np.load(resolve_input(a.tight_dataset, base=project_root,
                                                   label="tight dataset"),
                                      allow_pickle=False)["R"], np.float64))]
        tag = "tight"
    else:
        sets = [REF,
                ("v2 labels (42k)",
                 np.asarray(np.load(resolve_input(a.v2_dataset, base=project_root,
                                                   label="v2 dataset"),
                                      allow_pickle=False)["R"], np.float64)),
                ("v3 anchors (10k)", mm["R"][isanc].astype(np.float64)),
                ("v3 all states (250k)", mm["R"].astype(np.float64))]
        tag = "dataset"
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID, "axes.grid": True,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "font.size": 9,
        "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 150})
    seq = LinearSegmentedColormap.from_list("s", SEQ)
    div = LinearSegmentedColormap.from_list("d", [DIV_LO, DIV_MID, DIV_HI])

    re_ = np.linspace(-180, 180, 73)
    z_all = np.concatenate([np.asarray(bias.projection.transform(R), np.float64)[:, :2] for _, R in sets])
    te = [np.linspace(np.percentile(z_all[:, i], 0.2), np.percentile(z_all[:, i], 99.8), 73) for i in (0, 1)]

    def dens(R, kind):
        if kind == "rama":
            H, _, _ = np.histogram2d(cv(R, "phi"), cv(R, "psi"), bins=[re_, re_])
        else:
            z = np.asarray(bias.projection.transform(R), np.float64)[:, :2]
            H, _, _ = np.histogram2d(z[:, 0], z[:, 1], bins=te)
        return H / H.sum()

    for kind, edges, labs in [("rama", (re_, re_), ("$\\phi$ (deg)", "$\\psi$ (deg)")),
                              ("tica", tuple(te), ("TIC 1", "TIC 2"))]:
        Ps = [dens(R, kind) for _, R in sets]
        fig, ax = plt.subplots(2, 4, figsize=(17.5, 8.0), constrained_layout=True)
        vmax = np.percentile(np.concatenate([p[p > 0] for p in Ps]), 99.5)
        for j, ((nm, _), P) in enumerate(zip(sets, Ps)):
            im = ax[0, j].pcolormesh(edges[0], edges[1], np.where(P > 0, P, np.nan).T,
                                     cmap=seq, vmin=0, vmax=vmax, rasterized=True)
            ax[0, j].set_title(nm); ax[0, j].set_xlabel(labs[0])
            if j == 0: ax[0, j].set_ylabel(labs[1])
        fig.colorbar(im, ax=ax[0, :], shrink=0.8, label="fraction of frames")
        ax[1, 0].axis("off")
        ax[1, 0].text(0.02, 0.5, "Bottom row:\n$\\log_2$(dataset / reference)\n\n"
                      "orange = OVER-represented\nblue = UNDER-represented\n"
                      "grey = reference has no frames\n(pure extrapolation)",
                      transform=ax[1, 0].transAxes, va="center", fontsize=9, color=INK2)
        for j in (1, 2, 3):
            m = (Ps[0] > 0) & (Ps[j] > 0)
            Rr = np.full_like(Ps[0], np.nan); Rr[m] = np.log2(Ps[j][m] / Ps[0][m])
            novel = (Ps[0] == 0) & (Ps[j] > 0)
            v = np.nanpercentile(np.abs(Rr), 99) or 1.0
            im2 = ax[1, j].pcolormesh(edges[0], edges[1], Rr.T, cmap=div,
                                      norm=TwoSlopeNorm(0.0, -v, v), rasterized=True)
            ax[1, j].pcolormesh(edges[0], edges[1], np.where(novel, 1.0, np.nan).T,
                                cmap=LinearSegmentedColormap.from_list("g", ["#9a9a94", "#9a9a94"]),
                                rasterized=True)
            ax[1, j].set_title(f"{sets[j][0]} vs reference   "
                               f"({100*Ps[j][novel].sum():.1f}% outside ref support)", fontsize=8.5)
            ax[1, j].set_xlabel(labs[0])
            fig.colorbar(im2, ax=ax[1, j], shrink=0.8, label="$\\log_2$ ratio")
        fig.suptitle(f"Stencil dataset in {'Ramachandran' if kind=='rama' else 'frozen TICA'} space "
                     f"— where the LABELS are, against the AA reference ensemble", fontsize=11)
        out = outdir / f"{kind}_{tag}_maps.png"
        fig.savefig(out, bbox_inches="tight"); plt.close(fig); print("wrote", out)

    manifest = write_manifest(
        outdir,
        inputs={"meanforce": meanforce, "stencil": stencil, "reference": reference,
                "bias_npz": bias_npz},
        parameters=vars(a), module="analysis.physics.dataset_maps",
        extra={"mode": mode})
    print(f"manifest: {manifest}")
    print("\n=== fraction of each label set OUTSIDE reference support ===")
    for kind in ("rama", "tica"):
        P0 = dens(sets[0][1], kind)
        for nm, R in sets[1:]:
            P = dens(R, kind)
            print(f"  {kind:5s} {nm:22s} {100*P[(P0==0)&(P>0)].sum():5.2f}%")


if __name__ == "__main__":
    main()
