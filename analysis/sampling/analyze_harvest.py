#!/usr/bin/env python3
"""Stage 2 (HARVEST CG) analysis: did the umbrella windows hold, and what did they cover?

    python -m sampling.analyze_harvest --campaign local_work/dhh_stage2_k450 \
        --discovery local_work/dhh_stage1_discover/flow \
        --bias-npz <smooth_reference_bias.npz> --outdir <dir> \
        --compare k=30:local_work/dhh_stage2_harvest k=150:local_work/dhh_stage2_k150

Stage 2 turns a handful of DISCOVERED states into many decorrelated CG configurations for
stage 3 to label. Two things decide whether it worked, and they pull against each other:

  - **Did the windows stay on their centres?** A window that slides into the nearest minimum
    is no longer sampling where it was told to. Several windows sliding into the SAME basin
    is worse than it looks: the slopes between basins go unsampled, and those are exactly
    where a conditional mean force is most informative.
  - **Did the harvest cover the discovered region?** Holding position is worthless if the
    windows collectively only see a fraction of what stage 1 found.

Measured on ala2 bb6 these do NOT trade off -- coverage rises monotonically with stiffness
(55% -> 62% -> 64% at kappa = 30 -> 150 -> 450) because the sliding was collapsing distinct
windows onto shared basins. Hence `--compare`: the stiffness is a measurement, not a default.

ANGLES ARE CIRCULAR. Every mean, spread and offset here goes through the unit circle. A
linear mean of samples straddling +-180 lands on the opposite side of the circle and reported
a 326 deg offset for a window that was 36 deg off.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze_discover import read_colvar


def cmean(x):
    return np.degrees(np.arctan2(np.sin(np.radians(x)).mean(),
                                 np.cos(np.radians(x)).mean()))


def csd(x):
    r = np.hypot(np.sin(np.radians(x)).mean(), np.cos(np.radians(x)).mean())
    return np.degrees(np.sqrt(-2.0 * np.log(max(float(r), 1e-12))))


def tica_of(R, b):
    d = np.linalg.norm(R[:, b["pair_indices"][:, 0], :] -
                       R[:, b["pair_indices"][:, 1], :], axis=-1)
    return (d - b["tica_mean"]) @ b["tica_coefficients"]


def coverage(z, edges, ref_cells, min_visits=5):
    H, _, _ = np.histogram2d(z[:, 0], z[:, 1], bins=edges)
    c = H >= min_visits
    return c, float((ref_cells & c).sum()) / max(int(ref_cells.sum()), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--discovery", type=Path, required=True)
    ap.add_argument("--bias-npz", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--compare", nargs="*", default=[],
                    help="LABEL:PATH of other stiffnesses to overlay")
    ap.add_argument("--centres-npz", type=Path, default=None,
                    help="fallback for campaigns built before `umbrella_centres_2d` was "
                         "recorded, which stored phi and psi centres as separate lists")
    ap.add_argument("--bins", type=int, default=60)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = np.load(a.bias_npz, allow_pickle=True)
    z_dis = tica_of(np.load(a.discovery / "cg_coords_all.npz")["R"].astype(np.float64), b)
    edges = [np.linspace(z_dis.min(0)[i], z_dis.max(0)[i], a.bins) for i in range(2)]
    Hd, _, _ = np.histogram2d(z_dis[:, 0], z_dis[:, 1], bins=edges)
    ref_cells = Hd >= 5

    R = np.load(a.campaign / "cg_coords_all.npz")["R"].astype(np.float64)
    z = tica_of(R, b)
    cells, cov = coverage(z, edges, ref_cells)

    man = json.loads((a.campaign / "campaign.json").read_text()) \
        if (a.campaign / "campaign.json").exists() else {}
    if a.centres_npz is not None:
        centres = np.load(a.centres_npz)["centres"].astype(float)
    else:
        centres = np.array(man.get("umbrella_centres_2d", []), dtype=float)

    # per-window realised centre and spread, from what the trajectories actually did
    rows = []
    for cd in sorted(p for p in a.campaign.glob("case_*") if p.is_dir()):
        f = cd / "colvar.dat"
        if not f.exists():
            continue
        # BY FIELD NAME, not column index: the harvest colvar is `time phi psi umb.bias`
        # while the discover colvar carries tic1/tic2/chirality too, so a fixed index reads
        # the wrong quantity depending on which stage wrote the file.
        c, d = read_colvar(f)
        if d.ndim != 2 or len(d) < 10:
            continue
        n = len(d) // 10                                   # drop 10% equilibration
        ph, ps = np.degrees(c["phi"][n:]), np.degrees(c["psi"][n:])
        rows.append((cmean(ph), cmean(ps), csd(ph), csd(ps)))
    W = np.array(rows) if rows else np.zeros((0, 4))

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    a0 = ax[0, 0]
    a0.pcolormesh(edges[0], edges[1], np.where(Hd > 0, Hd, np.nan).T, cmap="Greys",
                  shading="auto")
    a0.scatter(z[::53, 0], z[::53, 1], s=1.5, c="#2b6cb0", alpha=.25, edgecolor="none")
    a0.set(xlabel="tic1", ylabel="tic2",
           title=f"A  harvest (blue) over stage-1 discovery (grey)\n"
                 f"{len(W)} windows, {len(z):,} frames, covers {100*cov:.0f}% of "
                 f"{int(ref_cells.sum())} discovered cells")

    a1 = ax[0, 1]
    if len(W):
        a1.errorbar(W[:, 0], W[:, 1], xerr=W[:, 2], yerr=W[:, 3], fmt="o", ms=3,
                    lw=.6, alpha=.6, color="#2b6cb0", ecolor="#90cdf4")
    if len(centres):
        a1.scatter(centres[:, 0], centres[:, 1], s=16, marker="x", c="crimson",
                   label="commanded centre")
        a1.legend(frameon=False, fontsize=9)
    a1.set(xlim=(-180, 180), ylim=(-180, 180), xlabel="phi (deg)", ylabel="psi (deg)",
           xticks=[-180, -90, 0, 90, 180], yticks=[-180, -90, 0, 90, 180],
           title="B  realised window centres +- circular sd\n"
                 + (f"median sd {np.median(W[:, 2]):.1f} deg (phi)" if len(W) else ""))

    # C -- observed spread against what the RESTRAINT ALONE would give, sqrt(kT/kappa).
    # Which of the two dominates is a measurement, not an assumption: at kappa=30 the
    # observed 7-9 deg was far tighter than the restraint's own 16.5 deg, so the FES was
    # confining; by kappa=450 the two coincide and the restraint is.
    a2 = ax[1, 0]
    kap = man.get("kappa_phi_kJ_mol_rad2", man.get("kappa_kJ_mol_rad2"))
    sd_pred = np.degrees(np.sqrt(0.0083144621 * 298.0 / kap)) if kap else None
    if len(W):
        a2.hist(W[:, 2], bins=25, alpha=.75, color="#2b6cb0", label="phi")
        a2.hist(W[:, 3], bins=25, alpha=.55, color="#dd6b20", label="psi")
    if sd_pred:
        a2.axvline(sd_pred, color="k", ls="--", lw=1.2,
                   label=f"restraint alone: {sd_pred:.1f} deg")
    a2.legend(frameon=False, fontsize=9)
    ratio = (np.median(W[:, 2]) / sd_pred) if (len(W) and sd_pred) else float("nan")
    a2.set(xlabel="within-window circular sd (deg)", ylabel="windows",
           title=f"C  what confines each window\nobserved / restraint-only = {ratio:.2f}  "
                 + ("(restraint dominates)" if ratio > 0.8 else "(the FES dominates)"))

    # D -- the stiffness measurement. Soft windows slide into shared basins and LOSE coverage.
    a3 = ax[1, 1]
    labs, covs = [], []
    for spec in a.compare:
        lab, _, p = spec.partition(":")
        f = Path(p) / "cg_coords_all.npz"
        if not f.exists():
            continue
        _, cv = coverage(tica_of(np.load(f)["R"].astype(np.float64), b), edges, ref_cells)
        labs.append(lab); covs.append(100 * cv)
    labs.append(a.campaign.name); covs.append(100 * cov)
    a3.bar(range(len(covs)), covs, color=["#a0aec0"] * (len(covs) - 1) + ["#2b6cb0"])
    for i, v in enumerate(covs):
        a3.annotate(f"{v:.0f}%", (i, v), ha="center", va="bottom", fontsize=9)
    a3.set(xticks=range(len(labs)), ylabel="% of discovered cells covered",
           title="D  stiffness is a measurement\ncoverage rises with kappa -- soft windows "
                 "slide into shared basins")
    a3.set_xticklabels(labs, rotation=15, ha="right", fontsize=8)

    for x in ax.ravel():
        x.grid(alpha=.25, lw=.5)
    fig.tight_layout()
    fig.savefig(a.outdir / "harvest_coverage.png", dpi=140)
    plt.close(fig)
    print(f"{len(W)} windows, {len(z):,} frames, covers {100*cov:.0f}% of "
          f"{int(ref_cells.sum())} discovered cells")
    if len(W):
        print(f"  within-window circular sd: phi median {np.median(W[:, 2]):.1f} deg, "
              f"psi median {np.median(W[:, 3]):.1f} deg")
    print(f"wrote {a.outdir}/harvest_coverage.png")


if __name__ == "__main__":
    main()
