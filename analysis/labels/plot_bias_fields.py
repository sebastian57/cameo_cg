#!/usr/bin/env python3
"""Visualise where each emitted bias PULLS, in the TICA plane it acts in.

Per channel, four panels:
  A  reference FES                          -- the landscape as it is
  B  bias V with -grad V arrows             -- the force the bias applies
  C  predicted sampled density p*exp(-V/kT) -- where you end up
  D  log2(predicted / reference)            -- the ENRICHMENT: what you gain and lose

D is the panel that answers "where does it pull": red = oversampled relative to plain MD,
blue = given up.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KT = 0.5921868690749673


def panels(ax, cx, cy, rho, V, title, q=None):
    ext = [cx[0], cx[-1], cy[0], cy[-1]]
    occ = rho > 0
    F = np.where(occ, -KT * np.log(np.maximum(rho, 1e-300)), np.nan)
    F -= np.nanmin(F)
    im = ax[0].imshow(F.T, origin="lower", extent=ext, aspect="auto", cmap="viridis",
                      vmax=np.nanpercentile(F, 97))
    ax[0].set_title("A. reference FES (kcal/mol)", fontsize=9)
    plt.colorbar(im, ax=ax[0], fraction=0.046)

    im = ax[1].imshow(np.where(occ, V, np.nan).T, origin="lower", extent=ext, aspect="auto",
                      cmap="coolwarm")
    gx, gy = np.gradient(V, cx[1]-cx[0], cy[1]-cy[0])
    st = max(1, len(cx) // 22)
    X, Y = np.meshgrid(cx[::st], cy[::st], indexing="ij")
    m = occ[::st, ::st]
    # -grad V is the force the bias exerts: arrows point where it PUSHES the system
    ax[1].quiver(X[m], Y[m], -gx[::st, ::st][m], -gy[::st, ::st][m],
                 color="k", alpha=0.65, width=0.004, scale_units="xy")
    ax[1].set_title(f"B. bias V (colour) and -grad V (arrows)\n{title}", fontsize=9)
    plt.colorbar(im, ax=ax[1], fraction=0.046)

    pv = np.where(occ, rho * np.exp(-V / KT), 0.0)
    pv /= max(pv.sum(), 1e-30)
    im = ax[2].imshow(np.log10(np.where(pv > 0, pv, np.nan)).T, origin="lower", extent=ext,
                      aspect="auto", cmap="magma")
    ax[2].set_title("C. predicted sampled density\nlog10 p*exp(-V/kT)", fontsize=9)
    plt.colorbar(im, ax=ax[2], fraction=0.046)

    r = np.where(occ & (rho > 0), np.log2(np.maximum(pv, 1e-300) / np.maximum(rho, 1e-300)),
                 np.nan)
    v = np.nanpercentile(np.abs(r), 99)
    im = ax[3].imshow(r.T, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r",
                      vmin=-v, vmax=v)
    ax[3].set_title("D. ENRICHMENT log2(biased / reference)\nred = gained, blue = given up",
                    fontsize=9)
    plt.colorbar(im, ax=ax[3], fraction=0.046)
    if q is not None:
        for k in (0, 3):
            ax[k].contour(cx, cy, q.T, levels=[0.5], colors="lime", linewidths=1.6)
    for k in range(4):
        ax[k].set_xlabel("TIC 1", fontsize=8)
    ax[0].set_ylabel("TIC 2", fontsize=8)
    return pv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch1", required=True); ap.add_argument("--ch2", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    fig, ax = plt.subplots(2, 4, figsize=(21, 9))
    for row, (f, name) in enumerate([(a.ch1, "channel 1:  V1 = kT ln P_i  (constant per basin)"),
                                     (a.ch2, "channel 2:  V2 = -2kT ln(|grad q|+eps)")]):
        B = np.load(f, allow_pickle=True)
        cx, cy, V, rho = B["cx_data"], B["cy_data"], B["V_data"], B["rho"]
        pv = panels(ax[row], cx, cy, rho, V, name, q=B["q"])
        occ = rho > 0
        enr = np.where(occ, pv / np.maximum(rho, 1e-300), np.nan)
        print(f"\n{name}")
        print(f"  max enrichment {np.nanmax(enr):8.2f}x   max depletion {np.nanmin(enr[enr>0]):8.4f}x")
        lab = B["labels"]
        for i in [int(v) for v in np.unique(lab) if v >= 0]:
            m = (lab == i) & occ
            if rho[m].sum() < 0.01:
                continue
            print(f"  basin {i}: reference {100*rho[m].sum():5.2f}%  ->  biased "
                  f"{100*pv[m].sum():5.2f}%   ({pv[m].sum()/rho[m].sum():.2f}x)")
    fig.suptitle("Where each targeted bias pulls, in the TICA plane it acts in "
                 "(green contour = committor q = 0.5, the dividing surface)", y=1.00)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
