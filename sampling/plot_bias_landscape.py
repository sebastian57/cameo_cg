#!/usr/bin/env python3
"""Show what a TICA bias artifact actually targets, before spending GPU time on it.

    python -m sampling.plot_bias_landscape \
        --bias "lambda0.25 blended=.../smooth_reference_bias_lambda0p25.npz" \
        --bias "lambda0.5 transition85=.../smooth_transition_bias_l0p5_w0p85.npz" \
        --reference <mapped-AA reference>.npz --outdir <dir> --prefix <name>

Three views per artifact:

  1. bias potential V(z) over the TICA grid -- the raw landscape
  2. effective FES, F_ref(z) + V(z) -- what the biased simulation actually feels; flat
     regions here are regions whose population the bias has decided to equalise
  3. mean bias painted on the RAMACHANDRAN plane -- the same information in coordinates
     a human can reason about, which is the view that tells you whether you are
     targeting barriers or quietly flattening basin ratios

View 3 is the point. Low reference density IS the free energy, so a bias that lights up
basin interiors is overwriting the physics; one that lights up the corridors between
them is adding transition information the reference lacks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sampling.biases.tica_regional import SmoothTICABias  # noqa: E402
from sampling.mapping import get_mapping                  # noqa: E402

KB_KCAL = 0.0019872042586


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bias", action="append", required=True, metavar="LABEL=NPZ",
                    help="bias artifact, repeatable")
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--mapping", type=str, default="ala2_backbone_cb_6")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--prefix", type=str, default="bias")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=298.0)
    args = ap.parse_args()

    m = get_mapping(args.mapping)
    kT = KB_KCAL * args.temperature
    args.outdir.mkdir(parents=True, exist_ok=True)

    R = np.load(args.reference)["R"][:: args.stride].astype(np.float64)
    phi, psi = m.cvs["phi"].evaluate(R), m.cvs["psi"].evaluate(R)
    print(f"reference: {len(R)} frames (stride {args.stride})")

    biases = []
    for spec in args.bias:
        if "=" not in spec:
            raise SystemExit(f"--bias expects LABEL=NPZ, got {spec!r}")
        label, path = spec.split("=", 1)
        b = SmoothTICABias.load(path)
        z = b.projection.transform(R)
        V = np.array([b.tica_energy_gradient(zi)[0] for zi in z[:, :2]])
        biases.append({"label": label, "bias": b, "z": z, "V": V})
        print(f"  {label}: V over reference frames  mean {V.mean():.3f}  "
              f"min {V.min():.3f}  max {V.max():.3f} kcal/mol")

    z0 = biases[0]["z"]
    xe = np.linspace(z0[:, 0].min(), z0[:, 0].max(), args.bins + 1)
    ye = np.linspace(z0[:, 1].min(), z0[:, 1].max(), args.bins + 1)
    xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    Href, _, _ = np.histogram2d(z0[:, 0], z0[:, 1], bins=[xe, ye])
    p = Href / Href.sum()
    with np.errstate(divide="ignore"):
        Fref = -kT * np.log(p)
    Fref[~np.isfinite(Fref)] = np.nan
    Fref -= np.nanmin(Fref)

    pe = np.linspace(-180, 180, args.bins + 1)

    n = len(biases)
    fig, axes = plt.subplots(3, n + 1, figsize=(5.2 * (n + 1), 12.6),
                             constrained_layout=True, squeeze=False)

    # column 0: the reference itself
    ex_t = [xe[0], xe[-1], ye[0], ye[-1]]
    im = axes[0][0].imshow(Fref.T, origin="lower", extent=ex_t, aspect="auto",
                           cmap="turbo", vmin=0, vmax=6)
    axes[0][0].set_title("reference FES (TICA)"); fig.colorbar(im, ax=axes[0][0])
    axes[1][0].axis("off")
    axes[1][0].text(.5, .5, "row 1: bias V(z)\nrow 2: effective FES = F_ref + V\n"
                            "row 3: mean bias on Ramachandran",
                    ha="center", va="center", fontsize=11)
    Hr, _, _ = np.histogram2d(phi, psi, bins=[pe, pe])
    pr = Hr / Hr.sum()
    with np.errstate(divide="ignore"):
        Fr = -kT * np.log(pr)
    Fr[~np.isfinite(Fr)] = np.nan
    Fr -= np.nanmin(Fr)
    im = axes[2][0].imshow(Fr.T, origin="lower", extent=[-180, 180, -180, 180],
                           aspect="auto", cmap="turbo", vmin=0, vmax=6)
    axes[2][0].set_title("reference FES (Ramachandran)")
    axes[2][0].set_xlabel("phi"); axes[2][0].set_ylabel("psi")
    fig.colorbar(im, ax=axes[2][0])

    for k, e in enumerate(biases, start=1):
        grid = np.stack(np.meshgrid(xc, yc, indexing="ij"), axis=-1).reshape(-1, 2)
        Vg = np.array([e["bias"].tica_energy_gradient(zz)[0] for zz in grid]
                      ).reshape(len(xc), len(yc))
        im = axes[0][k].imshow(Vg.T, origin="lower", extent=ex_t, aspect="auto",
                               cmap="magma")
        axes[0][k].set_title(f"{e['label']}\nbias V(z)"); fig.colorbar(im, ax=axes[0][k])

        eff = Fref + Vg
        eff = eff - np.nanmin(eff)
        im = axes[1][k].imshow(eff.T, origin="lower", extent=ex_t, aspect="auto",
                               cmap="turbo", vmin=0, vmax=6)
        axes[1][k].set_title("effective FES = F_ref + V"); fig.colorbar(im, ax=axes[1][k])

        # mean bias per Ramachandran cell -- which conformations are targeted
        num, _, _ = np.histogram2d(phi, psi, bins=[pe, pe], weights=e["V"])
        cnt, _, _ = np.histogram2d(phi, psi, bins=[pe, pe])
        mean_V = np.divide(num, cnt, out=np.full_like(num, np.nan), where=cnt > 0)
        im = axes[2][k].imshow(mean_V.T, origin="lower", extent=[-180, 180, -180, 180],
                               aspect="auto", cmap="magma")
        axes[2][k].set_title("mean bias on Ramachandran")
        axes[2][k].set_xlabel("phi"); axes[2][k].set_ylabel("psi")
        fig.colorbar(im, ax=axes[2][k], label="V (kcal/mol)")

    fig.suptitle("What the TICA bias targets (evaluated on the AA reference)", fontsize=13)
    out = args.outdir / f"{args.prefix}_bias_landscape.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
