"""Map local PMF curvature magnitude into TICA and Ramachandran space.

For each anchor, the stencil gives central differences along up to 6 internal directions:

    ||H(R) v||  ~=  ||F(R + d) - F(R - d)|| / ||(R + d) - (R - d)||

The denominator uses the REALIZED displacement read back from the trajectory, never the
nominal eps: .gro files store 0.001 nm, so realized bead positions differ from the nominal
target by up to 0.0150 A (19% of the inner step). See
KNOWLEDGE_BASE/P_cameo_cg/DESIGN/FINITE_DIFFERENCE_STENCIL_HARVEST.md.

Reports per anchor:
    C_mean  mean over available directions      C_max   max over directions
    C_aniso C_max / C_mean  (1 = isotropic)

and bins them into (tic1, tic2) and (phi, psi), so the question "are the over-sharpening
regions the high-curvature regions?" becomes a picture.

No model, no GPU, no training. Runs on a login node in ~1 minute.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis.common.cli import add_project_root_argument
from analysis.common.paths import repo_root, resolve_input, resolve_output
from analysis.common.provenance import write_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--layer", choices=["inner", "outer"], default="inner",
                    help="which +/- pair to use for the central difference")
    ap.add_argument("--meanforce", type=Path, required=True)
    ap.add_argument("--stencil", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--bias-npz", type=Path, required=True)
    add_project_root_argument(ap)
    a = ap.parse_args()
    project_root = repo_root(a.project_root)
    a.outdir = resolve_output(a.outdir, base=project_root)
    meanforce = resolve_input(a.meanforce, base=project_root, label="mean-force labels")
    stencil = resolve_input(a.stencil, base=project_root, label="stencil states")
    reference = resolve_input(a.reference, base=project_root, label="reference frames")
    bias_npz = resolve_input(a.bias_npz, base=project_root, label="bias artifact")
    from sampling.biases.tica_regional import SmoothTICABias
    from sampling.mapping import dihedral_deg, get_mapping, wrap_deg

    mp = get_mapping("ala2_backbone_cb_6")
    bias = SmoothTICABias.load(bias_npz)

    mm = np.load(meanforce)
    st = np.load(stencil)
    state = mm["state"]
    slot = state // 25
    direction = st["direction"][state]
    mult = np.round(st["multiplier"][state], 2)
    R, F = np.asarray(mm["R"], np.float64), np.asarray(mm["F"], np.float64)
    n_slots = int(slot.max()) + 1
    lvl = 1.0 if a.layer == "inner" else 2.0

    # index (slot, direction, sign) -> row
    idx = {}
    for i in range(len(state)):
        d = int(direction[i])
        if d < 0:
            idx[(int(slot[i]), -1, 0)] = i
        elif abs(mult[i]) == lvl:
            idx[(int(slot[i]), d, int(np.sign(mult[i])))] = i

    C = np.full((n_slots, 6), np.nan)
    for s in range(n_slots):
        for d in range(6):
            ip, im = idx.get((s, d, 1)), idx.get((s, d, -1))
            if ip is None or im is None:
                continue
            dr = np.linalg.norm(R[ip] - R[im])          # realized, not nominal
            if dr > 0:
                C[s, d] = np.linalg.norm(F[ip] - F[im]) / dr

    anchor_row = np.array([idx[(s, -1, 0)] for s in range(n_slots)])
    Ranc = R[anchor_row]
    C_mean = np.nanmean(C, 1)
    C_max = np.nanmax(C, 1)
    C_aniso = C_max / np.maximum(C_mean, 1e-12)

    z = np.asarray(bias.projection.transform(Ranc), np.float64)
    cvv = lambda X, n: wrap_deg(dihedral_deg(X, mp.cvs[n].bead_indices) + mp.cvs[n].shift_deg)
    phi, psi = cvv(Ranc, "phi"), cvv(Ranc, "psi")

    Rref = np.asarray(np.load(reference, allow_pickle=False)["R"], np.float64)
    zr = np.asarray(bias.projection.transform(Rref), np.float64)
    phir, psir = cvv(Rref, "phi"), cvv(Rref, "psi")

    print(f"layer={a.layer}   anchors={n_slots:,}   directions/anchor="
          f"{np.isfinite(C).sum(1).mean():.2f}")
    for nm, v in (("C_mean", C_mean), ("C_max", C_max), ("C_aniso", C_aniso)):
        q = np.nanpercentile(v, [5, 25, 50, 75, 95])
        print(f"  {nm:8s} median {np.nanmedian(v):9.1f}   "
              f"p5/p25/p75/p95 {q[0]:8.1f} {q[1]:8.1f} {q[3]:8.1f} {q[4]:8.1f}")

    # TICA-direction curvature vs the random complement (2026-08-18 finding: TICA dirs stiffer)
    print(f"\n  ||Hv|| by direction:  " + "  ".join(
        f"d{d}={np.nanmedian(C[:, d]):.0f}" for d in range(6)) +
        "   (d0,d1 = TICA gradients; d2-5 = random complement)")

    np.savez_compressed(a.outdir / f"curvature_{a.layer}.npz", C=C, C_mean=C_mean,
                        C_max=C_max, C_aniso=C_aniso, z=z, phi=phi, psi=psi,
                        anchor_row=anchor_row)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def binned(x, y, v, xe, ye):
        num, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=v)
        cnt, _, _ = np.histogram2d(x, y, bins=[xe, ye])
        out = np.where(cnt >= 3, num / np.maximum(cnt, 1), np.nan)
        return out, cnt

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    te = [np.linspace(-1.2, 4.9, 45), np.linspace(-0.55, 0.5, 45)]
    re = [np.linspace(-180, 180, 49), np.linspace(-180, 180, 49)]

    for row, (X, Y, XR, YR, ed, xl, yl) in enumerate([
            (z[:, 0], z[:, 1], zr[:, 0], zr[:, 1], te, "TIC 1", "TIC 2"),
            (phi, psi, phir, psir, re, "phi (deg)", "psi (deg)")]):
        href, _, _ = np.histogram2d(XR, YR, bins=ed)
        pref = href / href.sum()
        for col, (v, nm) in enumerate([(C_mean, "median $\\|Hv\\|$ (mean over dirs)"),
                                       (C_aniso, "anisotropy $C_{max}/C_{mean}$")]):
            g, _ = binned(X, Y, v, *ed)
            ax = axes[row, col]
            im = ax.pcolormesh(ed[0], ed[1], g.T, cmap="magma",
                               vmin=np.nanpercentile(g, 5), vmax=np.nanpercentile(g, 95))
            plt.colorbar(im, ax=ax); ax.set_title(nm, fontsize=10)
            ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax = axes[row, 2]
        im = ax.pcolormesh(ed[0], ed[1], np.where(pref > 0, np.log10(pref), np.nan).T,
                           cmap="viridis")
        plt.colorbar(im, ax=ax); ax.set_title("reference $\\log_{10}p$", fontsize=10)
        ax.set_xlabel(xl); ax.set_ylabel(yl)

    fig.suptitle(f"Local PMF curvature from the {a.layer} stencil layer, "
                 f"{n_slots:,} anchors — where is the force field stiff, and does it "
                 f"coincide with where the reference has mass?", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.outdir / f"curvature_map_{a.layer}.png", dpi=140, bbox_inches="tight")
    manifest = write_manifest(
        a.outdir,
        inputs={"meanforce": meanforce, "stencil": stencil,
                "reference": reference, "bias_npz": bias_npz},
        parameters=vars(a), module="analysis.physics.curvature_map",
        extra={"figure": str(a.outdir / f"curvature_map_{a.layer}.png"),
               "summary": str(a.outdir / f"curvature_{a.layer}.npz")})
    print(f"\nwrote {a.outdir}/curvature_map_{a.layer}.png and curvature_{a.layer}.npz")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
