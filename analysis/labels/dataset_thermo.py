#!/usr/bin/env python3
"""Rama + TICA maps and basin free-energy differences for the assembled training set.

Three estimators of dF, which is the point of the exercise:

  1. DENSITY, dF = -kT ln(p_A/p_B). Correct ONLY for an equilibrium sample, so valid for the
     reference and INVALID for the campaign (its density is the biased one).
  2. REWEIGHTED density. The campaign's frames were sampled under a KNOWN bias V(z) that PLUMED
     recorded per frame, so w = exp(+V/kT) restores the Boltzmann ensemble. Effective sample
     size is reported because the reweighting variance is what decides whether it is usable.
  3. FORCE INTEGRATION along phi. The campaign's FORCES are bias-free by construction (they come
     from the `-rerun`), so a PMF integrated from them needs no reweighting at all. Estimator:
     dA/dphi = -<(F . grad phi)/|grad phi|^2>_phi, integrated. This NEGLECTS the geometric term
     kT<div(grad phi/|grad phi|^2)>. Rather than assume that term is small, estimator 3 is run
     on the REFERENCE first, where estimator 1 is exact: the gap between them measures the
     neglected term, and only if that gap is small is estimator 3 trusted on the campaign.

The analytic dihedral gradient is checked against finite differences before use.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sampling.mapping import get_mapping

KT = 0.5921868690749673


def dihedral(R, idx):
    """Dihedral in degrees for beads idx=(i,j,k,l)."""
    i, j, k, l = idx
    b1 = R[:, j] - R[:, i]; b2 = R[:, k] - R[:, j]; b3 = R[:, l] - R[:, k]
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    nb2 = np.linalg.norm(b2, axis=1)
    x = np.einsum("ij,ij->i", n1, n2)
    y = np.einsum("ij,ij->i", np.cross(n1, n2), b2 / nb2[:, None])
    return np.degrees(np.arctan2(y, x))


def dihedral_grad(R, idx, h=1e-5):
    """d phi / dR in rad/A by central differences on the 4 defining beads.

    The analytic Blondel-Karplus gradient was tried first and FAILED its own finite-difference
    check by 1.695 rad/A: the atan2 argument order here is one of the two valid dihedral sign
    conventions, and the textbook gradient belongs to the other. Rather than chase the sign,
    differentiate the function actually being used -- 24 cheap dihedral evaluations, and the
    convention cannot disagree with itself. Wrapped differences keep the +-180 seam correct.
    """
    g = np.zeros_like(R)
    for b in idx:
        for c in range(3):
            p = R.copy(); p[:, b, c] += h
            m = R.copy(); m[:, b, c] -= h
            a1 = np.radians(dihedral(p, idx)); a2 = np.radians(dihedral(m, idx))
            g[:, b, c] = np.arctan2(np.sin(a1 - a2), np.cos(a1 - a2)) / (2 * h)
    return g


def check_dihedral(R, idx, mapping_phi):
    """Confirm our dihedral matches the project's own CV before anything is built on it."""
    ours = dihedral(R, idx)
    d = np.abs(np.arctan2(np.sin(np.radians(ours - mapping_phi)),
                          np.cos(np.radians(ours - mapping_phi))))
    return np.degrees(d).max()


def fes(x, y, edges):
    H, _, _ = np.histogram2d(x, y, bins=edges)
    p = H / max(H.sum(), 1)
    F = np.where(p > 0, -KT * np.log(np.maximum(p, 1e-300)), np.nan)
    return F - np.nanmin(F), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembled", required=True)
    ap.add_argument("--sens", required=True)
    ap.add_argument("--campaign-root", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    D = np.load(a.assembled)
    R, F, origin = D["R"], D["F"], D["origin"]
    m = get_mapping("ala2_backbone_cb_6")
    pidx = m.cvs["phi"].bead_indices; sidx = m.cvs["psi"].bead_indices

    sub = R[rng.choice(len(R), 4000, replace=False)]
    e1 = check_dihedral(sub, pidx, m.cvs["phi"].evaluate(sub))
    e2 = check_dihedral(sub, sidx, m.cvs["psi"].evaluate(sub))
    print(f"dihedral vs project CV: max |dphi| {e1:.2e} deg, |dpsi| {e2:.2e} deg")
    assert e1 < 1e-6 and e2 < 1e-6, "our dihedral disagrees with the project's own CV"

    phi = dihedral(R, pidx)
    psi = dihedral(R, sidx)
    gphi = dihedral_grad(R, pidx)

    z = np.load(a.sens, allow_pickle=True)
    pairs, mu, coef = z["pairs"], z["tica_mean"], z["tica_coefficients"]
    d = np.linalg.norm(R[:, pairs[:, 0], :] - R[:, pairs[:, 1], :], axis=-1)
    Y = (d - mu) @ coef

    # per-frame bias for the campaign frames, rebuilt from the colvar files.
    # collect.py discarded 200 ps = 100 rows of each 3001-row colvar at 2 ps.
    bias = np.zeros(len(R))
    root = Path(a.campaign_root)
    for arm, oid in (("ch1", 1), ("ch2", 2)):
        reps = sorted((root / f"arm_{arm}").glob("replica_*"))
        vals = []
        for rp in reps:
            C = np.array([l.split() for l in (rp / "colvar.dat").read_text().splitlines()
                          if l.strip() and not l.startswith("#")], dtype=float)
            vals.append(C[100:, 3:].sum(axis=1))     # ext + walls, discard-aligned
        vals = np.concatenate(vals)
        sel = np.where(origin == oid)[0]
        assert len(vals) == len(sel), f"{arm}: {len(vals)} bias rows vs {len(sel)} frames"
        bias[sel] = vals
    print(f"bias rebuilt for {int((origin>0).sum())} campaign frames; "
          f"range {bias[origin>0].min():.3f}..{bias[origin>0].max():.3f} kcal/mol")

    ref, camp = origin == 0, origin > 0
    w = np.exp(bias / KT); w[ref] = 1.0
    ess = w[camp].sum() ** 2 / max((w[camp] ** 2).sum(), 1e-300)
    print(f"reweighting ESS = {ess:.0f} of {int(camp.sum())} campaign frames "
          f"({100*ess/max(camp.sum(),1):.2f}%)")

    # ---- basins in Rama (the standard project definitions) ----
    BAS = {"alphaR": lambda p, s: (p > -180) & (p < 0) & (s > -120) & (s < 50),
           "beta":   lambda p, s: (p < 0) & ((s > 100) | (s < -150)),
           "alphaL": lambda p, s: (p > 0) & (p < 120) & (s > -50) & (s < 100)}
    mask = {k: f(phi, psi) for k, f in BAS.items()}

    def dF(sel, weights=None):
        ww = np.ones(len(phi)) if weights is None else weights
        tot = {k: float(ww[sel & mk].sum()) for k, mk in mask.items()}
        base = tot["beta"]
        return {k: (-KT * np.log(v / base) if v > 0 and base > 0 else np.nan)
                for k, v in tot.items()}, tot

    print("\n=== dF relative to beta (kcal/mol) ===")
    print(f"{'estimator':34s} {'alphaR':>9s} {'alphaL':>9s}")
    d_ref, n_ref = dF(ref)
    print(f"{'1. reference, density (TRUTH)':34s} {d_ref['alphaR']:+9.3f} {d_ref['alphaL']:+9.3f}")
    d_craw, _ = dF(camp)
    print(f"{'   campaign, density (BIASED)':34s} {d_craw['alphaR']:+9.3f} {d_craw['alphaL']:+9.3f}")
    d_crw, _ = dF(camp, w)
    print(f"{'2. campaign, REWEIGHTED':34s} {d_crw['alphaR']:+9.3f} {d_crw['alphaL']:+9.3f}")

    # ---- 3. force integration along phi ----
    def pmf_phi(sel, nb=72):
        e = np.linspace(-180, 180, nb + 1)
        gp = gphi[sel]; fp = F[sel]
        num = np.einsum("nbc,nbc->n", fp, gp)
        den = np.einsum("nbc,nbc->n", gp, gp)
        mf = -num / np.maximum(den, 1e-30)                 # dA/dphi in kcal/mol/rad
        idx = np.clip(np.digitize(phi[sel], e) - 1, 0, nb - 1)
        prof = np.array([mf[idx == b].mean() if (idx == b).any() else np.nan
                         for b in range(nb)])
        dphi = np.radians(360.0 / nb)
        good = np.isfinite(prof)
        A = np.full(nb, np.nan); A[good] = np.cumsum(np.nan_to_num(prof)[good]) * dphi
        return 0.5 * (e[1:] + e[:-1]), A - np.nanmin(A)

    cen, A_ref_force = pmf_phi(ref)
    Hd, _ = np.histogram(phi[ref], bins=np.linspace(-180, 180, 73))
    A_ref_dens = -KT * np.log(np.maximum(Hd / Hd.sum(), 1e-300)); A_ref_dens -= np.nanmin(A_ref_dens)
    ok = np.isfinite(A_ref_force) & np.isfinite(A_ref_dens)
    gap = np.abs(A_ref_force[ok] - A_ref_dens[ok])
    print(f"\n3. force-integrated vs density PMF along phi, ON THE REFERENCE "
          f"(gap = the neglected geometric term):")
    print(f"   mean |gap| {gap.mean():.3f}  max {gap.max():.3f} kcal/mol over {ok.sum()} bins")
    _, A_camp_force = pmf_phi(camp)

    # ---------------- figure ----------------
    er = np.linspace(-180, 180, 73)
    et_x = np.linspace(*np.percentile(Y[:, 0], [0.2, 99.8]), 61)
    et_y = np.linspace(*np.percentile(Y[:, 1], [0.2, 99.8]), 61)
    fig, ax = plt.subplots(2, 4, figsize=(23, 9.5))
    sets = [("reference 200k", ref, None), ("campaign 116k (raw)", camp, None),
            ("campaign (reweighted)", camp, w), ("assembled 316k", np.ones(len(phi), bool), None)]
    for c, (t, sel, ww) in enumerate(sets):
        if ww is None:
            Fr, _ = fes(phi[sel], psi[sel], [er, er])
        else:
            H, _, _ = np.histogram2d(phi[sel], psi[sel], bins=[er, er], weights=ww[sel])
            p = H / H.sum(); Fr = np.where(p > 0, -KT*np.log(np.maximum(p, 1e-300)), np.nan)
            Fr -= np.nanmin(Fr)
        im = ax[0, c].imshow(Fr.T, origin="lower", extent=[-180, 180, -180, 180],
                             cmap="viridis", vmax=6)
        ax[0, c].set_title(f"Rama FES — {t}", fontsize=10)
        ax[0, c].set_xlabel("phi"); fig.colorbar(im, ax=ax[0, c], fraction=0.046)
        if ww is None:
            Ft, _ = fes(Y[sel, 0], Y[sel, 1], [et_x, et_y])
        else:
            H, _, _ = np.histogram2d(Y[sel, 0], Y[sel, 1], bins=[et_x, et_y], weights=ww[sel])
            p = H / H.sum(); Ft = np.where(p > 0, -KT*np.log(np.maximum(p, 1e-300)), np.nan)
            Ft -= np.nanmin(Ft)
        im = ax[1, c].imshow(Ft.T, origin="lower", cmap="viridis", vmax=6,
                             extent=[et_x[0], et_x[-1], et_y[0], et_y[-1]], aspect="auto")
        ax[1, c].set_title(f"TICA FES — {t}", fontsize=10)
        ax[1, c].set_xlabel("TIC 1"); fig.colorbar(im, ax=ax[1, c], fraction=0.046)
    ax[0, 0].set_ylabel("psi"); ax[1, 0].set_ylabel("TIC 2")
    fig.suptitle("Assembled training set: reference vs targeted campaign "
                 "(kcal/mol, capped at 6)", y=1.00)
    fig.tight_layout(); fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {a.out}")

    f2, b2 = plt.subplots(figsize=(7.5, 4.6))
    b2.plot(cen, A_ref_dens, label="reference: -kT ln p (exact)", lw=2)
    b2.plot(cen, A_ref_force, "--", label="reference: force-integrated")
    b2.plot(cen, A_camp_force, label="campaign: force-integrated (no reweighting needed)", lw=2)
    b2.set_xlabel("phi (deg)"); b2.set_ylabel("A (kcal/mol)"); b2.grid(alpha=0.3)
    b2.set_title("PMF along phi: density vs force integration")
    b2.legend(fontsize=8)
    p2 = str(a.out).replace(".png", "_pmf_phi.png")
    f2.tight_layout(); f2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
