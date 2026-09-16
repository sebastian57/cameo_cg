#!/usr/bin/env python3
"""Where do the mean forces POINT, and can we integrate them into a free energy?

PART 1 -- direction field. Project the Cartesian bead mean force onto the (phi, psi) Jacobian
with the proper 2x2 metric,  F_xi = (J J^T)^-1 J F , giving the generalised force conjugate to
each dihedral. Plot it as a quiver over the Ramachandran plane and ask, per basin, whether it
points toward the basin minimum.

PART 2 -- integrate it. <F> = -grad A, so the field is (minus) a gradient and can be integrated.
phi/psi are periodic, so solve the Poisson equation grad^2 A = div(-F) by FFT: exact, no path
choice, no reference point. Compare the result to the reference FES -kT log P(phi,psi).

  THE CAVEAT THAT DECIDES HOW TO READ PART 2.  The measured force is the gradient of A(R) in
  the 18-dim bead space. The 2D PMF A(phi,psi) additionally integrates out the other 10 internal
  DOF, so the 2D mean force is the BOLTZMANN average of the projected force over those DOF
  within each (phi,psi) bin. Our states are farthest-point / p_ref selected, NOT Boltzmann
  distributed within a bin, so the bin average is biased. The reference-matched subset
  (tight117k) is much closer to Boltzmann and is therefore the honest one to integrate; the
  full 250k set is shown alongside to expose the size of the bias.
  Also omitted: the Fixman/Jacobian correction for the constrained ensemble -- another reason
  to read Part 2 as "does the label field carry the right FES shape", not as a PMF measurement.
"""
import argparse
from pathlib import Path

import numpy as np

from analysis.common.cli import add_project_root_argument
from analysis.common.paths import repo_root, resolve_input, resolve_output
from analysis.common.provenance import write_manifest

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = ["#2a78d6", "#f2f1ec", "#eb6834"]
KT = 0.5921868690749673


def dihed_grad(R, idx, h=1e-4):
    """d(dihedral)/dR by central differences, in deg/Angstrom. R is (N,6,3)."""
    from sampling.mapping import dihedral_deg, wrap_deg
    N = len(R); G = np.zeros((N, 6, 3))
    for b in range(6):
        for c in range(3):
            Rp = R.copy(); Rp[:, b, c] += h
            Rm = R.copy(); Rm[:, b, c] -= h
            G[:, b, c] = wrap_deg(dihedral_deg(Rp, idx) - dihedral_deg(Rm, idx)) / (2 * h)
    return G


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meanforce", type=Path, required=True)
    ap.add_argument("--tight-dataset", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    add_project_root_argument(ap)
    a = ap.parse_args()
    project_root = repo_root(a.project_root)
    meanforce = resolve_input(a.meanforce, base=project_root, label="mean-force labels")
    tight_dataset = resolve_input(a.tight_dataset, base=project_root, label="tight dataset")
    reference = resolve_input(a.reference, base=project_root, label="reference frames")
    a.outdir = resolve_output(a.outdir, base=project_root)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from sampling.mapping import dihedral_deg, get_mapping, wrap_deg

    mp = get_mapping("ala2_backbone_cb_6")
    seq = LinearSegmentedColormap.from_list("s", SEQ)
    div = LinearSegmentedColormap.from_list("d", DIV)
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID, "axes.grid": True,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150})

    mm = np.load(meanforce, allow_pickle=False)
    R_all = mm["R"].astype(np.float64); F_all = mm["F"].astype(np.float64)
    tight = np.load(tight_dataset, allow_pickle=False)
    Rt = np.asarray(tight["R"], np.float64)[tight["origin"] == 1]
    Ft = np.asarray(tight["F"], np.float64)[tight["origin"] == 1]
    ref = np.load(reference, allow_pickle=False)
    R_ref = np.asarray(ref["R"], np.float64)

    ip, ipsi = mp.cvs["phi"].bead_indices, mp.cvs["psi"].bead_indices
    sp, spsi = mp.cvs["phi"].shift_deg, mp.cvs["psi"].shift_deg
    cv = lambda R: (wrap_deg(dihedral_deg(R, ip) + sp), wrap_deg(dihedral_deg(R, ipsi) + spsi))

    nb = 48
    E = np.linspace(-180, 180, nb + 1); C = 0.5 * (E[1:] + E[:-1])

    def field(R, F):
        """bin-averaged generalised force (F_phi, F_psi) in kcal/mol/deg, and counts"""
        phi, psi = cv(R)
        Jp, Js = dihed_grad(R, ip), dihed_grad(R, ipsi)
        Jp = Jp.reshape(len(R), -1); Js = Js.reshape(len(R), -1); Ff = F.reshape(len(R), -1)
        a = (Jp * Jp).sum(1); b = (Jp * Js).sum(1); d = (Js * Js).sum(1)
        det = a * d - b * b
        gp = (Jp * Ff).sum(1); gs = (Js * Ff).sum(1)
        Fphi = ( d * gp - b * gs) / det          # (J J^T)^-1 J F
        Fpsi = (-b * gp + a * gs) / det
        i = np.clip(np.digitize(phi, E) - 1, 0, nb - 1)
        j = np.clip(np.digitize(psi, E) - 1, 0, nb - 1)
        n = np.zeros((nb, nb)); sx = np.zeros((nb, nb)); sy = np.zeros((nb, nb))
        np.add.at(n, (i, j), 1); np.add.at(sx, (i, j), Fphi); np.add.at(sy, (i, j), Fpsi)
        with np.errstate(invalid="ignore"):
            return np.where(n > 0, sx / n, np.nan), np.where(n > 0, sy / n, np.nan), n

    def integrate(Fx, Fy):
        """Poisson-solve grad^2 A = div(-F) on the periodic (phi,psi) torus, by FFT."""
        gx = np.nan_to_num(-Fx); gy = np.nan_to_num(-Fy)          # target grad A
        dx = (E[1] - E[0])
        div = ((np.roll(gx, -1, 0) - np.roll(gx, 1, 0)) + (np.roll(gy, -1, 1) - np.roll(gy, 1, 1))) / (2 * dx)
        k = 2 * np.pi * np.fft.fftfreq(nb, d=dx)
        KX, KY = np.meshgrid(k, k, indexing="ij"); K2 = KX**2 + KY**2; K2[0, 0] = 1.0
        A = np.real(np.fft.ifft2(np.fft.fft2(div) / (-K2))); A[0, 0] = A[0, 0]
        return A - np.nanmin(A)

    # reference FES from the trajectory histogram
    pr, _, _ = np.histogram2d(*cv(R_ref), bins=[E, E]); pr /= pr.sum()
    Aref = -KT * np.log(np.where(pr > 0, pr, np.nan)); Aref -= np.nanmin(Aref)

    sets = [("v3 full (250k)", R_all, F_all), ("v3 tight+fill (117k)", Rt, Ft)]
    out = {}
    for nm, R, F in sets:
        Fx, Fy, n = field(R, F)
        out[nm] = (Fx, Fy, n, integrate(Fx, Fy))
        print(f"{nm}: bins with labels {int((n>0).sum())}/{nb*nb}")

    # ---------------- figure 1: direction field ----------------
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)
    im = ax[0].pcolormesh(E, E, np.where(np.isfinite(Aref), Aref, np.nan).T, cmap=seq,
                          vmin=0, vmax=6, rasterized=True)
    ax[0].set_title("reference FES  $-kT\\ln P(\\phi,\\psi)$"); fig.colorbar(im, ax=ax[0], label="kcal/mol")
    for k, (nm, _, _) in enumerate(sets):
        Fx, Fy, n, _ = out[nm]
        a = ax[k + 1]
        a.pcolormesh(E, E, np.where(np.isfinite(Aref), Aref, np.nan).T, cmap=seq, vmin=0, vmax=6,
                     alpha=0.35, rasterized=True)
        s = 2
        X, Y = np.meshgrid(C[::s], C[::s], indexing="ij")
        U, V = Fx[::s, ::s], Fy[::s, ::s]
        M = np.sqrt(U**2 + V**2)
        a.quiver(X, Y, U, V, M, cmap=div, scale_units="xy", angles="xy",
                 scale=np.nanpercentile(M, 90) / 12, width=0.004)
        a.set_title(f"{nm} — mean-force direction")
    for a in ax:
        a.set_xlim(-180, 180); a.set_ylim(-180, 180); a.set_aspect("equal")
        a.set_xlabel("$\\phi$ (deg)")
    ax[0].set_ylabel("$\\psi$ (deg)")
    fig.suptitle("Arrows = generalised mean force $(F_\\phi, F_\\psi)$. A correct field points "
                 "DOWNHILL, i.e. toward the dark basin minima.", fontsize=10.5)
    force_figure = a.outdir / "fig1_force_directions.png"
    fig.savefig(force_figure, bbox_inches="tight"); plt.close(fig)

    # ---------------- figure 2: integrated FES ----------------
    fig, ax = plt.subplots(1, 4, figsize=(21, 5.0), constrained_layout=True)
    im = ax[0].pcolormesh(E, E, np.where(np.isfinite(Aref), Aref, np.nan).T, cmap=seq, vmin=0, vmax=6)
    ax[0].set_title("reference FES (histogram)"); fig.colorbar(im, ax=ax[0], label="kcal/mol")
    for k, (nm, _, _) in enumerate(sets):
        A = out[nm][3]; n = out[nm][2]
        Am = np.where(n > 0, A, np.nan); Am = Am - np.nanmin(Am)
        im = ax[k + 1].pcolormesh(E, E, Am.T, cmap=seq, vmin=0, vmax=6)
        ax[k + 1].set_title(f"{nm}\nFES INTEGRATED from mean forces")
        fig.colorbar(im, ax=ax[k + 1], label="kcal/mol")
        m = np.isfinite(Aref) & np.isfinite(Am) & (n >= 5)
        r = np.corrcoef(Aref[m], Am[m])[0, 1]
        rms = np.sqrt(np.nanmean((Aref[m] - Am[m] + np.nanmean(Am[m] - Aref[m]))**2))
        print(f"  {nm}: vs reference FES  r={r:.3f}  RMS={rms:.2f} kcal/mol  ({int(m.sum())} bins)")
        out[nm] = out[nm] + (r, rms)
    A = out[sets[1][0]][3]; n = out[sets[1][0]][2]
    Am = np.where(n > 0, A, np.nan); Am -= np.nanmin(Am)
    dd = Am - Aref
    v = np.nanpercentile(np.abs(dd), 98)
    im = ax[3].pcolormesh(E, E, (dd - np.nanmean(dd)).T, cmap=div, vmin=-v, vmax=v)
    ax[3].set_title("tight117k integrated − reference"); fig.colorbar(im, ax=ax[3], label="kcal/mol")
    for a in ax:
        a.set_xlim(-180, 180); a.set_ylim(-180, 180); a.set_aspect("equal"); a.set_xlabel("$\\phi$ (deg)")
    ax[0].set_ylabel("$\\psi$ (deg)")
    fig.suptitle("Free energy obtained by INTEGRATING the mean-force labels (FFT Poisson solve "
                 "on the periodic torus) — no model involved", fontsize=10.5)
    fes_figure = a.outdir / "fig2_integrated_fes.png"
    fig.savefig(fes_figure, bbox_inches="tight"); plt.close(fig)
    print("\nwrote fig1_force_directions.png, fig2_integrated_fes.png")


    manifest = write_manifest(
        a.outdir,
        inputs={"meanforce": meanforce, "tight_dataset": tight_dataset,
                "reference": reference},
        parameters=vars(a), module="analysis.physics.meanforce_fes",
        extra={"force_figure": str(force_figure), "fes_figure": str(fes_figure)})
    print(f"manifest: {manifest}")

if __name__ == "__main__":
    main()
