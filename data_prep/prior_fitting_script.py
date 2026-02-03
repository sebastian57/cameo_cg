#!/usr/bin/env python3

import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import yaml
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from pathlib import Path
import argparse

# JAX/jax_md compatibility patch (must be before any jax_md imports)
# jax_md uses jax.random.KeyArray which was removed in newer JAX versions
import jax
jax.random.KeyArray = jax.Array
_to_uncache = [mod for mod in sys.modules if mod.startswith('jax.random')]
for mod in _to_uncache:
    del sys.modules[mod]

from jax_md import space

jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp

# Logger — matches clean_code_base [Name] format from utils/logging.py
logger = logging.getLogger("PriorFit")
logger.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def order_from_resid(resid):
    r = np.asarray(resid)
    if r.ndim == 2:
        r = r[0]
    return np.argsort(r)

def build_bonds_angles_dihedrals(resid):

    order = order_from_resid(resid)
    N = len(order)

    bonds = np.stack([order[:-1], order[1:]], axis=1).astype(np.int32)
    angles = np.stack([order[:-2], order[1:-1], order[2:]], axis=1).astype(np.int32)

    if N >= 4:
        dihedrals = np.stack([order[:-3], order[1:-2], order[2:-1], order[3:]], axis=1).astype(np.int32)
    else:
        dihedrals = np.zeros((0, 4), dtype=np.int32)

    return (
        jnp.asarray(bonds, dtype=jnp.int32),
        jnp.asarray(angles, dtype=jnp.int32),
        jnp.asarray(dihedrals, dtype=jnp.int32),
    )


def bond_distances_single_frame(R, bonds, displacement):
    i, j = bonds[:, 0], bonds[:, 1]
    Ri, Rj = R[i], R[j]
    dR = jax.vmap(displacement)(Ri, Rj)
    return jnp.linalg.norm(dR, axis=-1)

def angles_single_frame(R, angles, displacement):
    ia, ib, ic = angles[:, 0], angles[:, 1], angles[:, 2]
    Ra, Rb, Rc = R[ia], R[ib], R[ic]

    v1 = jax.vmap(displacement)(Rb, Ra)  # a->b
    v2 = jax.vmap(displacement)(Rb, Rc)  # c->b

    dot = jnp.einsum("ij,ij->i", v1, v2)
    n1 = jnp.linalg.norm(v1, axis=-1)
    n2 = jnp.linalg.norm(v2, axis=-1)
    cos_th = dot / (n1 * n2 + 1e-12)
    cos_th = jnp.clip(cos_th, -1.0, 1.0)
    return jnp.arccos(cos_th)  # (0, pi)

def dihedral_angles_single_frame(R, dihedrals, displacement):

    if dihedrals.shape[0] == 0:
        return jnp.zeros((0,), dtype=R.dtype)

    i, j, k, l = dihedrals[:, 0], dihedrals[:, 1], dihedrals[:, 2], dihedrals[:, 3]
    Ri, Rj, Rk, Rl = R[i], R[j], R[k], R[l]

    b1 = jax.vmap(displacement)(Rj, Ri)  # i - j
    b2 = jax.vmap(displacement)(Rk, Rj)  # j - k
    b3 = jax.vmap(displacement)(Rl, Rk)  # k - l

    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)

    b2_norm = jnp.linalg.norm(b2, axis=-1)
    b2_hat = b2 / (b2_norm[:, None] + 1e-12)

    m1 = jnp.cross(n1, b2_hat)

    x = jnp.sum(n1 * n2, axis=-1)
    y = jnp.sum(m1 * n2, axis=-1)

    return jnp.arctan2(y, x)


def fit_bond_harmonic(d_np, T=320.0, kB=0.0019872041):
    d_np = np.asarray(d_np, dtype=np.float64)
    r0 = float(np.mean(d_np))
    var = float(np.mean((d_np - r0) ** 2))
    kBT = kB * T
    kr = float(kBT / max(var, 1e-12))
    return r0, kr

def fit_fourier_angles_stable(
    a_np,
    n_terms=50,
    nbins=400,
    T=320.0,
    kB=0.0019872041,
    theta_min=1.0,
    theta_max=2.8,
    min_count=50,
    pseudocount=5.0,
    ridge=1e-2,
):
    beta = 1.0 / (kB * T)

    counts, edges = np.histogram(a_np, bins=nbins, range=(theta_min, theta_max), density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = counts >= min_count
    if mask.sum() < 10:
        raise ValueError(f"Too few bins above min_count={min_count}. Lower min_count or increase data/nbins.")

    counts_smooth = counts.astype(np.float64) + pseudocount
    P_bin = counts_smooth / counts_smooth.sum()
    bin_width = edges[1] - edges[0]
    P = P_bin / bin_width

    sinth = np.clip(np.sin(centers), 1e-12, None)
    P_corr = np.clip(P / sinth, 1e-300, None)

    U = -(1.0 / beta) * np.log(P_corr)
    U = U - np.mean(U[mask])

    N = n_terms
    n = np.arange(1, N + 1)[None, :]
    th = centers[:, None]
    Xc = np.cos(n * th)
    Xs = np.sin(n * th)
    X = np.concatenate([Xc, Xs], axis=1)

    Xm = X[mask]
    Um = U[mask]

    w = np.clip(P[mask], 1e-12, None)
    Xm_w = Xm * w[:, None]
    Um_w = Um * w

    A = Xm_w.T @ Xm_w + ridge * np.eye(2 * N)
    y = Xm_w.T @ Um_w
    coeff = np.linalg.solve(A, y)

    a = coeff[:N]
    b = coeff[N:]
    return a, b, centers, U, mask, edges, counts


def U_fourier(theta, a, b):
    theta = np.asarray(theta)
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    n = np.arange(1, len(a) + 1)[None, :]
    th = theta[:, None]
    return (np.cos(n * th) @ a) + (np.sin(n * th) @ b)

def prior_pdf_from_U_theta(theta, U, T=320.0, kB=0.0019872041):
    beta = 1.0 / (kB * T)
    U_shift = U - np.min(U)
    w = np.sin(theta) * np.exp(-beta * U_shift)
    Z = np.trapz(w, theta)
    return w / Z


def fit_dihedral_fourier_2term(
    phi_np,
    nbins=360,
    T=320.0,
    kB=0.0019872041,
    min_count=20,
    pseudocount=0.1,
    ridge=1e-2,
):

    beta = 1.0 / (kB * T)

    counts, edges = np.histogram(phi_np, bins=nbins, range=(-np.pi, np.pi), density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = counts >= min_count
    if mask.sum() < 10:
        raise ValueError(f"Too few bins above min_count={min_count} for dihedrals.")

    counts_smooth = counts.astype(np.float64) + pseudocount
    P_bin = counts_smooth / counts_smooth.sum()
    bin_width = edges[1] - edges[0]
    P = P_bin / bin_width

    U = -(1.0 / beta) * np.log(np.clip(P, 1e-300, None))
    U = U - np.mean(U[mask])

    th = centers
    X = np.stack([np.cos(th), np.sin(th), np.cos(2 * th), np.sin(2 * th)], axis=1)  # (M,4)

    Xm = X[mask]
    Um = U[mask]

    w = np.clip(P[mask], 1e-12, None)
    Xm_w = Xm * w[:, None]
    Um_w = Um * w

    A = Xm_w.T @ Xm_w + ridge * np.eye(4)
    y = Xm_w.T @ Um_w
    coeff = np.linalg.solve(A, y)  # [A1,B1,A2,B2]

    A1, B1, A2, B2 = coeff
    k1 = float(np.sqrt(A1 * A1 + B1 * B1))
    g1 = float(np.arctan2(B1, A1))
    k2 = float(np.sqrt(A2 * A2 + B2 * B2))
    g2 = float(np.arctan2(B2, A2))

    return (k1, k2), (g1, g2), centers, U, mask, edges, counts, coeff


def prior_pdf_from_U_phi(phi, U, T=320.0, kB=0.0019872041):
    beta = 1.0 / (kB * T)
    U_shift = U - np.min(U)
    w = np.exp(-beta * U_shift)
    Z = np.trapz(w, phi)
    return w / Z


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Fit CG priors from dataset distributions and output YAML + plots.")
    parser.add_argument("--data", action="append", default=[], help="Path to a dataset npz. Can be repeated.")
    parser.add_argument("--out_yaml", default="fitted_priors.yaml", help="Output YAML filename.")
    parser.add_argument("--plots_dir", default="plots", help="Directory to save plots.")
    parser.add_argument("--T", type=float, default=320.0, help="Temperature (K).")
    parser.add_argument("--kB", type=float, default=0.0019872041, help="Boltzmann constant in kcal/mol/K.")
    parser.add_argument("--angle_terms", type=int, default=10, help="Number of Fourier terms for angle PMF fit.")
    parser.add_argument("--angle_bins", type=int, default=400, help="Histogram bins for angle PMF.")
    parser.add_argument("--dih_bins", type=int, default=360, help="Histogram bins for dihedral PMF.")
    parser.add_argument("--min_count", type=int, default=50, help="Minimum count per bin to include in PMF fit.")
    parser.add_argument("--ridge", type=float, default=1e-2, help="Ridge regularization for PMF fits.")
    parser.add_argument("--pseudocount", type=float, default=1.0, help="Histogram pseudocount.")
    parser.add_argument("--theta_min", type=float, default=1e-4)
    parser.add_argument("--theta_max", type=float, default=np.pi - 1e-4)
    # Repulsion parameters: optionally set in output YAML
    parser.add_argument("--epsilon", type=float, default=1.0, help="Repulsion epsilon to write to YAML.")
    parser.add_argument("--sigma", type=float, default=3.0, help="Repulsion sigma to write to YAML.")
    parser.add_argument("--rep_power", type=int, default=4, help="Repulsion power in (sigma/r)^power (for plotting only).")
    parser.add_argument("--skip_plots", action="store_true", default=False,
                        help="Skip all matplotlib figure generation.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable DEBUG-level logging.")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    dataset_paths = args.data if len(args.data) > 0 else [
        "datasets/4zohB01_temps_kcalmol_notnorm_1bead_aggforce.npz",
    ]

    if not args.skip_plots:
        ensure_dir(args.plots_dir)

    displacement, _ = space.free()

    all_bonds = []
    all_angles = []
    all_dihedrals = []

    all_rep_dists = []

    for path in dataset_paths:
        logger.info(f"Loading: {path}")
        data = np.load(path, allow_pickle=True)

        R = data["R"].astype(np.float32)       # (T,N,3)
        resid = data["resid"]                  # (T,N) or (N,)

        Tframes, N, _ = R.shape
        bonds, angles, dihedrals = build_bonds_angles_dihedrals(resid)

        R_jax = jnp.asarray(R)

        bond_all = jax.vmap(bond_distances_single_frame, in_axes=(0, None, None))(R_jax, bonds, displacement)
        ang_all  = jax.vmap(angles_single_frame, in_axes=(0, None, None))(R_jax, angles, displacement)
        dih_all  = jax.vmap(dihedral_angles_single_frame, in_axes=(0, None, None))(R_jax, dihedrals, displacement)

        d_np = np.asarray(bond_all).ravel()
        a_np = np.asarray(ang_all).ravel()
        phi_np = np.asarray(dih_all).ravel()

        d_np = d_np[np.isfinite(d_np)]
        a_np = a_np[np.isfinite(a_np)]
        phi_np = phi_np[np.isfinite(phi_np)]

        all_bonds.append(d_np)
        all_angles.append(a_np)
        all_dihedrals.append(phi_np)

        order = order_from_resid(resid)
        if N > 6:
            ii = order[:-6]
            jj = order[6:]
            sample_frames = np.linspace(0, Tframes - 1, num=min(Tframes, 200), dtype=int)
            Rsub = R[sample_frames]
            dr = Rsub[:, ii, :] - Rsub[:, jj, :]
            rep_d = np.linalg.norm(dr, axis=-1).ravel()
            rep_d = rep_d[np.isfinite(rep_d)]
            all_rep_dists.append(rep_d)

    all_bonds = np.concatenate(all_bonds, axis=0)
    all_angles = np.concatenate(all_angles, axis=0)
    all_dihedrals = np.concatenate(all_dihedrals, axis=0)
    all_rep_dists = np.concatenate(all_rep_dists, axis=0) if len(all_rep_dists) else None

    logger.info(f"Total samples: bonds={all_bonds.shape[0]}, angles={all_angles.shape[0]}, dihedrals={all_dihedrals.shape[0]}")

    # --- Raw histogram plots ---
    if not args.skip_plots:
        plt.figure()
        plt.hist(all_bonds, bins=100)
        plt.xlabel("Bond distance r (Å)")
        plt.ylabel("Count")
        plt.title("Bond distance histogram (all datasets)")
        savefig(Path(args.plots_dir) / "bond_distance_hist.png")

        plt.figure()
        plt.hist(all_angles, bins=100)
        plt.xlabel("Angle θ (rad)")
        plt.ylabel("Count")
        plt.title("Angle histogram (all datasets)")
        savefig(Path(args.plots_dir) / "angle_hist.png")

        plt.figure()
        plt.hist(all_dihedrals, bins=120, range=(-np.pi, np.pi))
        plt.xlabel("Dihedral φ (rad)")
        plt.ylabel("Count")
        plt.title("Dihedral histogram (all datasets)")
        savefig(Path(args.plots_dir) / "dihedral_hist.png")

        if all_rep_dists is not None and all_rep_dists.size > 0:
            plt.figure()
            plt.hist(all_rep_dists, bins=120)
            plt.xlabel("Rep pair distance r (Å) (i,i+6 sampled)")
            plt.ylabel("Count")
            plt.title("Repulsion-pair distance histogram (sampled)")
            savefig(Path(args.plots_dir) / "repulsion_pair_distance_hist.png")

    # --- Bond fit ---
    r0, kr = fit_bond_harmonic(all_bonds, T=args.T, kB=args.kB)
    logger.info(f"Bond fit: r0={r0:.5f} Å, kr={kr:.5f} kcal/mol/Å^2")

    if not args.skip_plots:
        r_grid = np.linspace(max(1e-3, np.min(all_bonds)), np.max(all_bonds), 2000)
        U_bond = 0.5 * kr * (r_grid - r0) ** 2
        U_bond -= U_bond.min()
        beta = 1.0 / (args.kB * args.T)
        pdf_bond = np.exp(-beta * U_bond)
        pdf_bond /= np.trapz(pdf_bond, r_grid)

        hist_r, edges_r = np.histogram(all_bonds, bins=120, density=True)
        cent_r = 0.5 * (edges_r[:-1] + edges_r[1:])

        plt.figure()
        plt.plot(cent_r, hist_r, label="Empirical P(r)", linewidth=2)
        plt.plot(r_grid, pdf_bond, label="Prior-implied (Gaussian sanity)", linewidth=2)
        plt.xlabel("r (Å)")
        plt.ylabel("density")
        plt.title("Bond prior sanity check")
        plt.legend()
        savefig(Path(args.plots_dir) / "bond_prior_implied_P.png")

    # --- Angle fit ---
    a_coeff, b_coeff, centers_th, U_bins_th, mask_th, edges_th, counts_th = fit_fourier_angles_stable(
        all_angles,
        n_terms=args.angle_terms,
        nbins=args.angle_bins,
        T=args.T,
        kB=args.kB,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        min_count=args.min_count,
        pseudocount=args.pseudocount,
        ridge=args.ridge,
    )

    logger.info(f"Angle Fourier fit: {args.angle_terms} terms")
    logger.debug(f"  a: {a_coeff}")
    logger.debug(f"  b: {b_coeff}")

    if not args.skip_plots:
        theta_grid = np.linspace(1.0, 2.8, 2000)
        U_fit_th = U_fourier(theta_grid, a_coeff, b_coeff)
        U_fit_th -= np.mean(U_fit_th)

        pdf_th_fit = prior_pdf_from_U_theta(theta_grid, U_fit_th, T=args.T, kB=args.kB)

        # Empirical angle density
        hist_th, edges = np.histogram(all_angles, bins=180, range=(args.theta_min, args.theta_max), density=True)
        cent_th = 0.5 * (edges[:-1] + edges[1:])

        # Plot PMF bins and fitted U(theta)
        plt.figure()
        plt.plot(theta_grid, U_fit_th, label="Fitted U(θ) Fourier")
        plt.scatter(centers_th[mask_th], U_bins_th[mask_th], s=10, alpha=0.6, label="PMF bins used")
        plt.xlabel("θ (rad)")
        plt.ylabel("U (kcal/mol) up to constant")
        plt.title("Angle PMF + Fourier fit")
        plt.legend()
        savefig(Path(args.plots_dir) / "angle_pmf_fit.png")

        # Plot implied angle density
        plt.figure()
        plt.plot(cent_th, hist_th, label="Empirical P(θ)", linewidth=2)
        plt.plot(theta_grid, pdf_th_fit, label="Prior-implied P(θ) (Fourier)", linewidth=2)
        plt.xlabel("θ (rad)")
        plt.ylabel("density")
        plt.title("Angle prior implied distribution")
        plt.legend()
        savefig(Path(args.plots_dir) / "angle_prior_implied_P.png")

    # --- Dihedral fit ---
    (k1, k2), (g1, g2), centers_phi, U_bins_phi, mask_phi, edges_phi, counts_phi, coeff_phi = fit_dihedral_fourier_2term(
        all_dihedrals,
        nbins=args.dih_bins,
        T=args.T,
        kB=args.kB,
        min_count=args.min_count,
        pseudocount=args.pseudocount,
        ridge=args.ridge,
    )

    logger.info(f"Dihedral fit (2-term): k=[{k1:.4f}, {k2:.4f}], gamma=[{g1:.4f}, {g2:.4f}]")
    logger.debug(f"  raw coeff [A1,B1,A2,B2]: {coeff_phi}")

    if not args.skip_plots:
        phi_grid = np.linspace(-np.pi, np.pi, 2000, endpoint=False)
        A1, B1, A2, B2 = coeff_phi
        U_fit_phi = A1 * np.cos(phi_grid) + B1 * np.sin(phi_grid) + A2 * np.cos(2 * phi_grid) + B2 * np.sin(2 * phi_grid)
        U_fit_phi -= np.mean(U_fit_phi)

        pdf_phi_fit = prior_pdf_from_U_phi(phi_grid, U_fit_phi, T=args.T, kB=args.kB)

        # Empirical dihedral density
        hist_phi, edges = np.histogram(all_dihedrals, bins=180, range=(-np.pi, np.pi), density=True)
        cent_phi = 0.5 * (edges[:-1] + edges[1:])

        # Plot dihedral PMF and fit
        plt.figure()
        plt.plot(phi_grid, U_fit_phi, label="Fitted U(φ) (2-term)")
        plt.scatter(centers_phi[mask_phi], U_bins_phi[mask_phi], s=10, alpha=0.6, label="PMF bins used")
        plt.xlabel("φ (rad)")
        plt.ylabel("U (kcal/mol) up to constant")
        plt.title("Dihedral PMF + fit")
        plt.legend()
        savefig(Path(args.plots_dir) / "dihedral_pmf_fit.png")

        # Plot implied dihedral density
        plt.figure()
        plt.plot(cent_phi, hist_phi, label="Empirical P(φ)", linewidth=2)
        plt.plot(phi_grid, pdf_phi_fit, label="Prior-implied P(φ)", linewidth=2)
        plt.xlabel("φ (rad)")
        plt.ylabel("density")
        plt.title("Dihedral prior implied distribution")
        plt.legend()
        savefig(Path(args.plots_dir) / "dihedral_prior_implied_P.png")

    # --- Repulsion plots (verification only, no fit) ---
    if not args.skip_plots:
        if all_rep_dists is not None and all_rep_dists.size > 0:
            r = np.linspace(max(1e-3, np.min(all_rep_dists)), np.percentile(all_rep_dists, 99.5), 2000)
            rep = args.epsilon * (args.sigma / np.maximum(r, 1e-6)) ** args.rep_power

            plt.figure()
            plt.plot(r, rep)
            plt.xlabel("r (Å)")
            plt.ylabel(f"E_rep = eps*(sigma/r)^{args.rep_power} (kcal/mol)")
            plt.title("Repulsion energy curve (as configured)")
            savefig(Path(args.plots_dir) / "repulsion_energy_curve.png")

            # Show implied weights (just for intuition)
            beta = 1.0 / (args.kB * args.T)
            w = np.exp(-beta * (args.epsilon * (args.sigma / np.maximum(all_rep_dists, 1e-6)) ** args.rep_power))
            plt.figure()
            plt.scatter(all_rep_dists[:5000], w[:5000], s=3, alpha=0.3)
            plt.xlabel("r (Å) (sampled rep pairs)")
            plt.ylabel("exp(-beta E_rep)")
            plt.title("Repulsion weight vs distance (intuition)")
            savefig(Path(args.plots_dir) / "repulsion_weight_vs_distance.png")

    # --- Write YAML ---
    priors = {
        "r0": float(r0),
        "kr": float(kr),
        "a": [float(x) for x in np.asarray(a_coeff).ravel()],
        "b": [float(x) for x in np.asarray(b_coeff).ravel()],
        "k_dihedral": [float(k1), float(k2)],
        "gamma_dihedral": [float(g1), float(g2)],
        "epsilon": float(args.epsilon),
        "sigma": float(args.sigma),
    }

    out = {
        "model": {
            "priors": priors
        },
        "meta": {
            "datasets_used": dataset_paths,
            "T": float(args.T),
            "kB": float(args.kB),
            "angle_terms": int(args.angle_terms),
            "angle_bins": int(args.angle_bins),
            "dih_bins": int(args.dih_bins),
            "min_count": int(args.min_count),
            "ridge": float(args.ridge),
            "pseudocount": float(args.pseudocount),
        }
    }

    with open(args.out_yaml, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    logger.info("=== Done ===")
    logger.info(f"Wrote YAML: {args.out_yaml}")
    if not args.skip_plots:
        logger.info(f"Plots saved in: {args.plots_dir}")
    logger.info(f"YAML priors keys: {list(priors.keys())}")


if __name__ == "__main__":
    main()
