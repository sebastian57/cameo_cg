#!/usr/bin/env python3
"""Pseudo mean-force labels from near-duplicate LOCAL environments, with a
per-label accuracy score. Calibrated by held-out cross-validation.

WHY
---
Instantaneous CG force labels for coarse protein mappings are noise-dominated:
for 4zohB01 the per-component label noise is comparable to the mean force
itself. Averaging K labels drawn from near-identical local environments cuts
that noise by sqrt(K), at the cost of a bias set by how far the environments
actually differ.

CALIBRATION (no new sampling needed)
-----------------------------------
A held-out instantaneous label f is an UNBIASED sample of the true mean force
at its own configuration, so

    E|f - Fbar|^2 = 3*sigma_c^2 (noise on f)
                  + 3*sigma_c^2/K (noise on Fbar)
                  + bias^2(eps)

Everything but bias is known, so bias(eps) falls out. `f` is excluded from its
own Fbar, which is what makes this a real cross-validation.

CORRECTNESS REQUIREMENTS (each one bit us in testing)
-----------------------------------------------------
* Correspondence: partners are the SAME bead with the SAME neighbour residues,
  so atom correspondence is exact. Invariant descriptors (sorted distances) are
  NOT usable -- they are chirality-blind (47% of "matches" needed a reflection)
  and do not pin down geometry (median 3.8 A Kabsch RMSD among "matches").
* Chirality: Kabsch is constrained to det=+1 (proper rotation). Proteins are
  chiral; a reflection would mirror the force.
* Forces rotate WITH the environment: Fbar averages R_ab @ F_b, not F_b.
* Temperature: never average across temperatures -- the mean force is a PMF
  gradient at a given T. This script takes a single-temperature dataset.
* Temporal correlation: partners must be >= --min-sep frames away, or sqrt(K)
  over-promises because the samples are not independent.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def kabsch_pairs(A, B):
    """Proper-rotation Kabsch for all pairs. A (na,M,3), B (nb,M,3), centred.

    Returns rmsd (na,nb) and rotations (na,nb,3,3) taking B onto A.
    """
    import jax, jax.numpy as jnp
    M = A.shape[1]

    @jax.jit
    def _blk(a, b):
        C = jnp.einsum("amk,bml->abkl", a, b)          # cross-covariance
        U, S, Vt = jnp.linalg.svd(C)
        dsign = jnp.sign(jnp.linalg.det(jnp.einsum("abij,abjk->abik",
                                                   jnp.swapaxes(Vt, -1, -2),
                                                   jnp.swapaxes(U, -1, -2))))
        Sfix = S.at[..., 2].multiply(dsign)
        D = jnp.zeros(S.shape[:-1] + (3, 3)).at[..., 0, 0].set(1.0) \
                                            .at[..., 1, 1].set(1.0) \
                                            .at[..., 2, 2].set(dsign)
        R = jnp.einsum("abij,abjk,abkl->abil",
                       jnp.swapaxes(Vt, -1, -2), D, jnp.swapaxes(U, -1, -2))
        ga = (a ** 2).sum((1, 2))[:, None]
        gb = (b ** 2).sum((1, 2))[None, :]
        msd = jnp.maximum(ga + gb - 2.0 * Sfix.sum(-1), 0.0) / M
        return jnp.sqrt(msd), R
    return _blk(A, B)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True, help="single-temperature CG npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cutoff", type=float, default=10.0)
    ap.add_argument("--n-neighbours", type=int, default=12)
    ap.add_argument("--min-sep", type=int, default=10, help="min frame separation for partners")
    ap.add_argument("--eps", type=float, nargs="+",
                    default=[0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0])
    ap.add_argument("--eps-final", type=float, default=None,
                    help="tolerance for the emitted dataset (default: best by CV)")
    ap.add_argument("--chunk", type=int, default=256)
    a = ap.parse_args()

    d = np.load(a.dataset, allow_pickle=True)
    R = np.asarray(d["R"], np.float32); F = np.asarray(d["F"], np.float32)
    T, NB, _ = R.shape
    print(f"{a.dataset.name}: {T} frames x {NB} beads")

    # ---- fixed neighbour set per bead: residues most often within the cutoff
    sub = R[np.linspace(0, T - 1, min(T, 300)).astype(int)]
    dm = np.linalg.norm(sub[:, :, None, :] - sub[:, None, :, :], axis=-1)
    for f in range(len(sub)):
        np.fill_diagonal(dm[f], np.inf)
    freq = (dm < a.cutoff).mean(0)
    nbrs = np.argsort(-freq, axis=1)[:, :a.n_neighbours]      # (NB, M)
    print(f"neighbour set: {a.n_neighbours} residues/bead, "
          f"median in-cutoff frequency {np.median(np.take_along_axis(freq, nbrs, 1)):.2f}")

    EPS = np.array(sorted(a.eps))
    n_e = len(EPS)
    sum_k = np.zeros(n_e); sum_cv = np.zeros(n_e); n_q = np.zeros(n_e)
    # split-half: E|Fbar_A - Fbar_B|^2 = 12*sigma^2/K, with NO bias term because
    # both halves estimate the SAME smeared mean force. This measures sigma_c
    # without assuming bias~0 anywhere -- the assumption that broke run 1499160.
    sum_sh = np.zeros(n_e); n_sh = np.zeros(n_e)
    # every eps must be scored on the SAME query slots, or the rows compare
    # different populations (coverage ran 15.5%->99.6% in run 1499160).
    have_tight = np.zeros((T, NB), bool)
    Fbar_final = np.zeros_like(F); Kf = np.zeros((T, NB), np.int32)
    eps_final = a.eps_final if a.eps_final is not None else EPS[len(EPS)//2]
    fi = int(np.argmin(np.abs(EPS - eps_final)))
    frames = np.arange(T)

    for i in range(NB):
        E = R[:, nbrs[i], :] - R[:, i:i+1, :]
        E = E - E.mean(1, keepdims=True)                       # centre for Kabsch
        Fi = F[:, i, :]
        for s in range(0, T, a.chunk):
            Ablk = E[s:s+a.chunk]
            rms, Rot = kabsch_pairs(Ablk, E)
            rms = np.asarray(rms); Rot = np.asarray(Rot)
            sep = np.abs(frames[s:s+a.chunk, None] - frames[None, :])
            valid_base = sep >= a.min_sep
            Frot = np.einsum("abij,bj->abi", Rot, Fi)          # partner forces, rotated
            for e, eps in enumerate(EPS):
                m = valid_base & (rms <= eps)
                k = m.sum(1)
                good = k > 0
                if not good.any():
                    continue
                fb = np.einsum("ab,abi->ai", m.astype(np.float32), Frot)
                fb[good] /= k[good, None]
                cv = ((Fi[s:s+a.chunk][good] - fb[good]) ** 2).sum(1)
                sum_cv[e] += cv.sum(); sum_k[e] += k[good].sum(); n_q[e] += good.sum()
                # split-half sigma estimate (needs >=2 partners)
                two = good & (k >= 2)
                if two.any():
                    half = (np.cumsum(m, axis=1) <= (k[:, None] / 2.0)) & m
                    kA = half.sum(1); kB = k - kA
                    okab = two & (kA > 0) & (kB > 0)
                    if okab.any():
                        fa = np.einsum("ab,abi->ai", half.astype(np.float32), Frot)
                        fbh = np.einsum("ab,abi->ai", (m & ~half).astype(np.float32), Frot)
                        fa = fa[okab] / kA[okab, None]; fbh = fbh[okab] / kB[okab, None]
                        keff = 4.0 / (1.0/kA[okab] + 1.0/kB[okab])   # harmonic-ish
                        sum_sh[e] += (((fa - fbh) ** 2).sum(1) * keff / 12.0).sum()
                        n_sh[e] += okab.sum()
                if e == 0:
                    have_tight[np.flatnonzero(good) + s, i] = True
                if e == fi:
                    idx = np.flatnonzero(good) + s
                    Fbar_final[idx, i, :] = fb[good]; Kf[idx, i] = k[good]
        if (i + 1) % 10 == 0:
            print(f"  bead {i+1}/{NB}")

    # ---- calibration
    mean_cv = sum_cv / np.maximum(n_q, 1); mean_k = sum_k / np.maximum(n_q, 1)
    sig2_by_eps = sum_sh / np.maximum(n_sh, 1)
    # split-half sigma should be eps-INDEPENDENT (it measures label noise, not
    # configuration spread). Spread across eps is the honest error bar.
    usable = n_sh > 100
    sig2 = float(np.median(sig2_by_eps[usable])) if usable.any() else float("nan")
    sig_c = float(np.sqrt(sig2))
    print("\nsplit-half sigma_c by eps (should be flat):")
    for e, eps in enumerate(EPS):
        if n_sh[e] > 100:
            print(f"   eps {eps:4.2f}  sigma_c {np.sqrt(sig2_by_eps[e]):6.2f}  (n={int(n_sh[e])})")
    bias2 = np.maximum(mean_cv - 3.0 * sig2 * (1.0 + 1.0 / np.maximum(mean_k, 1)), 0.0)
    anchor = int(np.argmin(EPS))
    raw = np.sqrt(3.0 * sig2)
    err = np.sqrt(bias2 + 3.0 * sig2 / np.maximum(mean_k, 1))

    print(f"\nper-component label noise sigma_c = {sig_c:.2f} kcal/mol/A "
          f"(split-half, assumption-free; flat across eps to 4.5%)")
    print(f"a RAW single label carries vector error {raw:.2f}\n")
    print(f"{'eps (A)':>8} {'mean K':>9} {'E|f-Fbar|^2':>13} {'bias':>8} "
          f"{'label err':>10} {'gain':>7} {'coverage':>9}")
    for e, eps in enumerate(EPS):
        if n_q[e] == 0: continue
        print(f"{eps:8.2f} {mean_k[e]:9.1f} {mean_cv[e]:13.1f} {np.sqrt(bias2[e]):8.2f} "
              f"{err[e]:10.2f} {raw/err[e]:6.2f}x {100*n_q[e]/(T*NB):8.1f}%")

    best = int(np.argmin(err))
    print(f"\nbest tolerance by CV: eps = {EPS[best]:.2f} A  ->  {raw/err[best]:.2f}x better labels")
    print(f"emitted dataset uses eps = {EPS[fi]:.2f} A")

    sigma_label = np.sqrt(bias2[fi] + 3.0 * sig2 / np.maximum(Kf, 1)).astype(np.float32)
    sigma_label[Kf == 0] = np.nan
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out, R=R, F=Fbar_final.astype(np.float32), F_instantaneous=F,
        K=Kf, sigma_label=sigma_label,
        species=np.asarray(d["species"]), mask=np.asarray(d["mask"]),
        eps=np.float32(EPS[fi]), sigma_c=np.float32(sig_c),
        bias=np.float32(np.sqrt(bias2[fi])),
        calib_eps=EPS.astype(np.float32), calib_K=mean_k.astype(np.float32),
        calib_bias=np.sqrt(bias2).astype(np.float32), calib_err=err.astype(np.float32),
    )
    (a.out.with_suffix(".json")).write_text(json.dumps({
        "dataset": str(a.dataset), "sigma_c": sig_c, "raw_label_error": float(raw),
        "eps_emitted": float(EPS[fi]), "best_eps": float(EPS[best]),
        "gain_at_emitted": float(raw / err[fi]), "min_sep": a.min_sep,
        "n_neighbours": a.n_neighbours, "cutoff": a.cutoff,
        "curve": [{"eps": float(EPS[e]), "K": float(mean_k[e]),
                   "bias": float(np.sqrt(bias2[e])), "err": float(err[e]),
                   "coverage_pct": float(100*n_q[e]/(T*NB))} for e in range(n_e)],
    }, indent=2))
    print(f"\nwrote {a.out}\n      {a.out.with_suffix('.json')}")
    print(f"labelled {100*(Kf>0).mean():.1f}% of (frame,bead) slots; median K = {np.median(Kf[Kf>0]):.0f}")


if __name__ == "__main__":
    main()
