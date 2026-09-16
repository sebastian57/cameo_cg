#!/usr/bin/env python3
"""ala2 bb6: pseudo mean-force calibration + K-map, vs v4 stencil placement.

For a 6-bead molecule the "local environment" IS the whole molecule, so
environment matching is exact and unambiguous:
  descriptor = 15 ORDERED pairwise distances (beads are distinguishable -> no
  permutation ambiguity) + a signed volume (chirality). This avoids both failure
  modes that broke the protein descriptor (sorted distances are chirality-blind
  and do not pin geometry).

Outputs
  * calibration: sigma_c by split-half, bias(eps), gain -- directly comparable
    to the 4zohB01 protein numbers
  * K-map over (phi,psi): where free labels DO and DO NOT work
  * comparison against the v4 stencil anchor density and the reference density,
    to test whether K carries information beyond 2D density (it counts partners
    in the FULL internal configuration space, the flow/TICA bias works in 2D).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sampling.mapping import get_mapping


def descriptor(R):
    """15 ordered pair distances + signed volume. (n,6,3) -> (n,16)."""
    n = R.shape[0]
    iu = np.triu_indices(R.shape[1], 1)
    d = np.linalg.norm(R[:, iu[0]] - R[:, iu[1]], axis=-1)
    v = np.einsum("ni,ni->n", np.cross(R[:, 1] - R[:, 0], R[:, 2] - R[:, 0]),
                  R[:, 4] - R[:, 0])[:, None] / 10.0     # chirality, scaled to ~A
    return np.concatenate([d, v], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--v4", type=Path, default=None, help="v4 stencil training npz for placement overlay")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--n-query", type=int, default=20000)
    ap.add_argument("--min-sep", type=int, default=50)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.05,0.10,0.15,0.20,0.30,0.50])
    ap.add_argument("--eps-map", type=float, default=0.20)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--bins", type=int, default=48)
    ap.add_argument("--stratify", action="store_true",
                    help="sample queries uniformly over occupied (phi,psi) cells, "
                         "so rare regions are represented (default: uniform over frames)")
    a = ap.parse_args(); a.outdir.mkdir(parents=True, exist_ok=True)
    import jax, jax.numpy as jnp

    d = np.load(a.dataset)
    R = np.asarray(d["R"], np.float32); F = np.asarray(d["F"], np.float32)
    T = len(R); print(f"{a.dataset.name}: {T} frames x {R.shape[1]} beads")
    X = descriptor(R)
    m = get_mapping("ala2_backbone_cb_6")
    phi = m.cvs["phi"].evaluate(R); psi = m.cvs["psi"].evaluate(R)

    rng = np.random.default_rng(0)
    if a.stratify:
        # Uniform-over-FRAMES sampling starves rare (phi,psi) cells of queries,
        # so the K-map goes blank exactly in the low-density regions the
        # placement question is about. Sample uniformly over occupied CELLS
        # instead: every cell that exists in the reference gets queried.
        edq = np.linspace(-180, 180, a.bins + 1)
        ixq = np.clip(np.digitize(phi, edq) - 1, 0, a.bins - 1)
        iyq = np.clip(np.digitize(psi, edq) - 1, 0, a.bins - 1)
        cellq = ixq * a.bins + iyq
        uc = np.unique(cellq)
        per = max(1, a.n_query // len(uc))
        pick = []
        for c in uc:
            mm = np.flatnonzero(cellq == c)          # NOT `m`: that is the CG mapping
            pick.append(rng.choice(mm, min(per, len(mm)), replace=False))
        q = np.sort(np.concatenate(pick))
        print(f"stratified: {len(uc)} occupied cells, <={per} queries each -> {len(q)} queries")
    else:
        q = np.sort(rng.choice(T, min(a.n_query, T), replace=False))
    EPS = np.array(sorted(a.eps)); n_e = len(EPS)
    Xj = jnp.asarray(X)

    @jax.jit
    def dists(qb):
        return jnp.sqrt(jnp.maximum(((qb[:, None, :] - Xj[None, :, :]) ** 2).sum(-1), 0.0))

    def kabsch(P, Q):
        H = Q.T @ P; U, S, Vt = np.linalg.svd(H)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        return Vt.T @ D @ U.T

    Kall = np.zeros((len(q), n_e), np.int32)
    cv_sum = np.zeros(n_e); cv_n = np.zeros(n_e)
    sh_sum = np.zeros(n_e); sh_n = np.zeros(n_e)
    ei_map = int(np.argmin(np.abs(EPS - a.eps_map)))
    Fbar_map = np.zeros((len(q), 6, 3), np.float32)

    for s in range(0, len(q), a.chunk):
        qi = q[s:s+a.chunk]
        dd = np.array(dists(jnp.asarray(X[qi])))   # np.asarray gives a READ-ONLY view of a jax array
        sep = np.abs(qi[:, None] - np.arange(T)[None, :])
        dd[sep < a.min_sep] = np.inf
        for r, gi in enumerate(qi):
            Rq = R[gi]
            for e, eps in enumerate(EPS):
                Kall[s+r, e] = int((dd[r] <= eps).sum())
            # Calibration must score EVERY eps on the SAME queries, or the rows
            # compare different populations (this is what made the protein bias
            # run backwards: coverage 15.5% at the tightest eps vs 99.6% at the
            # widest). Gate on having a partner at the TIGHTEST eps.
            common = Kall[s+r, 0] > 0
            # Batch the alignment ONCE at the widest eps, then subset. Doing a
            # Python-level SVD per partner per eps costs ~9 s/query (job 1500179,
            # killed); batched it is ~10 ms.
            idx_all = np.flatnonzero(dd[r] <= EPS[-1])
            if len(idx_all) == 0:
                continue
            Rc = Rq - Rq.mean(0)
            Rj = R[idx_all] - R[idx_all].mean(1, keepdims=True)      # (K,6,3)
            H = np.einsum("kmi,mj->kij", Rj, Rc)                      # (K,3,3)
            U, S, Vt = np.linalg.svd(H)                               # batched
            dsg = np.sign(np.linalg.det(np.einsum("kij,kjl->kil",
                                                  Vt.transpose(0,2,1), U.transpose(0,2,1))))
            D = np.zeros((len(idx_all), 3, 3)); D[:,0,0] = 1; D[:,1,1] = 1; D[:,2,2] = dsg
            Rot = np.einsum("kij,kjl,klm->kim", Vt.transpose(0,2,1), D, U.transpose(0,2,1))
            Frot_all = np.einsum("kij,kmj->kmi", Rot, F[idx_all])     # (K,6,3)
            dsub = dd[r][idx_all]
            for e, eps in enumerate(EPS):
                sel = dsub <= eps
                if not sel.any(): continue
                idx = idx_all[sel]
                Fr = Frot_all[sel]
                fb = Fr.mean(0)
                if common:
                    cv_sum[e] += ((F[gi] - fb) ** 2).sum(); cv_n[e] += 1
                if common and len(idx) >= 2:
                    h = len(idx)//2
                    fa_, fb_ = Fr[:h].mean(0), Fr[h:].mean(0)
                    keff = 4.0/(1.0/h + 1.0/(len(idx)-h))
                    sh_sum[e] += ((fa_-fb_)**2).sum()*keff/(12.0*6); sh_n[e] += 1
                if e == ei_map: Fbar_map[s+r] = fb
        if (s//a.chunk) % 10 == 0: print(f"  query {s}/{len(q)}")

    mean_cv = cv_sum/np.maximum(cv_n,1)/6.0     # per bead
    # NOTE ON ESTIMATORS -- these measure DIFFERENT variances and must not be mixed:
    #   split-half : sigma_n^2 + Var_ball(F*)/3   (spread ACROSS the ball)
    #   CV         : bias_centre^2 + 3*sigma_n^2*(1+1/K)
    # The query sits at the CENTRE of its own ball, so first-order variation
    # cancels; for harmonic modes the mean force is linear in displacement and
    # the cancellation is exact. Using split-half sigma in the CV formula forced
    # bias^2 negative in every earlier run (jobs 1499160/1499202/1502777).
    # sigma_n therefore comes from CV itself, and its CONSTANCY across eps is the
    # diagnostic for whether bias_centre is really negligible.
    common_mask = Kall[:, 0] > 0
    mean_k = Kall[common_mask].mean(0) if common_mask.any() else Kall.mean(0)
    print(f"calibration uses the {common_mask.sum()} of {len(Kall)} queries with a "
          f"partner at eps={EPS[0]:.2f} (common subset, so eps rows are comparable)")
    sig2_sh = np.median((sh_sum/np.maximum(sh_n,1))[sh_n > 50]) if (sh_n > 50).any() else np.nan
    # sigma_n from CV, per eps. If bias_centre ~ 0 these should agree across eps.
    sig2_cv = mean_cv / (3.0*(1.0 + 1.0/np.maximum(mean_k, 1)))
    print(f"\nsigma_n from CV, per eps (FLAT => bias_centre negligible):")
    for e, eps in enumerate(EPS):
        print(f"   eps {eps:5.3f}  K {mean_k[e]:8.1f}  E|f-Fbar|^2 {mean_cv[e]:9.2f}  "
              f"sigma_n {np.sqrt(sig2_cv[e]):7.3f}")
    sig2 = float(np.min(sig2_cv))          # tightest-eps-consistent value
    sig_c = float(np.sqrt(sig2))
    print(f"\nsigma_n (CV, min over eps) = {sig_c:.3f} kcal/mol/A")
    print(f"split-half sigma            = {np.sqrt(sig2_sh):.3f}  "
          f"(= sigma_n^2 + Var_ball/3; larger by construction, NOT the label noise)")
    bias2 = np.maximum(mean_cv - 3.0*sig2*(1.0+1.0/np.maximum(mean_k,1)), 0.0)
    raw = np.sqrt(3.0*sig2); err = np.sqrt(bias2 + 3.0*sig2/np.maximum(mean_k,1))
    print(f"\nraw single-label error {raw:.3f} kcal/mol/A")
    print(f"{'eps':>7} {'mean K':>9} {'bias':>8} {'label err':>10} {'gain':>7} {'coverage':>9}")
    for e, eps in enumerate(EPS):
        cov = 100*(Kall[:, e] > 0).mean()
        print(f"{eps:7.3f} {mean_k[e]:9.1f} {np.sqrt(bias2[e]):8.3f} {err[e]:10.3f} "
              f"{raw/err[e]:6.2f}x {cov:8.1f}%")

    # ---- K-map and overlays -------------------------------------------------
    ed = np.linspace(-180, 180, a.bins+1)
    Kq = Kall[:, ei_map].astype(float)
    def binmed(vals, px, py):
        H = np.full((a.bins, a.bins), np.nan)
        ix = np.clip(np.digitize(px, ed)-1, 0, a.bins-1); iy = np.clip(np.digitize(py, ed)-1, 0, a.bins-1)
        for i in range(a.bins):
            for j in range(a.bins):
                mk = (ix == i) & (iy == j)
                if mk.sum() >= 3: H[i, j] = np.median(vals[mk])
        return H
    Kmap = binmed(Kq, phi[q], psi[q])
    dens, _, _ = np.histogram2d(phi, psi, bins=[ed, ed]); dens_map = np.where(dens > 0, dens, np.nan)

    panels = [(np.log10(np.maximum(Kmap, 0.5)), f"log10 median K  (eps={EPS[ei_map]:.2f})", "viridis"),
              (np.log10(dens_map), "log10 reference density", "magma")]
    v4map = None
    if a.v4 and a.v4.exists():
        d4 = np.load(a.v4); R4 = np.asarray(d4["R"], np.float32)
        p4 = m.cvs["phi"].evaluate(R4); s4 = m.cvs["psi"].evaluate(R4)
        h4, _, _ = np.histogram2d(p4, s4, bins=[ed, ed])
        v4map = np.where(h4 > 0, h4, np.nan)
        panels.append((np.log10(v4map), "log10 v4 stencil anchor density", "cividis"))

    fig, axes = plt.subplots(1, len(panels), figsize=(6*len(panels), 5.2))
    for ax, (M, t, cm) in zip(np.atleast_1d(axes), panels):
        im = ax.imshow(M.T, origin="lower", extent=[-180,180,-180,180], aspect="auto", cmap=cm)
        ax.set_title(t); ax.set_xlabel("phi [deg]"); ax.set_ylabel("psi [deg]"); fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(a.outdir/"ala2_kmap_vs_placement.png", dpi=150); plt.close(fig)

    # does K carry information beyond 2D density?
    # Split-half reliability: build two K-maps from disjoint halves of the
    # queries. Their correlation is the ceiling any other predictor can reach,
    # so it separates real structure in K from per-cell estimation noise.
    h = rng.permutation(len(q)); hA, hB = h[:len(h)//2], h[len(h)//2:]
    KA = binmed(Kq[hA], phi[q][hA], psi[q][hA])
    KB = binmed(Kq[hB], phi[q][hB], psi[q][hB])
    gh = np.isfinite(KA) & np.isfinite(KB)
    r_half = float(np.corrcoef(np.log10(np.maximum(KA[gh],0.5)),
                               np.log10(np.maximum(KB[gh],0.5)))[0,1]) if gh.sum() > 5 else float("nan")
    # Spearman-Brown: reliability of the FULL map from two half maps
    r_rel = 2*r_half/(1+r_half) if np.isfinite(r_half) else float("nan")
    print(f"\nK-map split-half r = {r_half:+.3f} over {int(gh.sum())} cells "
          f"-> full-map reliability {r_rel:+.3f} (this is the ceiling for ANY predictor)")

    good = np.isfinite(Kmap) & np.isfinite(dens_map)
    lk = np.log10(np.maximum(Kmap[good],0.5)); ld = np.log10(dens_map[good])
    r = np.corrcoef(ld, lk)[0,1]
    print(f"\ncorr(log10 reference density, log10 K) over {good.sum()} cells = {r:+.3f}"
          f"   -> density explains {100*r*r:.1f}% of raw K variance")
    if np.isfinite(r_rel) and r_rel > 0:
        print(f"   corrected for map reliability: density explains "
              f"{100*min(r*r/r_rel,1.0):.1f}% of the RELIABLE K variance "
              f"-> {100*max(1-r*r/r_rel,0.0):.1f}% is real structure density cannot see")
    out = {"sigma_c": sig_c, "raw_label_error": float(raw), "eps": EPS.tolist(),
           # the MEASURED per-eps sigma_n; its rise marks bias onset. Must be
           # saved: it cannot be recovered from (err, bias), which are defined
           # from a single sigma_n and would give a flat line by construction.
           "sigma_n_by_eps": np.sqrt(sig2_cv).tolist(),
           "mean_cv": mean_cv.tolist(), "coverage_pct": [float(100*(Kall[:,e]>0).mean()) for e in range(n_e)],
           "mean_K": mean_k.tolist(), "bias": np.sqrt(bias2).tolist(), "err": err.tolist(),
           "gain": (raw/err).tolist(), "corr_logdens_logK": float(r),
           "dens_explains_K_pct": float(100*r*r), "kmap_split_half_r": r_half,
           "kmap_reliability": r_rel,
           "dens_explains_reliable_pct": float(100*min(r*r/r_rel,1.0)) if np.isfinite(r_rel) and r_rel>0 else None}
    if v4map is not None:
        g2 = np.isfinite(Kmap) & np.isfinite(v4map)
        r2 = np.corrcoef(np.log10(np.maximum(Kmap[g2],0.5)), np.log10(v4map[g2]))[0,1]
        out["corr_logK_log_v4density"] = float(r2)
        print(f"corr(log10 K, log10 v4 stencil density) over {g2.sum()} cells = {r2:+.3f}")
    (a.outdir/"ala2_kmap.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(a.outdir/"ala2_kmap.npz", q=q, K=Kall, eps=EPS, phi=phi[q], psi=psi[q],
                        Kmap=Kmap, dens=dens, Fbar=Fbar_map, sigma_c=sig_c)
    print(f"\nwrote {a.outdir/'ala2_kmap_vs_placement.png'}\n      {a.outdir/'ala2_kmap.json'}")


if __name__ == "__main__":
    main()
