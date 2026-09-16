"""v4 stencil state generation: equilibrium-first anchors, cost-equalised local probe.

Replaces the v3 design (`build_stencil_states.py`). Two independent stages.

=============================================================================
STAGE A -- anchor selection (NO CG model required)
=============================================================================
Anchors are drawn from two sources and merged cell-wise in the frozen flow latent:

  reference : frames of the mapped AA reference trajectory. Equilibrium by construction,
              AA coordinates already exist, and they cost nothing to acquire.
  pool      : frames from the Stage-1/2 enhanced-sampling harvest. Needed ONLY where the
              reference trajectory has no mass -- the rare basins and the transition
              corridor.

Rule: reference-first, pool-fills-gaps. A cell is served by reference frames whenever it has
enough of them, and by pool frames otherwise. This is what shrinks DHH: Stages 1-2 no longer
have to cover configuration space, only the part equilibrium sampling cannot reach.

The merged anchor set is then tempered to  p_ref^alpha  by cell-wise resampling.
  alpha = 1  : exactly the equilibrium density
  alpha = 0  : flat over occupied cells (pure coverage)
  alpha ~ 0.83 : the density realised by `gamma05`, the best bb6 model to date (fitted
                 2026-08-20). tight117k (alpha ~ 1.0) and the raw harvest (alpha ~ 0.70)
                 both lost to it, so the optimum is interior and mild.

v3 instead used 50% p_ref draw + 50% pair-distance FARTHEST POINT. Farthest point optimises
geometric spread, which is why the harvest anchors sat at alpha ~ 0.70 -- far flatter than
anything that has trained well.

=============================================================================
STAGE B -- directions and step sizes (needs any trained CG potential)
=============================================================================
v3 displaced along 6 fixed directions (2 TICA gradients + 4 random) at one Cartesian
eps = 0.08 A. Measured consequences (2026-08-20):

  * off-manifold cost dA = 0.5*lambda*eps^2 spread 3.18x across the sampled directions and
    >12.6x across eigendirections; the stiffest eigendirection sits 41.8 kT out at 2eps,
    Boltzmann weight ~1e-18.
  * the TICA gradient directions are the STIFFEST (v.Hv 864/1205 vs 765-773) -- a third of the
    direction budget spent on the most expensive, least informative directions.
  * model Hessian error is 17% along the softest direction vs 2.4% along the stiffest, so the
    design measured hardest exactly what it already knew.

v4 instead eigendecomposes the model Hessian and steps at constant thermodynamic cost:

    eps_k = min( sqrt(2 * dA* / lambda_k),  eps_max )

Two bounds, for two different reasons:
  * dA*    caps how far off the equilibrium manifold any displacement goes. Because every
           displacement now sits at the SAME dA*, the displaced states carry a HOMOGENEOUS
           Boltzmann weight exp(-dA*/kT) instead of one varying over ~18 orders of magnitude.
           A homogeneous over-weighting is a constant; a heterogeneous one is a distortion.
  * eps_max is a LINEARITY bound, not a cost bound. H(2e)/H(e) = 0.987 is verified only out to
           0.16 A. Soft directions would happily take 0.5 A at equal cost; they are capped
           here instead, and simply end up cheaper than dA*.

Directions are the softest RESOLVABLE modes. Resolvability matters because curvature noise is
sqrt(2)*SE/(2*eps), so a mode needs lambda >= snr_target * that to be measured at all. The two
softest internal modes of ala2 bb6 are typically negative or near zero -- a thermally sampled
frame sits on a slope, not at a minimum -- and are excluded rather than clipped.

A fraction of directions is kept RANDOM (drawn from the same resolvable set) as an unbiased
control, so the campaign can measure whether model-chosen directions actually beat random ones.
Using a model here is not active learning in the risky sense: four models, including one that
never saw a curvature label, agreed on the eigenvectors to |cos| >= 0.995.

=============================================================================
Output schema is a superset of the v3 `stencil_states.npz`.
`build_stencil_campaign.py` consumes only `R` and `anchor`, so it is unaffected, but it needs
a new AA-seed path for reference-sourced anchors (see `anchor_source` / `anchor_index`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = "/e/project1/cameo/schmidt36/cameo_cg"
KT_298 = 0.5921868690749673

SRC_REFERENCE = 0
SRC_POOL = 1


# --------------------------------------------------------------------- stage A
def latent_of(R, bias, params_f, cfg_f, to_latent):
    z = np.asarray(bias.projection.transform(np.asarray(R, np.float64)), np.float64)
    return np.asarray(to_latent(params_f, cfg_f, z[:, :cfg_f.n_dims]), np.float64)


def load_exact_anchor_selection(path, reference_R, pool_R=None):
    """Load central anchors selected by a prior stage without re-selection."""
    st = np.load(path)
    if "R" not in st or "anchor_source" not in st or "anchor_index" not in st:
        raise ValueError(f"exact anchor file {path} needs R, anchor_source and anchor_index")
    R0 = np.asarray(st["R"], np.float64)
    reference_R = np.asarray(reference_R)
    if R0.ndim != 3 or R0.shape[1:] != reference_R.shape[1:]:
        raise ValueError(f"exact anchor coordinates have shape {R0.shape}; expected "
                         f"(n, {reference_R.shape[1]}, 3)")
    src = np.asarray(st["anchor_source"], np.int8)
    idx = np.asarray(st["anchor_index"], np.int64)
    if len(src) != len(R0) or len(idx) != len(R0):
        raise ValueError("exact anchor metadata must have one row per anchor")
    if np.any(~np.isin(src, (SRC_REFERENCE, SRC_POOL))) or np.any(idx < 0):
        raise ValueError("exact anchor metadata contains invalid source or negative index")
    if np.any(idx[src == SRC_REFERENCE] >= len(reference_R)):
        raise ValueError("exact anchor file indexes outside the reference trajectory")
    if np.any(src == SRC_POOL) and pool_R is None:
        raise ValueError("exact anchor file contains pool anchors but --pool is missing")
    if pool_R is not None and np.any(idx[src == SRC_POOL] >= len(pool_R)):
        raise ValueError("exact anchor file indexes outside the pool trajectory")
    if "anchor" in st and not np.array_equal(np.asarray(st["anchor"], np.int64),
                                               np.arange(len(R0))):
        raise ValueError("exact anchor rows must have consecutive anchor IDs")
    if "direction" in st and np.any(np.asarray(st["direction"], np.int8) != -1):
        raise ValueError("exact anchor input must contain central direction==-1 rows only")
    return R0, src, idx, {
        "n_from_reference": int((src == SRC_REFERENCE).sum()),
        "n_from_pool": int((src == SRC_POOL).sum()),
        "cells_occupied": None,
        "cells_reference_only": None,
        "cells_pool_only": None,
        "exact_anchor_input": str(path),
    }


def cell_index(u, bins, lim):
    edges = np.linspace(-lim, lim, bins + 1)
    ix = np.clip(np.digitize(u[:, 0], edges) - 1, 0, bins - 1)
    iy = np.clip(np.digitize(u[:, 1], edges) - 1, 0, bins - 1)
    return ix * bins + iy


def draw_anchors(u_ref, u_pool, n_anchors, alpha, rng, bins, lim, min_cell):
    """Reference-first, pool-fills-gaps, then temper the merged set to p_ref^alpha.

    Returns (source, index, per-cell report). `source` is SRC_REFERENCE / SRC_POOL and
    `index` indexes into the corresponding coordinate array.
    """
    nc = bins * bins
    c_ref = cell_index(u_ref, bins, lim)
    n_ref = np.bincount(c_ref, minlength=nc)
    p_ref = n_ref / max(n_ref.sum(), 1)

    if u_pool is None:
        c_pool = np.empty(0, int); n_pool = np.zeros(nc, int)
    else:
        c_pool = cell_index(u_pool, bins, lim)
        n_pool = np.bincount(c_pool, minlength=nc)

    # a cell's density target uses the REFERENCE density where it exists; cells the reference
    # never visits get the floor p_floor so they are covered but not over-weighted.
    occupied = (n_ref + n_pool) > 0
    p_floor = p_ref[p_ref > 0].min() if (p_ref > 0).any() else 1.0
    p_eff = np.where(n_ref > 0, p_ref, np.where(occupied, p_floor, 0.0))
    tgt = np.where(occupied, p_eff ** alpha, 0.0)
    # a cell can only be served if SOME source has enough frames
    tgt = np.where((n_ref >= min_cell) | (n_pool >= min_cell), tgt, 0.0)
    tgt = tgt / tgt.sum()

    want = np.floor(n_anchors * tgt).astype(int)
    cap = np.maximum(n_ref, n_pool)
    want = np.minimum(want, cap)
    short = n_anchors - want.sum()
    if short > 0:
        room = cap - want
        cand = np.flatnonzero((room > 0) & (tgt > 0))
        if len(cand):
            w = tgt[cand] / tgt[cand].sum()
            want[cand] += np.minimum(rng.multinomial(short, w), room[cand])

    def by_cell(c):
        order = np.argsort(c, kind="stable")
        s = np.searchsorted(c[order], np.arange(nc))
        e = np.searchsorted(c[order], np.arange(nc), side="right")
        return order, s, e

    o_r, s_r, e_r = by_cell(c_ref)
    if len(c_pool):
        o_p, s_p, e_p = by_cell(c_pool)

    src, idx = [], []
    n_from_ref = n_from_pool = 0
    for c in np.flatnonzero(want > 0):
        k = int(want[c])
        if n_ref[c] >= k:                                    # reference-first
            pick = rng.choice(o_r[s_r[c]:e_r[c]], k, replace=False)
            src.append(np.full(k, SRC_REFERENCE, np.int8)); idx.append(pick); n_from_ref += k
        elif len(c_pool) and n_pool[c] >= k:
            pick = rng.choice(o_p[s_p[c]:e_p[c]], k, replace=False)
            src.append(np.full(k, SRC_POOL, np.int8)); idx.append(pick); n_from_pool += k
        else:                                                # split across both
            kr = min(k, n_ref[c])
            if kr:
                pick = rng.choice(o_r[s_r[c]:e_r[c]], kr, replace=False)
                src.append(np.full(kr, SRC_REFERENCE, np.int8)); idx.append(pick); n_from_ref += kr
            kp = k - kr
            if kp and len(c_pool):
                kp = min(kp, n_pool[c])
                pick = rng.choice(o_p[s_p[c]:e_p[c]], kp, replace=False)
                src.append(np.full(kp, SRC_POOL, np.int8)); idx.append(pick); n_from_pool += kp

    return (np.concatenate(src), np.concatenate(idx),
            dict(n_from_reference=n_from_ref, n_from_pool=n_from_pool,
                 cells_occupied=int(occupied.sum()),
                 cells_reference_only=int(((n_ref > 0) & (n_pool == 0)).sum()),
                 cells_pool_only=int(((n_ref == 0) & (n_pool > 0)).sum())))


# --------------------------------------------------------------------- stage B
def rigid_body_basis(R0):
    """Orthonormal basis of the 6 exact PMF zero modes (3 translations + 3 rotations)."""
    n = len(R0)
    c = R0 - R0.mean(0)
    B = np.zeros((6, n, 3))
    for k in range(3):
        B[k, :, k] = 1.0
    B[3, :, 1] = -c[:, 2]; B[3, :, 2] = c[:, 1]
    B[4, :, 0] = c[:, 2];  B[4, :, 2] = -c[:, 0]
    B[5, :, 0] = -c[:, 1]; B[5, :, 1] = c[:, 0]
    Q, _ = np.linalg.qr(B.reshape(6, -1).T)
    return Q


def internal_hessian(H, R0):
    Q = rigid_body_basis(R0)
    P = np.eye(H.shape[0]) - Q @ Q.T
    return P @ H @ P, Q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # sources
    ap.add_argument("--reference", required=True, help="mapped CG reference trajectory npz")
    ap.add_argument("--pool", default=None, help="Stage-2 harvest CG coords npz (gap filling)")
    ap.add_argument("--anchors-npz", default=None,
                    help="exact central-anchor NPZ; bypasses Stage-A re-selection")
    ap.add_argument("--bias-npz", required=True)
    ap.add_argument("--flow", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    # stage A
    ap.add_argument("--n-anchors", type=int, default=46000)
    ap.add_argument("--alpha", type=float, default=0.83)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--lim", type=float, default=4.5)
    ap.add_argument("--min-cell", type=int, default=5)
    # stage B
    ap.add_argument("--n-soft", type=int, default=4)
    ap.add_argument("--n-random", type=int, default=2)
    ap.add_argument("--target-dA", type=float, default=2.5,
                    help="off-manifold cost per displacement, kcal/mol")
    ap.add_argument("--layers", type=float, nargs="+", default=[1.0])
    ap.add_argument("--eps-min", type=float, default=0.02)
    ap.add_argument("--eps-max", type=float, default=0.16, help="verified linearity bound")
    ap.add_argument("--label-se", type=float, default=3.615)
    ap.add_argument("--snr-target", type=float, default=3.0)
    ap.add_argument("--model-config", default=None)
    ap.add_argument("--model-params", default=None)
    # validation
    ap.add_argument("--mapping", default="ala2_backbone_cb_6")
    ap.add_argument("--max-bond", type=float, default=3.0)
    ap.add_argument("--min-pair", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--dry-run", action="store_true", help="stage A only")
    ap.add_argument("--anchors-only", action="store_true",
                    help="write stencil_states.npz with anchor rows only "
                         "(no stage B displacements; no model needed)")
    ap.add_argument("--extra-anchor-npz", default=None,
                    help="npz with anchor_source/anchor_index arrays appended to "
                         "the drawn anchors (e.g. corridor/rim emphasis frames)")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    import sys
    sys.path.insert(0, PROJECT_ROOT)
    from sampling.biases.tica_regional import SmoothTICABias
    from sampling.flow_density import load_flow, to_latent

    bias = SmoothTICABias.load(a.bias_npz)
    params_f, cfg_f = load_flow(a.flow)
    R_ref = np.asarray(np.load(a.reference)["R"], np.float64)
    u_ref = latent_of(R_ref, bias, params_f, cfg_f, to_latent)
    R_pool = u_pool = None
    if a.pool:
        R_pool = np.asarray(np.load(a.pool)["R"], np.float64)
        u_pool = latent_of(R_pool, bias, params_f, cfg_f, to_latent)

    if a.anchors_npz:
        if a.extra_anchor_npz:
            raise SystemExit("--anchors-npz and --extra-anchor-npz are mutually exclusive")
        R0, src, idx, rep = load_exact_anchor_selection(a.anchors_npz, R_ref, R_pool)
        print(f"[stage A] loaded {len(src):,} exact anchors from {a.anchors_npz}")
    else:
        src, idx, rep = draw_anchors(u_ref, u_pool, a.n_anchors, a.alpha, rng,
                                     a.bins, a.lim, a.min_cell)
    if a.extra_anchor_npz and not a.anchors_npz:
        ex = np.load(a.extra_anchor_npz)
        src = np.concatenate([src, ex["anchor_source"].astype(src.dtype)])
        idx = np.concatenate([idx, ex["anchor_index"].astype(idx.dtype)])
        rep["n_from_reference"] += int((ex["anchor_source"] == SRC_REFERENCE).sum())
        rep["n_from_pool"] += int((ex["anchor_source"] == SRC_POOL).sum())
        print(f"[stage A] +{len(ex['anchor_index']):,} extra anchors from "
              f"{a.extra_anchor_npz}")
    if not a.anchors_npz:
        # NB: np.where would evaluate both branches on ALL rows; with mixed sources
        # each branch must only see its own indices.
        mref = src == SRC_REFERENCE
        R0 = np.empty((len(idx),) + R_ref.shape[1:], dtype=R_ref.dtype)
        R0[mref] = R_ref[np.asarray(idx)[mref]]
        if R_pool is not None:
            R0[~mref] = R_pool[np.asarray(idx)[~mref]]
        else:
            R0[~mref] = R_ref[np.asarray(idx)[~mref]]
    print(f"[stage A] {len(src):,} anchors  "
          f"({rep['n_from_reference']:,} reference / {rep['n_from_pool']:,} pool)")
    if a.anchors_npz:
        print("          exact anchor order and source indices preserved")
    else:
        print(f"          cells: {rep['cells_occupied']} occupied, "
              f"{rep['cells_reference_only']} reference-only, {rep['cells_pool_only']} POOL-ONLY "
              f"<- the only part DHH must still supply")
    ua = latent_of(R0, bias, params_f, cfg_f, to_latent)
    E = np.linspace(-a.lim, a.lim, a.bins + 1)
    ha = np.histogram2d(ua[:, 0], ua[:, 1], bins=[E, E])[0]; ha /= ha.sum()
    hp = np.histogram2d(u_ref[:, 0], u_ref[:, 1], bins=[E, E])[0]; hp /= hp.sum()
    m = (hp > 1e-4) & (ha > 0)
    if a.anchors_npz:
        print("[stage A] density report skipped for exact-anchor input")
    else:
        print(f"[stage A] realised density exponent: "
              f"{np.polyfit(np.log(hp[m]), np.log(ha[m]), 1)[0]:+.3f} (requested {a.alpha})")
    if a.dry_run:
        np.savez_compressed(a.outdir / "anchors.npz", R=R0, anchor_source=src,
                            anchor_index=idx, **{k: v for k, v in rep.items()})
        print(f"wrote {a.outdir}/anchors.npz")
        return
    if a.anchors_only:
        n = len(R0)
        out = a.outdir / "stencil_states.npz"
        np.savez_compressed(out, R=R0.astype(np.float32),
                            anchor=np.arange(n, dtype=np.int64),
                            direction=np.full(n, -1, np.int8),
                            multiplier=np.zeros(n, np.float32),
                            eps_state=np.zeros(n, np.float32),
                            lam_state=np.zeros(n, np.float32),
                            kind=np.zeros(n, np.int8),
                            anchor_source=src, anchor_index=idx,
                            layers=np.asarray([1.0]), eps=0.0)
        (a.outdir / "manifest.json").write_text(json.dumps(
            {**vars(a), **rep, "n_states": int(n), "anchors_only": True},
            indent=2, default=str) + "\n")
        print(f"[anchors-only] wrote {n:,} states -> {out}")
        return
    if not (a.model_config and a.model_params):
        raise SystemExit("stage B needs --model-config and --model-params (or use --dry-run)")

    import jax, jax.numpy as jnp
    sys.path.append(f"{PROJECT_ROOT}")
    from analysis.md.analyze_model_residuals_by_region import _load_model
    from sampling.mapping import get_mapping
    mp = get_mapping(a.mapping)
    model, mparams, mask0, species0 = _load_model(a.model_config, a.model_params,
                                                  a.reference, mp.n_beads)
    gradU = jax.grad(lambda r: model.compute_energy(mparams, r, mask0, species0))
    hess = jax.jit(jax.jacfwd(gradU))

    nb = mp.n_beads
    K = a.n_soft + a.n_random
    lam_floor = a.snr_target * np.sqrt(2.0) * a.label_se / (2.0 * a.eps_max)
    print(f"\n[stage B] resolvability floor lambda >= {lam_floor:.1f} "
          f"(SE {a.label_se}, eps_max {a.eps_max}, SNR {a.snr_target})")

    n_states = len(R0) * (1 + K * 2 * len(a.layers))
    S_R = np.zeros((n_states, nb, 3), np.float32); S_anchor = np.zeros(n_states, np.int64)
    S_dir = np.zeros(n_states, np.int8); S_mult = np.zeros(n_states, np.float32)
    S_eps = np.zeros(n_states, np.float32); S_lam = np.zeros(n_states, np.float32)
    S_kind = np.zeros(n_states, np.int8)                    # 0 anchor, 1 soft eigen, 2 random
    lam_keep, eps_keep, rej, skipped = [], [], 0, 0
    n_plus = n_minus = 0
    n_eps_min = n_eps_max = 0
    w = 0
    for i, r0 in enumerate(R0):
        H = np.asarray(hess(jnp.asarray(r0, jnp.float32)), np.float64).reshape(nb * 3, nb * 3)
        H = 0.5 * (H + H.T)
        Hi, _ = internal_hessian(H, r0)
        lam, vec = np.linalg.eigh(Hi)
        keep = np.argsort(np.abs(lam))[6:]                  # drop the 6 rigid-body modes
        lam_i, vec_i = lam[keep], vec[:, keep]
        resolvable = np.flatnonzero(lam_i >= lam_floor)
        if len(resolvable) < a.n_soft:
            skipped += 1
            continue
        order = resolvable[np.argsort(lam_i[resolvable])]   # softest resolvable first
        dirs = [(vec_i[:, j], float(lam_i[j]), 1) for j in order[:a.n_soft]]
        for _ in range(a.n_random):
            c = np.zeros(len(lam_i)); c[resolvable] = rng.normal(size=len(resolvable))
            c /= np.linalg.norm(c)
            dirs.append((c @ vec_i.T, float(c @ (lam_i * c)), 2))

        S_R[w] = r0; S_anchor[w] = i; S_dir[w] = -1; S_kind[w] = 0; w += 1
        for d, (v, lam_k, kind) in enumerate(dirs):
            eps = float(np.clip(np.sqrt(2.0 * a.target_dA / max(lam_k, 1e-9)),
                                a.eps_min, a.eps_max))
            lam_keep.append(lam_k); eps_keep.append(eps)
            n_eps_min += int(eps <= a.eps_min + 1e-9)
            n_eps_max += int(eps >= a.eps_max - 1e-9)
            for L in a.layers:
                for sgn in (+1.0, -1.0):
                    cand = r0 + sgn * L * eps * v.reshape(nb, 3)
                    dd = np.linalg.norm(cand[:, None] - cand[None, :], axis=-1)
                    if (np.linalg.norm(cand[1:] - cand[:-1], axis=-1).max() > a.max_bond
                            or dd[np.triu_indices(nb, 1)].min() < a.min_pair):
                        rej += 1; cand = r0
                    if sgn > 0:
                        n_plus += 1
                    else:
                        n_minus += 1
                    S_R[w] = cand; S_anchor[w] = i; S_dir[w] = d
                    S_mult[w] = sgn * L; S_eps[w] = eps; S_lam[w] = lam_k
                    S_kind[w] = kind; w += 1
        if (i + 1) % 5000 == 0:
            print(f"  ...{i+1:,}/{len(R0):,}", flush=True)

    if a.anchors_npz:
        expected = len(R0) * (1 + K * 2 * len(a.layers))
        if skipped:
            raise SystemExit(f"exact-anchor expansion skipped {skipped} anchors; "
                             "no incomplete v5.1 stencil file was written")
        if rej:
            raise SystemExit(f"exact-anchor expansion rejected {rej} displaced states; "
                             "no incomplete v5.1 stencil file was written")
        if w != expected or n_plus != n_minus:
            raise SystemExit(f"exact-anchor expansion is incomplete: {w}/{expected} states, "
                             f"plus={n_plus}, minus={n_minus}")

    lam_keep = np.asarray(lam_keep); eps_keep = np.asarray(eps_keep)
    dA = 0.5 * lam_keep * eps_keep ** 2
    n_anc = int((S_dir[:w] == -1).sum())
    print(f"\n[stage B] {w:,} states = {n_anc:,} anchors x (1 + {K} x {2*len(a.layers)})"
          f"   ({skipped} anchors skipped: too few resolvable modes)")
    print(f"  lambda : median {np.median(lam_keep):8.1f}  p5 {np.percentile(lam_keep,5):7.1f}"
          f"  p95 {np.percentile(lam_keep,95):7.1f}")
    print(f"  eps    : median {np.median(eps_keep):8.4f} A  p5 {np.percentile(eps_keep,5):.4f}"
          f"  p95 {np.percentile(eps_keep,95):.4f}   "
          f"({100*(eps_keep>=a.eps_max-1e-9).mean():.1f}% at the linearity cap)")
    print(f"  dA     : median {np.median(dA):.3f} kcal/mol ({np.median(dA)/KT_298:.1f} kT), "
          f"spread p95/p5 = {np.percentile(dA,95)/np.percentile(dA,5):.2f}x")
    print(f"  Boltzmann weight of a displaced state: median "
          f"{np.exp(-np.median(dA)/KT_298):.2e}  (v3: 1e-02 inner, 1e-08 outer, "
          f"~1e-18 worst case)")
    print(f"  geometry rejections: {rej}")
    print(f"  activation audit: plus={n_plus:,} minus={n_minus:,} "
          f"eps_min={n_eps_min:,} eps_max={n_eps_max:,}")

    out = a.outdir / "stencil_states.npz"
    np.savez_compressed(out, R=S_R[:w], anchor=S_anchor[:w], direction=S_dir[:w],
                        multiplier=S_mult[:w], eps_state=S_eps[:w], lam_state=S_lam[:w],
                        kind=S_kind[:w], anchor_source=src, anchor_index=idx,
                        layers=np.asarray(a.layers), eps=float(np.median(eps_keep)))
    (a.outdir / "manifest.json").write_text(
        json.dumps({**vars(a), **rep, "lambda_floor": float(lam_floor),
                    "n_states": int(w),
                    "activation_audit": {
                        "anchors_requested": int(len(R0)),
                        "anchors_active": int(n_anc),
                        "anchors_skipped": int(skipped),
                        "states_expected": int(len(R0) * (1 + K * 2 * len(a.layers))),
                        "states_written": int(w),
                        "plus_displacements": int(n_plus),
                        "minus_displacements": int(n_minus),
                        "eps_at_min": int(n_eps_min),
                        "eps_at_max": int(n_eps_max),
                        "geometry_rejections": int(rej),
                    }}, indent=2, default=str) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
