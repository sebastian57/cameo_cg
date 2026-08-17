#!/usr/bin/env python3
"""Stage-3 v3: SYNTHESISED finite-difference stencils around each anchor.

WHAT THIS FIXES
    v2 Stage 3 measured 42,000 conditional mean forces at farthest-point-selected states. Every
    label was accurate (median SE 0.974 vs a reference conditional sigma of 18.56) and every
    label was ISOLATED. Measured on that dataset, ||dF|| between label pairs is FLAT --
    127.5 / 129.7 / 132.0 at separations 0.25-0.40 / 0.40-0.60 / 0.60-1.00 A. In the linear
    regime ||dF|| ~ ||H dR|| must grow PROPORTIONALLY to dR. Flat means the labels are already
    fully decorrelated: not one pair among the 42,000 samples the linear regime, so the dataset
    contains NO local slope-variation constraint anywhere.

    The 150k AA reference block does not supply it either: ||dF|| 108-112 against a label-noise
    floor of 111.4, i.e. SNR ~ 1.0 -- the differences ARE the noise. And empirically the 192k
    model, which contained those frames, still dug a -0.640 +/- 0.074 kcal/mol artificial well.

    That missing constraint is the freedom the network used to over-sharpen.

THE PHYSICS
    For the CG model, F = -grad A where A is the POTENTIAL OF MEAN FORCE (not a potential
    energy -- Stage 3 freezes the CG beads and averages the AA forces, which is a free-energy
    derivative). With H = grad^2 A,

        F(R0 + eps v) - F(R0 - eps v) ~ -2 eps H(R0) v + O(eps^3),

    so a CENTRAL difference along v measures a Hessian-vector product directly. Several
    directions constrain the action of H on the physically relevant subspace. The Hessian is
    never constructed: the labels themselves carry it, and the network is simply trained on
    them.

WHY CENTRAL, AND WHY TWO LAYERS
    Central differences double the signal (2 eps H v rather than eps H v) for the SAME label
    noise, and cancel both the anchor's own label error and the leading truncation term
    (O(eps^2) instead of O(eps)). Two layers at eps and 2 eps additionally give a Richardson
    combination and -- more usefully here -- an ANHARMONICITY CHECK: if the eps and 2 eps
    estimates of H v disagree beyond noise, that anchor is outside the harmonic regime and the
    curvature label should be down-weighted rather than believed.

PARAMETER CHOICE -- MEASURED, NOT ASSUMED
    Cartesian eps -> induced separations (median over 1500 anchors, random internal directions):

        eps (A)   pair-dist   latent   samples across the 0.525 well   central-diff SNR
          0.05      0.0786    0.1206              4.4                        3.1
          0.08      0.1259    0.2038              2.6                        5.0
          0.10      0.1568    0.2510              2.1                        6.2
          0.16      0.2520    0.4200              1.25                       9.9

    Two constraints pull opposite ways: label noise wants LARGE eps (SNR = 2 H eps / 6.18, with
    H median 192 and the noise on a label difference 6.18 over the 18-vector), while resolving
    the 0.525-latent-unit artificial well wants SMALL eps. eps = 0.08 A satisfies both -- SNR
    5.0 and 2.6 samples across the well -- and the outer layer at 0.16 A carries SNR 9.9 for the
    Hessian-vector product itself. In the steep regions where the model over-sharpens
    (H p90 = 896) these become 23 and 46.

DIRECTIONS
    The PMF is invariant under rigid motion, so 6 of the 18 Cartesian DOF are EXACT zero modes.
    Displacing along them measures nothing and would waste a third of the budget, so they are
    projected out explicitly (3 translations + 3 infinitesimal rotations at the configuration).
    Of the 12 remaining internal DOF, the first two stencil directions are the TICA gradients
    -- the directions in which the model demonstrably errs -- and the other four span a random
    orthonormal complement, re-drawn per anchor so the UNION over anchors constrains all 12.

USAGE
    python -m sampling.build_stencil_states \
        --pool local_work/v2_stage2_harvest/cg_coords_all.npz \
        --bias-npz <smooth_*.npz> --flow <flow_*.npz> \
        --n-anchors 10000 --n-dirs 6 --eps 0.08 --layers 1 2 \
        --outdir local_work/v3_stage3_stencil
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NOISE_DIFF_18VEC = 6.18      # measured on the v2 labels
H_MEDIAN, H_P90 = 192.0, 896.0
WELL_WIDTH_LATENT = 0.525
COST_NODE_S_PER_STATE = 0.882


def _log(m: str) -> None:
    print(m, flush=True)


def rigid_basis(r: np.ndarray) -> np.ndarray:
    """Orthonormal basis (3n, 6) of the rigid-body subspace at configuration r.

    These are EXACT zero modes of the PMF, so displacements must be orthogonal to them or a
    third of the stencil measures nothing. Unweighted (not mass-weighted) is correct here: the
    PMF is invariant under rigid motion regardless of the metric."""
    n = len(r)
    c = r - r.mean(axis=0)
    cols = []
    for k in range(3):
        v = np.zeros((n, 3))
        v[:, k] = 1.0
        cols.append(v.ravel())
    for k in range(3):
        a = np.zeros(3)
        a[k] = 1.0
        cols.append(np.cross(a, c).ravel())
    Q, _ = np.linalg.qr(np.asarray(cols).T)
    return Q


def stencil_directions(r: np.ndarray, tica_jac: np.ndarray, n_dirs: int,
                       rng: np.random.Generator) -> np.ndarray:
    """(n_dirs, 3n) orthonormal internal directions; the first rows are the TICA gradients."""
    n3 = r.size
    Q = rigid_basis(r)
    proj = lambda v: v - Q @ (Q.T @ v)

    basis: list[np.ndarray] = []
    for g in tica_jac.reshape(len(tica_jac), -1):        # (n_tics, 3n)
        v = proj(np.asarray(g, np.float64))
        for b in basis:
            v -= (v @ b) * b
        nrm = np.linalg.norm(v)
        if nrm > 1e-8:
            basis.append(v / nrm)
        if len(basis) == n_dirs:
            break
    while len(basis) < n_dirs:
        v = proj(rng.normal(size=n3))
        for b in basis:
            v -= (v @ b) * b
        nrm = np.linalg.norm(v)
        if nrm > 1e-8:
            basis.append(v / nrm)
    return np.asarray(basis)


def validate(R: np.ndarray, ref_chi_sign: np.ndarray, max_bond: float, min_pair: float):
    """Bond lengths, closest approach and chirality sign -- a displaced CG geometry must remain
    physically accessible or the constrained AA run around it is meaningless."""
    bonds = np.linalg.norm(R[:, 1:] - R[:, :-1], axis=-1)
    dmat = np.linalg.norm(R[:, :, None, :] - R[:, None, :, :], axis=-1)
    iu = np.triu_indices(R.shape[1], k=1)
    minpair = dmat[:, iu[0], iu[1]].min(axis=1)
    v = np.cross(R[:, 1] - R[:, 2], R[:, 3] - R[:, 2])
    chi = np.einsum("ij,ij->i", v, R[:, 4] - R[:, 2])
    return (bonds.max(axis=1) <= max_bond) & (minpair >= min_pair) & \
           (np.sign(chi) == ref_chi_sign), bonds.max(axis=1), minpair


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--bias-npz", required=True)
    ap.add_argument("--flow", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--n-anchors", type=int, default=10000)
    ap.add_argument("--n-dirs", type=int, default=6)
    ap.add_argument("--eps", type=float, default=0.08, help="inner layer, Cartesian Angstrom")
    ap.add_argument("--step-mode", choices=("cartesian", "featurenorm"), default="featurenorm",
                    help="'cartesian' uses one eps for every direction. 'featurenorm' (default) "
                         "rescales each direction so the induced PAIR-DISTANCE displacement is "
                         "equal -- necessary because the TICA directions are SOFT modes: at a "
                         "common Cartesian eps they move the latent far (p90 1.41) while barely "
                         "changing the force, and the stiff orthogonal directions do the "
                         "opposite. Equalising in the 15-dim feature metric probes every mode "
                         "on the scale the NETWORK sees, which is where it interpolates.")
    ap.add_argument("--target-feat", type=float, default=0.126,
                    help="target ||d features|| per inner-layer step, Angstrom; the default is "
                         "what a random internal direction produces at eps=0.08 A")
    ap.add_argument("--layers", type=float, nargs="+", default=[1.0, 2.0],
                    help="layer multipliers of --eps; each is sampled at + and -")
    ap.add_argument("--pref-frac", type=float, default=0.5,
                    help="fraction of anchors drawn with weight ~ p_ref (ensemble correctness); "
                         "the rest by pair-distance farthest point (coverage of the "
                         "genuinely under-sampled corridor)")
    ap.add_argument("--max-bond", type=float, default=3.0)
    ap.add_argument("--min-pair", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--mapping", default="ala2_backbone_cb_6")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    from sampling.biases.tica_regional import SmoothTICABias
    from sampling.flow_density import load_flow, log_prob, to_latent
    from sampling.mapping import dihedral_deg, get_mapping, wrap_deg
    from sampling.select_stage3_patches import farthest_point_gpu

    rng = np.random.default_rng(a.seed)
    mapping = get_mapping(a.mapping)
    bias = SmoothTICABias.load(a.bias_npz)
    params, cfg = load_flow(a.flow)

    R = np.asarray(np.load(a.pool)["R"], np.float64)
    Z = np.asarray(bias.projection.transform(R), np.float64)[:, :cfg.n_dims]
    U = np.asarray(to_latent(params, cfg, Z), np.float64)
    F15 = bias.projection.features(R)
    _log(f"[pool] {len(R)} configurations")

    # ---- anchors: p_ref-weighted (ensemble) + farthest point (coverage) -------------------
    n_pref = int(round(a.pref_frac * a.n_anchors))
    n_geo = a.n_anchors - n_pref
    lp = np.asarray(log_prob(params, cfg, Z), np.float64)
    w = np.exp(lp - lp.max())
    w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    _log(f"[anchors] {n_pref} by p_ref importance draw (ESS {ess:.0f}) + "
         f"{n_geo} by pair-distance farthest point")
    picked = rng.choice(len(R), n_pref, replace=False, p=w).tolist() if n_pref else []
    if n_geo:
        rem = np.setdiff1d(np.arange(len(R)), np.asarray(picked, dtype=int))
        idx, sep = farthest_point_gpu(F15[rem], n_geo, int(rng.integers(len(rem))))
        picked += rem[idx].tolist()
        _log(f"[anchors]   farthest-point final separation {sep[-1]:.4f} A")
    anchors = np.asarray(picked, dtype=int)

    # ---- stencil -------------------------------------------------------------------------
    mult = np.concatenate([[m, -m] for m in a.layers])            # +e,-e,+2e,-2e,...
    per_anchor = 1 + a.n_dirs * len(mult)
    _log(f"[stencil] {a.n_dirs} directions x {len(mult)} offsets {sorted(set(mult))} "
         f"-> {per_anchor} states/anchor, {a.n_anchors * per_anchor} total")

    nb = R.shape[1]
    out_R = np.empty((len(anchors) * per_anchor, nb, 3), np.float64)
    out_anchor = np.empty(len(anchors) * per_anchor, np.int64)
    out_dir = np.empty(len(anchors) * per_anchor, np.int8)
    out_mult = np.empty(len(anchors) * per_anchor, np.float32)

    p = 0
    for ai in anchors:
        r = R[ai]
        _, jac = bias.projection.value_and_jacobian(r)            # (n_tics, nb, 3)
        B = stencil_directions(r, jac, a.n_dirs, rng)
        out_R[p] = r
        out_anchor[p] = ai
        out_dir[p] = -1
        out_mult[p] = 0.0
        p += 1
        # Per-direction step length. featurenorm measures the linear response of the 15 pair
        # distances to a small trial displacement and rescales so every direction moves the
        # network's own input metric equally; a floor and ceiling keep the soft modes from
        # demanding a physically absurd Cartesian step.
        if a.step_mode == "featurenorm":
            f_r = bias.projection.features(r[None])[0]
            trial = 1.0e-3
            rate = np.array([np.linalg.norm(
                bias.projection.features((r + trial * B[di].reshape(nb, 3))[None])[0] - f_r)
                / trial for di in range(a.n_dirs)])
            eps_dir = np.clip(a.target_feat / np.maximum(rate, 1e-9), 0.25 * a.eps, 4.0 * a.eps)
        else:
            eps_dir = np.full(a.n_dirs, a.eps)
        for di in range(a.n_dirs):
            step = (eps_dir[di] * B[di]).reshape(nb, 3)
            for m in mult:
                out_R[p] = r + m * step
                out_anchor[p] = ai
                out_dir[p] = di
                out_mult[p] = m
                p += 1
    assert p == len(out_R)

    # ---- validation ----------------------------------------------------------------------
    cv = lambda X, n: wrap_deg(dihedral_deg(X, mapping.cvs[n].bead_indices)
                               + mapping.cvs[n].shift_deg)
    v0 = np.cross(R[anchors][:, 1] - R[anchors][:, 2], R[anchors][:, 3] - R[anchors][:, 2])
    chi_sign_anchor = np.sign(np.einsum("ij,ij->i", v0, R[anchors][:, 4] - R[anchors][:, 2]))
    ref_sign = np.repeat(chi_sign_anchor, per_anchor)
    ok, maxbond, minpair = validate(out_R, ref_sign, a.max_bond, a.min_pair)
    _log(f"[validate] {ok.sum()}/{len(ok)} states pass ({100*ok.mean():.3f}%); "
         f"max bond p100 {maxbond.max():.3f} A, min pair p0 {minpair.min():.3f} A")
    bad_anchors = np.unique(out_anchor[~ok])
    if len(bad_anchors):
        drop = np.isin(out_anchor, bad_anchors)
        _log(f"[validate] dropping {len(bad_anchors)} anchors entirely "
             f"({drop.sum()} states) so every surviving stencil is COMPLETE -- a partial "
             f"stencil breaks the central-difference pairing")
        out_R, out_anchor, out_dir, out_mult = (out_R[~drop], out_anchor[~drop],
                                                out_dir[~drop], out_mult[~drop])

    # ---- diagnostics ---------------------------------------------------------------------
    lat_all = np.asarray(to_latent(params, cfg,
                                   np.asarray(bias.projection.transform(out_R),
                                              np.float64)[:, :cfg.n_dims]), np.float64)
    inner = np.abs(out_mult) == min(a.layers)
    anchor_row = out_dir == -1
    lat_anchor = {int(k): lat_all[i] for i, k in zip(np.flatnonzero(anchor_row),
                                                     out_anchor[anchor_row])}
    lat_off = np.linalg.norm(lat_all[inner] - np.asarray([lat_anchor[int(k)]
                                                          for k in out_anchor[inner]]), axis=1)
    phi, psi = cv(out_R, "phi"), cv(out_R, "psi")
    reg = np.full(len(out_R), "other", dtype=object)
    reg[(phi > -180) & (phi < -20) & ((psi > 90) | (psi < -150))] = "beta"
    reg[(phi > -160) & (phi < -20) & (psi > -120) & (psi < 50)] = "alphaR"
    reg[(phi > 20) & (phi < 100) & (psi > -20) & (psi < 100)] = "alphaL"
    reg[(phi > -15) & (phi < 15)] = "other"

    summary = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "n_anchors_final": int(len(np.unique(out_anchor))),
        "states_per_anchor": int(per_anchor),
        "n_states": int(len(out_R)),
        "cost_node_hours": float(len(out_R) * COST_NODE_S_PER_STATE / 3600.0),
        "latent_offset_inner_layer": {
            "median": float(np.median(lat_off)), "p90": float(np.percentile(lat_off, 90)),
            "samples_across_well": float(WELL_WIDTH_LATENT / max(np.median(lat_off), 1e-12))},
        "central_difference_snr": {
            "inner_H_median": float(2 * H_MEDIAN * a.eps / NOISE_DIFF_18VEC),
            "outer_H_median": float(2 * H_MEDIAN * a.eps * max(a.layers) / NOISE_DIFF_18VEC),
            "inner_H_p90": float(2 * H_P90 * a.eps / NOISE_DIFF_18VEC)},
        "basins_pct": {b: float(100 * (reg == b).mean())
                       for b in ("beta", "alphaR", "alphaL", "other")},
        "basins_reference_truth": {"beta": 65.20, "alphaR": 29.97, "alphaL": 2.79, "other": 2.04},
        "validation": {"passed_pct": float(100 * ok.mean()),
                       "anchors_dropped": int(len(bad_anchors))},
    }
    np.savez_compressed(a.outdir / "stencil_states.npz",
                        R=out_R.astype(np.float32), anchor=out_anchor,
                        direction=out_dir, multiplier=out_mult,
                        eps=np.float64(a.eps), layers=np.asarray(a.layers, np.float64))
    (a.outdir / "stencil_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    _log("\n=== STENCIL GEOMETRY ===")
    q = summary["latent_offset_inner_layer"]
    _log(f"  inner-layer latent offset: median {q['median']:.4f}  p90 {q['p90']:.4f}")
    _log(f"  --> {q['samples_across_well']:.1f} samples across the {WELL_WIDTH_LATENT} well")
    s = summary["central_difference_snr"]
    _log(f"  central-difference SNR: inner {s['inner_H_median']:.1f} (H median), "
         f"{s['inner_H_p90']:.1f} (H p90) | outer {s['outer_H_median']:.1f}")
    _log("\n=== BASIN BALANCE (%) ===")
    _log(f"  {'':12s} {'beta':>8s} {'alphaR':>8s} {'alphaL':>8s} {'other':>8s}")
    for k in ("basins_pct", "basins_reference_truth"):
        d = summary[k]
        _log(f"  {k[:12]:12s} {d['beta']:8.2f} {d['alphaR']:8.2f} {d['alphaL']:8.2f} "
             f"{d['other']:8.2f}")
    _log(f"\n  {summary['n_states']} states -> {summary['cost_node_hours']:.1f} node-hours")
    _log(f"\nwrote {a.outdir}/stencil_states.npz and stencil_summary.json")


if __name__ == "__main__":
    main()
