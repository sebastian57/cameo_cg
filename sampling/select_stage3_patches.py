#!/usr/bin/env python3
"""Stage-3 state selection: global anchors + local neighbourhoods (patches).

WHY THIS REPLACES PLAIN FARTHEST-POINT
    v2 Stage 3 selected 42,000 states by farthest point in the 15-dim pair-distance space and
    ran one frozen-bead AA simulation at each. Every label was accurate (median SE 0.974 against
    a reference conditional sigma of 18.56) but every label was ISOLATED: a mean force at a
    point constrains the PMF GRADIENT there and says nothing about how that gradient VARIES.
    Between isolated anchors the network is free to invent, and it does -- the latent diagnosis
    (SAMPLING/latent_ensemble_diagnosis) measured an artificial free-energy well of
    -0.640 +/- 0.074 kcal/mol, 0.525 latent units wide, whose depth accounts for the entire
    beta->alphaR population transfer (65.2/30.0 -> 38.4/52.4) to 0.8%.

    A patch of several distinct configurations around each anchor turns
        "the mean force at one point"
    into
        "the mean-force field across a small piece of the manifold",
    which constrains the local slope variation the network currently invents.

TWO MEASURED FACTS THAT SET THE PARAMETERS

  1. THE PATCH RADIUS LIVES IN PAIR-DISTANCE SPACE, NOT LATENT SPACE.
     The network interpolates in its own input space (15 pair distances); that is where it
     invents structure. The latent is 2-dimensional and degenerate -- configurations adjacent
     in u can be far apart in pair-distance -- so it is the right tool to VERIFY that a patch
     resolves the sharpening scale, and the wrong tool to define one.

  2. FARTHEST-POINT ANCHORS SIT WHERE THE POOL IS THINNEST, by construction.
     Measured pool density around v2 anchors vs around random pool configurations:
         radius 0.15 A ->  16 vs 128   (8.0x thinner)
         radius 0.20 A -> 222 vs 996   (4.5x thinner)
         radius 0.25 A -> 1240 vs 3513 (2.8x thinner)
     So patches are hardest to build exactly where the anchors are. Fraction of v2 anchors that
     could supply >= 8 distinct neighbours: 67.2% at 0.15 A, 98.2% at 0.20 A, 100% at 0.25 A.
     Hence the default radius 0.22 A and the explicit density pre-filter below.

ANCHOR ALLOCATION -- THE ENSEMBLE-WEIGHTING KNOB
    Farthest point in PAIR-DISTANCE space spreads anchors uniformly in GEOMETRY, which favours
    sparse/transition regions; that is what produced the transition-heavy v2 set. Farthest point
    in LATENT space spreads them uniformly in PROBABILITY, because latent volume IS reference
    probability mass under the acquisition flow -- so basins receive weight in proportion to
    how much the ensemble actually contains.

    Neither extreme is right. Pure geometry under-weights the basins and breaks the ensemble;
    pure probability under-covers the genuinely under-sampled corridor that the whole DHH
    workflow exists to reach. `--latent-frac` mixes them; the diagnostics report basin balance
    for whatever is chosen so the decision can be made from data rather than taste.

USAGE
    python -m sampling.select_stage3_patches \
        --pool local_work/v2_stage2_harvest/cg_coords_all.npz \
        --bias-npz <smooth_*.npz> --flow <flow_*.npz> \
        --n-anchors 10000 --n-neighbours 8 --radius 0.22 \
        --outdir local_work/v3_stage3_selection
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _log(m: str) -> None:
    print(m, flush=True)


def farthest_point_gpu(X: np.ndarray, n: int, start: int):
    """float64 GPU farthest point -- identical contract to
    sampling/build_meanforce_campaign.py::farthest_point_gpu (see its docstring on why
    float64: float32 picks the same SET but a different ORDER, which breaks reproducibility)."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    jax.config.update("jax_enable_x64", True)
    Xj = jnp.asarray(X, jnp.float64)
    d0 = jnp.linalg.norm(Xj - Xj[start], axis=1)

    def step(d, _):
        j = jnp.argmax(d)
        return jnp.minimum(d, jnp.linalg.norm(Xj - Xj[j], axis=1)), (j, d[j])

    _, (js, seps) = lax.scan(step, d0, None, length=n - 1)
    return (np.concatenate([[start], np.asarray(js)]),
            np.concatenate([[np.inf], np.asarray(seps)]))


def patch_pick(cand_feats: np.ndarray, k: int, min_sep: float) -> np.ndarray:
    """Choose k members of a patch that SPAN it: greedy farthest point among the candidates,
    with a hard minimum separation so consecutive MD frames cannot enter as near-duplicates.

    Returns indices INTO cand_feats; fewer than k if the floor cannot be met."""
    if len(cand_feats) == 0:
        return np.empty(0, dtype=int)
    chosen = [0]                                   # candidate 0 is the anchor itself
    d = np.linalg.norm(cand_feats - cand_feats[0], axis=1)
    while len(chosen) < k + 1:
        j = int(np.argmax(d))
        if d[j] < min_sep:
            break
        chosen.append(j)
        d = np.minimum(d, np.linalg.norm(cand_feats - cand_feats[j], axis=1))
    return np.asarray(chosen[1:], dtype=int)       # drop the anchor; caller keeps it separately


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="stage-2 cg_coords_all.npz")
    ap.add_argument("--bias-npz", required=True, help="TICA projection artifact")
    ap.add_argument("--flow", type=Path, required=True, help="flow_*.npz for the latent")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--n-anchors", type=int, default=10000)
    ap.add_argument("--n-neighbours", type=int, default=8)
    ap.add_argument("--radius", type=float, default=0.22,
                    help="patch radius in PAIR-DISTANCE space, Angstrom")
    ap.add_argument("--min-sep", type=float, default=0.05,
                    help="minimum pair-distance separation inside a patch; stops consecutive "
                         "MD frames entering as near-duplicates")
    ap.add_argument("--latent-frac", type=float, default=0.5,
                    help="fraction of anchors chosen by farthest point in LATENT space "
                         "(uniform in probability); the rest use pair-distance space "
                         "(uniform in geometry). 0 = pure v2 behaviour, 1 = pure probability")
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--mapping", default="ala2_backbone_cb_6")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    from scipy.spatial import cKDTree

    from sampling.biases.tica_regional import SmoothTICABias
    from sampling.flow_density import load_flow, to_latent
    from sampling.mapping import dihedral_deg, get_mapping, wrap_deg

    rng = np.random.default_rng(a.seed)
    mapping = get_mapping(a.mapping)
    bias = SmoothTICABias.load(a.bias_npz)
    params, cfg = load_flow(a.flow)

    R = np.asarray(np.load(a.pool)["R"], np.float64)
    F = bias.projection.features(R)                                    # (N, 15) Angstrom
    Z = np.asarray(bias.projection.transform(R), np.float64)[:, :cfg.n_dims]
    U = np.asarray(to_latent(params, cfg, Z), np.float64)              # (N, d) latent
    _log(f"[pool] {len(R)} configurations, features {F.shape}, latent {U.shape}")

    # ---- density pre-filter: an anchor must be able to supply a full patch ----------------
    tree = cKDTree(F)
    counts = np.asarray(tree.query_ball_point(F, a.radius, return_length=True))
    eligible = np.flatnonzero(counts >= a.n_neighbours + 1)
    _log(f"[filter] {len(eligible)}/{len(R)} ({100*len(eligible)/len(R):.1f}%) pool configs have "
         f">= {a.n_neighbours} neighbours within {a.radius} A and can anchor a full patch")
    if len(eligible) < a.n_anchors:
        raise SystemExit(f"only {len(eligible)} eligible anchors for {a.n_anchors} requested; "
                         f"raise --radius or lower --n-neighbours")

    # ---- anchors: mix of probability-uniform (latent) and geometry-uniform (pair-distance) -
    n_lat = int(round(a.latent_frac * a.n_anchors))
    n_geo = a.n_anchors - n_lat
    _log(f"[anchors] {n_lat} by LATENT farthest point (uniform in probability) + "
         f"{n_geo} by PAIR-DISTANCE farthest point (uniform in geometry)")

    picked: list[int] = []
    if n_lat:
        idx, sep = farthest_point_gpu(U[eligible], n_lat, int(rng.integers(len(eligible))))
        picked += eligible[idx].tolist()
        _log(f"[anchors]   latent   final separation {sep[-1]:.5f}")
    if n_geo:
        remaining = np.setdiff1d(eligible, np.asarray(picked, dtype=int))
        idx, sep = farthest_point_gpu(F[remaining], n_geo, int(rng.integers(len(remaining))))
        picked += remaining[idx].tolist()
        _log(f"[anchors]   pairdist final separation {sep[-1]:.5f} A")
    anchors = np.asarray(picked, dtype=int)

    # ---- patches -------------------------------------------------------------------------
    used = np.zeros(len(R), dtype=bool)
    used[anchors] = True
    neigh_of = {}
    short = 0
    cand_all = tree.query_ball_point(F[anchors], a.radius)
    for ai, cands in zip(anchors, cand_all):
        c = np.asarray([ai] + [j for j in cands if j != ai and not used[j]], dtype=int)
        sel = patch_pick(F[c], a.n_neighbours, a.min_sep)
        chosen = c[sel]
        used[chosen] = True
        neigh_of[int(ai)] = chosen.tolist()
        if len(chosen) < a.n_neighbours:
            short += 1
    states = np.concatenate([anchors, np.concatenate([np.asarray(v, dtype=int)
                                                      for v in neigh_of.values() if len(v)])])
    _log(f"[patches] {len(anchors)} anchors, {len(states)-len(anchors)} neighbours, "
         f"{len(states)} states total; {short} anchors short of a full patch "
         f"({100*short/len(anchors):.2f}%)")

    # ---- diagnostics ---------------------------------------------------------------------
    cv = lambda R_, n: wrap_deg(dihedral_deg(R_, mapping.cvs[n].bead_indices)
                                + mapping.cvs[n].shift_deg)

    def basins(R_):
        phi, psi = cv(R_, "phi"), cv(R_, "psi")
        reg = np.full(len(R_), "other", dtype=object)
        reg[(phi > -180) & (phi < -20) & ((psi > 90) | (psi < -150))] = "beta"
        reg[(phi > -160) & (phi < -20) & (psi > -120) & (psi < 50)] = "alphaR"
        reg[(phi > 20) & (phi < 100) & (psi > -20) & (psi < 100)] = "alphaL"
        reg[(phi > -15) & (phi < 15)] = "other"
        return reg

    reg_states, reg_anchor = basins(R[states]), basins(R[anchors])
    intra = []
    for ai, nb in neigh_of.items():
        if nb:
            intra.append(np.linalg.norm(F[np.asarray(nb)] - F[ai], axis=1))
    intra = np.concatenate(intra) if intra else np.zeros(0)
    lat_intra = []
    for ai, nb in neigh_of.items():
        if nb:
            lat_intra.append(np.linalg.norm(U[np.asarray(nb)] - U[ai], axis=1))
    lat_intra = np.concatenate(lat_intra) if lat_intra else np.zeros(0)

    summary = {
        "config": vars(a) | {"outdir": str(a.outdir), "flow": str(a.flow)},
        "pool": int(len(R)), "eligible": int(len(eligible)),
        "n_anchors": int(len(anchors)), "n_states": int(len(states)),
        "anchors_short_of_full_patch": int(short),
        "patch_radius_pairdist": {
            "median": float(np.median(intra)), "p10": float(np.percentile(intra, 10)),
            "p90": float(np.percentile(intra, 90)), "min": float(intra.min())},
        "patch_radius_latent": {
            "median": float(np.median(lat_intra)), "p90": float(np.percentile(lat_intra, 90)),
            "artificial_well_width_latent": 0.525,
            "samples_across_well": float(0.525 / max(np.median(lat_intra), 1e-12))},
        "basins": {
            "anchors": {b: float(100 * (reg_anchor == b).mean())
                        for b in ("beta", "alphaR", "alphaL", "other")},
            "all_states": {b: float(100 * (reg_states == b).mean())
                           for b in ("beta", "alphaR", "alphaL", "other")},
            "reference_truth": {"beta": 65.20, "alphaR": 29.97, "alphaL": 2.79, "other": 2.04},
            "v2_42k_states": None},
        "cost_node_hours": float(len(states) * 0.882 / 3600.0),
    }
    np.savez_compressed(a.outdir / "stage3_states.npz",
                        R=R[states].astype(np.float32), pool_index=states,
                        anchor_index=anchors,
                        is_anchor=np.isin(states, anchors))
    (a.outdir / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    _log("\n=== PATCH GEOMETRY ===")
    p = summary["patch_radius_pairdist"]
    _log(f"  pair-distance anchor->neighbour: median {p['median']:.4f} A  "
         f"p10 {p['p10']:.4f}  p90 {p['p90']:.4f}  min {p['min']:.4f}")
    q = summary["patch_radius_latent"]
    _log(f"  latent anchor->neighbour:        median {q['median']:.4f}  p90 {q['p90']:.4f}")
    _log(f"  --> {q['samples_across_well']:.1f} samples across the 0.525-wide artificial well")
    _log("\n=== BASIN BALANCE (%) ===")
    _log(f"  {'':12s} {'beta':>8s} {'alphaR':>8s} {'alphaL':>8s} {'other':>8s}")
    for k in ("anchors", "all_states", "reference_truth"):
        d = summary["basins"][k]
        _log(f"  {k:12s} {d['beta']:8.2f} {d['alphaR']:8.2f} {d['alphaL']:8.2f} {d['other']:8.2f}")
    _log(f"\n  states {len(states)} -> Stage-3 cost {summary['cost_node_hours']:.1f} node-hours")
    _log(f"\nwrote {a.outdir}/stage3_states.npz and selection_summary.json")


if __name__ == "__main__":
    main()
