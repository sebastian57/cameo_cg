"""Systematic-force-bias penalty for force matching.

WHY. `dF` between basins is the line integral of the mean force, so it sees only the SYSTEMATIC
component of the force residual, `b_R = E[F_pred - F_AA]`. Ordinary force matching minimises
`E||r||^2 = ||b||^2 + E||r - b||^2`, and this project measured the second term to be ~93% of the
total (LESSONS L64). Gradient descent therefore spends nearly all its capacity on the part free
energies cannot see -- which is why a 12% RMSE improvement (mean-force-only) left `dF` unchanged
and the systematic component identical to three decimals.

WHAT. Add a term measuring the bias directly, with the mean INSIDE the norm:

    L = L_FM + lambda * SUM_bins w_bin * || E_{x in bin}[ F_pred(x) - F_AA(x) ] ||^2

projected onto the (phi, psi) directions, because that projected mean-force field is the object
whose line integral is the FES. Because the model underfits, this forces a trade: accept more
random error (invisible to dF) to reduce systematic error (all dF sees).

HOW THE ESTIMATOR PROBLEM IS SOLVED. A minibatch carries only 1-4 mean-force labels per bin,
far below the ~12 needed to resolve the bias, and a naive per-batch mean has
`E||b_hat||^2 = ||b||^2 + sigma^2/n` -- the variance term would drag the penalty back toward the
ordinary loss. This uses chemtrain's `penalty_fn(params)` hook instead, which is evaluated on a
FIXED PANEL every step, independent of the minibatch. Each bin then holds tens of labels, the
estimate is deterministic, and no EMA or variance correction is needed.

CAVEAT. Minimising `||b_hat||^2` over a fixed panel can fit the panel's own label noise
(SE ~1.0 per component, so the per-bin noise floor is ~sigma_proj/sqrt(n)). Keep `per_bin` high
enough that the true bias (measured 1.8-2.3) dominates that floor, and validate on a HELD-OUT
panel -- `build_bias_panel` returns one.
"""
from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional

import numpy as np


def bias_penalty_enabled(config) -> bool:
    return bool(config.get("training", "bias_penalty", "enabled", default=False))


def bias_penalty_config(config) -> Dict[str, Any]:
    cfg = config.get("training", "bias_penalty", default={}) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "lambda": float(cfg.get("lambda", 1.0)),
        "label_paths": list(cfg.get("label_paths", [])),
        "bin_deg": float(cfg.get("bin_deg", 30.0)),
        "per_bin": int(cfg.get("per_bin", 40)),
        "min_per_bin": int(cfg.get("min_per_bin", 12)),
        "holdout_frac": float(cfg.get("holdout_frac", 0.25)),
        "seed": int(cfg.get("seed", 20260904)),
        "mapping": str(cfg.get("mapping", "ala2_backbone_cb_6")),
        "undisplaced_only": bool(cfg.get("undisplaced_only", False)),
        "chunk": int(cfg.get("chunk", 256)),
    }


class BiasPanel(NamedTuple):
    R: np.ndarray            # (n, beads, 3)
    F: np.ndarray            # (n, beads, 3)  conditional MEAN force labels
    mask: np.ndarray         # (n, beads)
    species: np.ndarray      # (n, beads)
    proj: np.ndarray         # (n, 2, beads, 3) = (J J^T)^-1 J, the generalised-force operator
    jac: np.ndarray          # (n, 2, beads, 3) raw d(phi,psi)/dR, kept for validation
    bin_id: np.ndarray       # (n,) contiguous 0..n_bins-1
    bin_weight: np.ndarray   # (n_bins,) normalised to sum 1
    n_bins: int
    counts: np.ndarray       # (n_bins,) labels per bin


def _generalised_force_operator(R: np.ndarray, mapping_name: str):
    """P = (J J^T)^-1 J, the operator giving the GENERALISED forces conjugate to (phi, psi).

    Not the two unit gradients: phi and psi share atoms, so d(phi)/dR and d(psi)/dR are NOT
    orthogonal (measured overlap ~0.45 here). Summing the two raw projections double-counts the
    shared component by a factor 1 + cos^2(theta) = 1.20. The proper 2x2 metric removes it -- the
    same correction `analysis/physics/meanforce_fes.py` applies for exactly this reason.

    Returns (phi, psi, P) with P shaped (n, 2, beads, 3); the analytic jax Jacobian is reused
    because `sampling.mapping.evaluate` is numpy and not differentiable.
    """
    from analysis.md.projected_force_analysis import (
        DihedralCVSpec, ramachandran_values_and_jacobians)
    from sampling.mapping import get_mapping
    m = get_mapping(mapping_name)
    specs = [DihedralCVSpec(indices=tuple(m.cvs[k].bead_indices), shift_deg=m.cvs[k].shift_deg)
             for k in ("phi", "psi")]
    vals, J = ramachandran_values_and_jacobians(np.asarray(R, np.float64), specs)
    n, _, nb, _ = J.shape
    Jf = J.reshape(n, 2, nb * 3)                       # (n, 2, 18)
    G = np.einsum("nai,nbi->nab", Jf, Jf)              # (n, 2, 2) metric J J^T
    G += 1e-12 * np.eye(2)[None]
    P = np.einsum("nab,nbi->nai", np.linalg.inv(G), Jf)
    return vals[:, 0], vals[:, 1], P.reshape(n, 2, nb, 3), J


def build_bias_panel(label_paths, *, bin_deg=30.0, per_bin=40, min_per_bin=12,
                     holdout_frac=0.25, seed=20260904, mapping="ala2_backbone_cb_6",
                     undisplaced_only=False, state_files=None):
    """Stratified panel of conditional mean-force labels, plus a disjoint held-out panel.

    Bins with fewer than `min_per_bin` labels are DROPPED: their bias estimate would be noise,
    and including them would let the penalty chase it.
    """
    Rs, Fs = [], []
    for i, p in enumerate(label_paths):
        d = np.load(p)
        keep = np.ones(len(d["R"]), bool)
        if undisplaced_only and state_files and state_files[i]:
            eps = np.load(state_files[i])["eps_state"][d["state"].astype(int)]
            keep = eps == 0
        Rs.append(np.asarray(d["R"], np.float64)[keep])
        Fs.append(np.asarray(d["F"], np.float64)[keep])
    R = np.concatenate(Rs); F = np.concatenate(Fs)

    phi, psi, P, J = _generalised_force_operator(R, mapping)
    nb = int(round(360.0 / bin_deg))
    ed = np.linspace(-180.0, 180.0, nb + 1)
    ix = np.clip(np.digitize(phi, ed) - 1, 0, nb - 1)
    iy = np.clip(np.digitize(psi, ed) - 1, 0, nb - 1)
    key = ix * nb + iy

    rng = np.random.default_rng(seed)
    tr_idx, ho_idx, bin_of = [], [], {}
    for k in np.unique(key):
        member = np.flatnonzero(key == k)
        if len(member) < min_per_bin:
            continue
        rng.shuffle(member)
        n_ho = int(round(len(member) * holdout_frac))
        ho = member[:n_ho]
        tr = member[n_ho:][:per_bin]
        if len(tr) < min_per_bin:
            continue
        bin_of[k] = len(bin_of)
        tr_idx.append(tr); ho_idx.append(ho[:per_bin])
    if not tr_idx:
        raise ValueError("bias panel is empty: no Ramachandran bin met min_per_bin")

    def _pack(idx_list):
        idx = np.concatenate(idx_list)
        bid = np.concatenate([np.full(len(v), i, np.int32) for i, v in enumerate(idx_list)])
        counts = np.array([len(v) for v in idx_list], np.int64)
        n_beads = R.shape[1]
        return BiasPanel(
            R=R[idx].astype(np.float32), F=F[idx].astype(np.float32),
            mask=np.ones((len(idx), n_beads), np.float32),
            species=np.tile(np.arange(n_beads, dtype=np.int32), (len(idx), 1)),
            proj=P[idx].astype(np.float32), jac=J[idx].astype(np.float32),
            bin_id=bid, bin_weight=(counts / counts.sum()).astype(np.float32),
            n_bins=len(idx_list), counts=counts)
    return _pack(tr_idx), _pack(ho_idx)


def make_bias_penalty(*, force_of, panel: BiasPanel, lam: float, neighbors=None,
                      chunk: int = 256):
    """penalty_fn(params) -> lambda * SUM_bins w_bin * ||E_bin[generalised force error]||^2.

    `force_of(params, R, mask, species, neighbor)` returns predicted forces for ONE structure.

    MEMORY. Evaluating all ~2k panel structures in one vmap materialises their Allegro
    forward+backward activations alongside the training batch inside the same jitted update and
    OOMed at a 97.12 GiB allocation on a 96 GB GH200 (job 1666455). The panel is therefore split
    into slices, each wrapped in `jax.checkpoint` so its activations are recomputed in the
    backward pass instead of being held.

    NO `lax.scan`. chemtrain runs the update under `shard_map` with MANUAL mesh axes, and a scan
    carry created with `jnp.zeros` there picks up an Auto sharding -- "Context mesh ... Manual
    should match the mesh of sharding ... Auto passed to broadcast_in_dim" (job 1672066). An
    unrolled Python loop accumulating with `+` creates no such array. Only the per-structure
    generalised force errors (n, 2) are concatenated, so the reduction stays tiny.
    """
    import jax
    import jax.numpy as jnp

    n = len(panel.R)
    n_bins = int(panel.n_bins)
    lam = float(lam)
    R = jnp.asarray(panel.R); F = jnp.asarray(panel.F)
    mask = jnp.asarray(panel.mask); species = jnp.asarray(panel.species)
    proj = jnp.asarray(panel.proj)
    bin_id = jnp.asarray(panel.bin_id)
    w = jnp.asarray(panel.bin_weight)
    counts = jnp.asarray(panel.counts, jnp.float32)
    nbrs = None if neighbors is None else jax.tree.map(jnp.asarray, neighbors)
    bounds = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]

    def penalty(params):
        parts = []
        for a, b in bounds:
            def one(p, a=a, b=b):
                if nbrs is None:
                    Fp = jax.vmap(lambda r, m, s: force_of(p, r, m, s, None))(
                        R[a:b], mask[a:b], species[a:b])
                else:
                    nb = jax.tree.map(lambda x: x[a:b], nbrs)
                    Fp = jax.vmap(lambda r, m, s, q: force_of(p, r, m, s, q))(
                        R[a:b], mask[a:b], species[a:b], nb)
                res = (Fp - F[a:b]) * mask[a:b][..., None]
                return jnp.einsum("nabc,nbc->na", proj[a:b], res)
            parts.append(jax.checkpoint(one)(params))
        gf = jnp.concatenate(parts, axis=0)                       # (n, 2), small
        bias = jax.ops.segment_sum(gf, bin_id, num_segments=n_bins) / counts[:, None]
        return lam * jnp.sum(w * jnp.sum(bias ** 2, axis=1))

    return penalty


class TiledPanel(NamedTuple):
    R: np.ndarray            # (n_tiles, beads_per_tile, 3)
    F: np.ndarray            # (n_tiles, beads_per_tile, 3)
    mask: np.ndarray         # (n_tiles, beads_per_tile)
    species: np.ndarray      # (n_tiles, beads_per_tile)
    segment_id: np.ndarray   # (n_tiles, beads_per_tile) structure index WITHIN the tile
    proj: np.ndarray         # (n_tiles, beads_per_tile, 2, 3) per-bead projection rows
    edges: np.ndarray        # (n_tiles, 2, E) block-diagonal Sparse layout
    seg_bin: np.ndarray      # (n_tiles, segs_per_tile) bin id per segment, n_bins for padding
    n_bins: int
    counts: np.ndarray
    bin_weight: np.ndarray


def tile_panel(panel: BiasPanel, *, target_beads: int = 1024, gap: float = 30.0) -> TiledPanel:
    """Pack the panel into tiles the way the training loader packs the dataset.

    Untiled, the panel costs ~0.60 ms per structure against the training batch's 0.15 ms -- 4x
    worse purely from evaluating 6-bead systems one at a time (measured: 149.8 s/epoch vs a
    52.96 s baseline, job 1701281). Packing ~170 structures per 1024-bead tile recovers that.

    Connectivity is an explicit BLOCK-DIAGONAL edge list, so structures in a tile cannot
    interact regardless of geometry -- `static_neighbor_list` marks the graph as externally
    supplied and the model consumes it verbatim. Structures are additionally offset onto a cubic
    grid with `gap` spacing so that any distance-based code path also sees them as separate.
    """
    n, nb = panel.R.shape[0], panel.R.shape[1]
    per_tile = target_beads // nb                       # 170 for 1024 beads / 6
    n_tiles = int(np.ceil(n / per_tile))
    B = per_tile * nb
    side = int(np.ceil(per_tile ** (1.0 / 3.0)))
    off = np.array([[i // (side * side), (i // side) % side, i % side]
                    for i in range(per_tile)], np.float64) * gap

    R = np.zeros((n_tiles, B, 3), np.float32); F = np.zeros_like(R)
    mask = np.zeros((n_tiles, B), np.float32)
    species = np.zeros((n_tiles, B), np.int32)
    seg = np.full((n_tiles, B), per_tile, np.int32)      # padding -> its own segment
    proj = np.zeros((n_tiles, B, 2, 3), np.float32)
    seg_bin = np.full((n_tiles, per_tile), panel.n_bins, np.int32)
    for k in range(n):
        tl, sl = divmod(k, per_tile)
        a = sl * nb
        R[tl, a:a+nb] = panel.R[k] - panel.R[k].mean(0) + off[sl]
        F[tl, a:a+nb] = panel.F[k]
        mask[tl, a:a+nb] = panel.mask[k]
        species[tl, a:a+nb] = panel.species[k]
        seg[tl, a:a+nb] = sl
        proj[tl, a:a+nb] = np.transpose(panel.proj[k], (1, 0, 2))    # (beads, 2, 3)
        seg_bin[tl, sl] = panel.bin_id[k]
    rec, sen = zip(*[(s * nb + i, s * nb + j)
                     for s in range(per_tile) for i in range(nb) for j in range(nb) if i != j])
    edges = np.tile(np.array([rec, sen], np.int32)[None], (n_tiles, 1, 1))
    return TiledPanel(R=R, F=F, mask=mask, species=species, segment_id=seg, proj=proj,
                      edges=edges, seg_bin=seg_bin, n_bins=panel.n_bins,
                      counts=panel.counts, bin_weight=panel.bin_weight)


def make_tiled_bias_penalty(*, energy_of, tiled: TiledPanel, lam: float, neighbors,
                            axis_name: str = None, n_devices: int = 1):
    """penalty_fn(params) over a TILED panel, optionally SHARDED across devices.

    chemtrain evaluates `penalty_fn(params)` inside `shard_map` (max_likelihood.py:118, wrapped at
    :559) and reduces the gradient with `lax.pmean(grad, 'batch')` (:585). The penalty depends only
    on `params`, which are REPLICATED, so by default every device evaluates the whole panel and
    pmean averages four identical numbers -- correct, but 4x redundant. Measured: a panel tile cost
    3.05x a training tile (77.9 vs 25.6 ms), which is that redundancy.

    With `axis_name` set, each device evaluates only its own slice of tiles and the per-bin SUMS
    are combined with `lax.psum` BEFORE the square. This ordering is required: ||mean||^2 does not
    decompose across shards, so squaring per-device partials would compute a different quantity.
    """
    import jax
    import jax.numpy as jnp

    n_tiles = tiled.R.shape[0]
    # n_devices MUST be the size of the shard_map axis (jax.device_count()), not the local count:
    # dynamic_slice clamps out-of-range starts, so an undercount silently double-counts tiles.
    per_dev = int(np.ceil(n_tiles / max(n_devices, 1)))
    padded = per_dev * max(n_devices, 1)
    def _pad_tiles(a):
        if padded == n_tiles:
            return a
        return np.concatenate([a, np.zeros((padded - n_tiles,) + a.shape[1:], a.dtype)])
    # padded tiles carry mask 0 and segment ids pointing at the dump bin, so contribute nothing
    R = jnp.asarray(_pad_tiles(tiled.R)); F = jnp.asarray(_pad_tiles(tiled.F))
    mask = jnp.asarray(_pad_tiles(tiled.mask))
    species = jnp.asarray(_pad_tiles(tiled.species))
    seg = jnp.asarray(_pad_tiles(tiled.segment_id)); proj = jnp.asarray(_pad_tiles(tiled.proj))
    seg_bin = jnp.asarray(_pad_tiles(
        np.where(np.arange(n_tiles)[:, None] < n_tiles, tiled.seg_bin, tiled.n_bins)))
    if padded > n_tiles:
        seg_bin = seg_bin.at[n_tiles:].set(tiled.n_bins)
    nbrs = jax.tree.map(
        lambda a: jnp.concatenate([a, jnp.repeat(a[:1], padded - n_tiles, axis=0)])
        if padded > n_tiles else a, neighbors)

    n_bins = int(tiled.n_bins)
    segs = tiled.seg_bin.shape[1]
    counts = jnp.asarray(tiled.counts, jnp.float32)
    w = jnp.asarray(tiled.bin_weight)
    lam = float(lam)

    @jax.checkpoint
    def _tile_gf(p, r, f, m, s, sg, pj, nb):
        # remat: recompute this tile's activations in the backward pass instead of holding them.
        # The tiled rewrite originally dropped the `jax.checkpoint` the chunked version had, and
        # the production run OOMed at 90.46 GiB while a 2-epoch smoke with the same config
        # happened to fit (jobs 1702784 vs 1702013) -- i.e. it was sitting right on the limit.
        Fp = -jax.grad(lambda x: energy_of(p, x, m, s, neighbor=nb, segment_id=sg))(r)
        res = (Fp - f) * m[:, None]
        contrib = jnp.einsum("bac,bc->ba", pj, res)
        return jax.ops.segment_sum(contrib, sg, num_segments=segs + 1)[:segs]

    def penalty(params):
        if axis_name is None:
            sel = (R, F, mask, species, seg, proj, seg_bin, nbrs)
        else:
            d = jax.lax.axis_index(axis_name)
            take = lambda a: jax.lax.dynamic_slice_in_dim(a, d * per_dev, per_dev, axis=0)
            sel = (take(R), take(F), take(mask), take(species), take(seg), take(proj),
                   take(seg_bin), jax.tree.map(take, nbrs))
        Rs, Fs, ms, ss, sgs, pjs, sbs, nbs = sel
        gf = jax.vmap(_tile_gf, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))(
            params, Rs, Fs, ms, ss, sgs, pjs, nbs)
        tot = jax.ops.segment_sum(gf.reshape(-1, 2), sbs.reshape(-1),
                                  num_segments=n_bins + 1)[:n_bins]
        if axis_name is not None:
            tot = jax.lax.psum(tot, axis_name)      # sums BEFORE the square
        bias = tot / counts[:, None]
        return lam * jnp.sum(w * jnp.sum(bias ** 2, axis=1))

    return penalty
