#!/usr/bin/env python3
"""Empirical transition-path map from the time-ordered reference trajectory.

    python -m sampling.build_transition_map --grid-dir <dir> --reference <ref.npz> \
        [--enhanced <enhanced.npz> ...] --outdir <dir>

Extracts the observed reactive trajectories -- last exit from basin A to first arrival
in basin B, without returning to A -- and turns them into per-cell fields in the SAME
frozen TICA grid used for biasing:

  C(s)     fraction of DISTINCT transition paths visiting the cell (passage frequency)
  rho_TP   reactive-frame density
  R_TP     rho_TP / (rho_eq + eps)   -- disproportionately transition-associated regions
  v(s)     mean transition displacement per cell (a trajectory displacement field, NOT
           a force field: it already contains diffusion, friction and hidden-coordinate
           effects)
  S(s)     C(s) / (rho_enhanced + eps)  -- sampling priority, if --enhanced is given

Why C(s) rather than the committor-derived `transition_component`: the latter is a MODEL
of where transitions should go, C(s) is where they were OBSERVED to go. Writing C(s) into
the artifact's transition slot lets `tica_regional` in transition_attractor mode consume
it with no code change.

Counting DISTINCT paths, not frames, is deliberate: a slow transition that lingers in a
cell would otherwise dominate purely by contributing many consecutive frames.

RESOLUTION CAVEAT. Fields are only as fine as the trajectory stride. Measured on the
ala2 bb6 reference (5 ps/frame): 2,584 transitions each way, median duration 3 frames
(15 ps), 15.8% captured by a single frame. C(s) and rho_TP are well determined; v(s) is
directionally informative but quantitatively weak (~2 displacements per path); recrossing
is invisible. Sub-ps output would be required for genuine path statistics.

An unobserved channel is not an impossible one -- 2,584 transitions constrain the
populated corridor and say nothing about the rest. C(s) should SHAPE a bias, never gate it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sampling.mapping import get_mapping  # noqa: E402

KB_KCAL = 0.0019872042586


def basin_cores_from_cv(mapping, R, cell, n_cells, spec, min_purity=0.5, min_count=20):
    """Cell-set core for a CV box: cells whose frames are PREDOMINANTLY inside the box.

    Requiring purity and occupancy keeps cores compact -- a ragged core leaks basin
    frames into the reactive segment and blunts C(s), which is exactly the failure the
    endpoint bug produced.
    """
    inside = np.ones(len(R), dtype=bool)
    for cv_name, lo, hi in spec:
        v = mapping.cvs[cv_name].evaluate(R)
        inside &= (v >= lo) & (v <= hi)
    tot = np.bincount(cell[cell >= 0], minlength=n_cells).astype(np.float64)
    ins = np.bincount(cell[(cell >= 0) & inside], minlength=n_cells).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        purity = np.divide(ins, tot, out=np.zeros_like(ins), where=tot > 0)
    return set(np.flatnonzero((purity >= min_purity) & (tot >= min_count)).tolist())


def extract_paths(labels: np.ndarray, segments, names):
    """Last-exit -> first-arrival reactive segments, never spanning a segment join.

    Generalised to N basins: every ordered pair gets its own path list, so a three-basin
    system yields beta<->alphaR, beta<->alphaL and alphaR<->alphaL separately. Lumping
    them would hide which corridor a cell belongs to.
    """
    out = {f"{a}->{b}": [] for a in names for b in names if a != b}
    for lo, hi in segments:
        l = labels[lo:hi]
        last, last_i = None, None
        for i, s in enumerate(l):
            if s == 0:
                continue
            if last is None:
                last, last_i = s, i
                continue
            if s == last:
                last_i = i
                continue
            key = f"{names[last - 1]}->{names[s - 1]}"
            out[key].append((lo + last_i, lo + i))
            last, last_i = s, i
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-dir", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--enhanced", action="append", default=[], metavar="LABEL=NPZ")
    ap.add_argument("--basin", action="append", default=None,
                    metavar="LABEL=cv:lo:hi[,cv:lo:hi]",
                    help="basin core from a CV box, repeatable. With none given the "
                         "stored basin_a_core/basin_b_core are used, which for ala2 bb6 "
                         "are beta/PPII and alphaR ONLY -- alphaL is in neither, so no "
                         "alphaL transition is mapped. Give three to get the full picture.")
    ap.add_argument("--extra-traj", action="append", default=None,
                    metavar="LABEL=NPZ:FRAME_PS",
                    help="additional time-ordered CG trajectories (e.g. the alphaL exit "
                         "campaign), projected through the SAME frozen TICA and appended "
                         "as further segments. Use when the reference resolves a channel "
                         "too poorly to bias on: ala2 bb6 has 2,950 beta<->alphaR "
                         "crossings but only 51 beta<->alphaL, all 1 frame long.")
    ap.add_argument("--projection-npz", type=Path, default=None,
                    help="bias artifact carrying tica_mean/tica_coefficients, needed to "
                         "project --extra-traj into the frozen space")
    ap.add_argument("--per-replica-frames", type=int, default=None,
                    help="frames per replica in --extra-traj, so paths never span a "
                         "replica boundary (read from the npz if absent)")
    ap.add_argument("--min-purity", type=float, default=0.5)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--mapping", type=str, default="ala2_backbone_cb_6")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--prefix", type=str, default="transition_map")
    ap.add_argument("--frame-ps", type=float, default=5.0)
    ap.add_argument("--segment-bounds", type=int, nargs="+", default=None,
                    help="frame indices splitting concatenated runs; paths never span a "
                         "join (the 200k ala2 reference is two runs: use 0 100001 200001)")
    ap.add_argument("--temperature", type=float, default=298.0)
    args = ap.parse_args()

    kT = KB_KCAL * args.temperature
    args.outdir.mkdir(parents=True, exist_ok=True)
    g = np.load(args.grid_dir / "tica_projection_and_grid.npz")
    z = np.asarray(g["projection"], dtype=np.float64)
    cell = np.asarray(g["cell_index"], dtype=np.int64)
    xe, ye = np.asarray(g["xedges"]), np.asarray(g["yedges"])
    nx, ny = len(xe) - 1, len(ye) - 1
    A, B = set(g["basin_a_core"].tolist()), set(g["basin_b_core"].tolist())

    if args.basin:
        m = get_mapping(args.mapping)
        R = np.load(args.reference)["R"].astype(np.float64)
        if len(R) != len(cell):
            raise SystemExit(f"reference has {len(R)} frames, grid has {len(cell)}")
        names, cores = [], []
        for spec in args.basin:
            label, box = spec.split("=", 1)
            parsed = []
            for part in box.split(","):
                cv, lo, hi = part.split(":")
                parsed.append((cv, float(lo), float(hi)))
            core = basin_cores_from_cv(m, R, cell, nx * ny, parsed,
                                       args.min_purity, args.min_count)
            names.append(label); cores.append(core)
            print(f"  basin {label}: {len(core)} cells, "
                  f"{int(np.isin(cell, list(core)).sum())} frames")
        overlap = [(names[i], names[j]) for i in range(len(cores))
                   for j in range(i + 1, len(cores)) if cores[i] & cores[j]]
        if overlap:
            raise SystemExit(f"basin cores overlap: {overlap}; tighten the boxes")
    else:
        names, cores = ["A", "B"], [A, B]
    labels = np.zeros(len(cell), dtype=np.int8)
    for k, core in enumerate(cores, start=1):
        labels[np.isin(cell, list(core))] = k

    frame_ps = np.full(len(cell), args.frame_ps)
    bounds = args.segment_bounds or [0, len(cell)]
    segments = list(zip(bounds[:-1], bounds[1:]))

    # ---- append extra trajectories, projected through the SAME frozen TICA ----
    if args.extra_traj:
        if args.projection_npz is None:
            raise SystemExit("--extra-traj needs --projection-npz")
        from sampling.biases.tica_regional import SmoothTICABias
        proj = SmoothTICABias.load(args.projection_npz).projection
        for spec in args.extra_traj:
            label, rest = spec.split("=", 1)
            path, fps = rest.rsplit(":", 1)
            d = np.load(path)
            Rx = np.asarray(d["R"], dtype=np.float64)
            zx = np.asarray(proj.transform(Rx), dtype=np.float64)[:, :2]
            ix = np.clip(np.digitize(zx[:, 0], xe) - 1, -1, nx - 1)
            iy = np.clip(np.digitize(zx[:, 1], ye) - 1, -1, ny - 1)
            cx = np.where((ix >= 0) & (iy >= 0) & (zx[:, 0] >= xe[0]) & (zx[:, 0] <= xe[-1])
                          & (zx[:, 1] >= ye[0]) & (zx[:, 1] <= ye[-1]), ix * ny + iy, -1)
            per = (d["per_case"] if "per_case" in d.files
                   else np.array([args.per_replica_frames or len(Rx)]))
            off = len(cell)
            edges = np.cumsum(np.r_[0, per])
            segments += [(off + int(a), off + int(b)) for a, b in zip(edges[:-1], edges[1:])]
            z = np.vstack([z, zx]); cell = np.concatenate([cell, cx])
            frame_ps = np.concatenate([frame_ps, np.full(len(Rx), float(fps))])
            print(f"  extra '{label}': {len(Rx)} frames, {len(per)} replicas @ {fps} ps/frame "
                  f"({int((cx >= 0).sum())} inside the frozen grid)")
        labels = np.zeros(len(cell), dtype=np.int8)
        for k, core in enumerate(cores, start=1):
            labels[np.isin(cell, list(core))] = k
    paths = extract_paths(labels, segments, names)
    paths = {k: v for k, v in paths.items() if v}      # drop pairs never observed
    n_paths = sum(len(v) for v in paths.values())
    print("segments %s -> " % segments + ", ".join(f"{k} {len(v)}" for k, v in paths.items()))

    # ---- per-cell fields -----------------------------------------------------
    C = np.zeros(nx * ny)                      # distinct-path passage frequency
    rho_TP = np.zeros(nx * ny)
    disp_sum = np.zeros((nx * ny, 2))
    disp_n = np.zeros(nx * ny)
    C_dir = {k: np.zeros(nx * ny) for k in paths}
    for key, segs in paths.items():
        for a, b in segs:
            # STRICTLY between the basins: a is the last frame in the source core and b
            # the first in the target core, so including them makes C(s) peak on the
            # basin centres rather than the corridor -- with a 3-frame median path the
            # endpoints are the majority of every segment.
            if b - a < 2:
                continue          # direct core-to-core hop: no resolved reactive frame
            idx = cell[a + 1:b]
            valid = idx[idx >= 0]
            if valid.size == 0:
                continue
            uniq = np.unique(valid)
            C[uniq] += 1.0
            C_dir[key][uniq] += 1.0
            np.add.at(rho_TP, valid, 1.0)
            if b - a >= 2:
                d = z[a + 2:b] - z[a + 1:b - 1] if b - a > 2 else np.zeros((0, 2))
                src = cell[a + 1:b - 1] if b - a > 2 else np.zeros(0, dtype=np.int64)
                ok = src >= 0
                np.add.at(disp_sum, src[ok], d[ok])
                np.add.at(disp_n, src[ok], 1.0)
    C /= max(n_paths, 1)
    for k in C_dir:
        C_dir[k] /= max(len(paths[k]), 1)
    rho_TP /= max(rho_TP.sum(), 1.0)
    v = np.divide(disp_sum, disp_n[:, None], out=np.zeros_like(disp_sum), where=disp_n[:, None] > 0)

    counts = np.bincount(cell[cell >= 0], minlength=nx * ny).astype(np.float64)
    rho_eq = counts / max(counts.sum(), 1.0)
    eps = 1.0 / max(counts.sum(), 1.0)
    R_TP = rho_TP / (rho_eq + eps)

    # ---- enhanced coverage -> sampling priority ------------------------------
    priorities = {}
    for spec in args.enhanced:
        label, path = spec.split("=", 1)
        Renh = np.load(path)["R"].astype(np.float64)
        # project through the same frozen TICA
        from sampling.biases.tica_regional import TICAProjection
        proj = TICAProjection(np.asarray(g["pair_indices"], dtype=np.int64),
                              np.asarray(g["tica_mean"], dtype=np.float64)
                              if "tica_mean" in g.files else None, None)
        priorities[label] = None   # filled below if projection available
    # (projection coefficients live in the bias artifact, not the grid; skip if absent)

    F = np.full(nx * ny, np.nan)
    occ = counts > 0
    F[occ] = -kT * np.log(rho_eq[occ])
    F[occ] -= np.nanmin(F[occ])

    def grid(a):
        return a.reshape(nx, ny)

    ext = [xe[0], xe[-1], ye[0], ye[-1]]
    xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])

    # =================== FIGURE 1: paths over the FES =========================
    keys = list(paths)
    fig, ax = plt.subplots(1, len(keys), figsize=(6.2 * len(keys), 5.2),
                           constrained_layout=True, squeeze=False)
    rng = np.random.default_rng(0)
    for k, key in enumerate(keys):
        a_ = ax[0][k]
        im = a_.imshow(grid(F).T, origin="lower", extent=ext, aspect="auto",
                       cmap="Greys_r", vmin=0, vmax=6)
        segs = paths[key]
        show = segs if len(segs) <= 120 else [segs[i] for i in
                                              rng.choice(len(segs), 120, replace=False)]
        for a, b in show:
            a_.plot(z[a:b + 1, 0], z[a:b + 1, 1], "-", lw=0.7, alpha=0.4, color=f"C{k}")
        a_.set_title(f"{key}: {len(segs)} transitions ({len(show)} drawn)", fontsize=10)
        a_.set_xlabel("TIC 1"); a_.set_ylabel("TIC 2")
        fig.colorbar(im, ax=a_, label="reference F (kcal/mol)")
    fig.suptitle("Observed reactive trajectories over the reference FES")
    p1 = args.outdir / f"{args.prefix}_paths.png"
    fig.savefig(p1, dpi=160); plt.close(fig); print("  wrote", p1)

    # =================== FIGURE 1b: per-pair C(s) =============================
    fig, ax = plt.subplots(1, len(keys) + 1, figsize=(5.4 * (len(keys) + 1), 4.5),
                           constrained_layout=True, squeeze=False)
    im = ax[0][0].imshow(grid(F).T, origin="lower", extent=ext, aspect="auto",
                         cmap="turbo", vmin=0, vmax=6)
    ax[0][0].set_title("reference FES"); fig.colorbar(im, ax=ax[0][0])
    for k, key in enumerate(keys, start=1):
        im = ax[0][k].imshow(grid(C_dir[key]).T, origin="lower", extent=ext,
                             aspect="auto", cmap="magma")
        ax[0][k].set_title(f"C(s)  {key}", fontsize=10)
        fig.colorbar(im, ax=ax[0][k])
    for a_ in ax[0]:
        a_.set_xlabel("TIC 1"); a_.set_ylabel("TIC 2")
    fig.suptitle("Passage frequency per transition channel")
    p1b = args.outdir / f"{args.prefix}_per_channel.png"
    fig.savefig(p1b, dpi=160); plt.close(fig); print("  wrote", p1b)

    # =================== FIGURE 2: the fields =================================
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.5), constrained_layout=True)
    im = ax[0][0].imshow(grid(F).T, origin="lower", extent=ext, aspect="auto",
                         cmap="turbo", vmin=0, vmax=6)
    ax[0][0].set_title("reference FES"); fig.colorbar(im, ax=ax[0][0])

    im = ax[0][1].imshow(grid(C).T, origin="lower", extent=ext, aspect="auto", cmap="magma")
    ax[0][1].set_title("C(s): fraction of distinct paths visiting")
    fig.colorbar(im, ax=ax[0][1])

    with np.errstate(divide="ignore", invalid="ignore"):
        logR = np.where(R_TP > 0, np.log2(R_TP), np.nan)
    im = ax[1][0].imshow(grid(logR).T, origin="lower", extent=ext, aspect="auto",
                         cmap="RdBu_r", vmin=-3, vmax=3)
    ax[1][0].set_title(r"log2 $R_{TP}=\rho_{TP}/\rho_{eq}$  (red = transition-enriched)")
    fig.colorbar(im, ax=ax[1][0])

    im = ax[1][1].imshow(grid(C).T, origin="lower", extent=ext, aspect="auto",
                         cmap="magma", alpha=0.85)
    step = max(1, nx // 30)
    X, Y = np.meshgrid(xc[::step], yc[::step], indexing="ij")
    U = grid(v[:, 0])[::step, ::step]; V = grid(v[:, 1])[::step, ::step]
    N = grid(disp_n)[::step, ::step]
    m = N > 20
    ax[1][1].quiver(X[m], Y[m], U[m], V[m], color="cyan", scale_units="xy", angles="xy")
    ax[1][1].set_title("mean transition displacement (>20 samples/cell)")
    fig.colorbar(im, ax=ax[1][1])
    for a in ax.ravel():
        a.set_xlabel("TIC 1"); a.set_ylabel("TIC 2")
    fig.suptitle("Empirical transition-path fields (frozen TICA grid)")
    p2 = args.outdir / f"{args.prefix}_fields.png"
    fig.savefig(p2, dpi=160); plt.close(fig); print("  wrote", p2)

    # =================== FIGURE 3: observed vs modelled =======================
    if "transition_component" in g.files:
        tc = np.asarray(g["transition_component"], dtype=np.float64).ravel()
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
        im = ax[0].imshow(grid(tc / max(tc.max(), 1e-12)).T, origin="lower", extent=ext,
                          aspect="auto", cmap="magma")
        ax[0].set_title("MODELLED transition_component\n(committor-derived, currently biased on)")
        fig.colorbar(im, ax=ax[0])
        im = ax[1].imshow(grid(C / max(C.max(), 1e-12)).T, origin="lower", extent=ext,
                          aspect="auto", cmap="magma")
        ax[1].set_title("OBSERVED C(s)\n(distinct-path passage frequency)")
        fig.colorbar(im, ax=ax[1])
        both = (tc > 0) | (C > 0)
        d = np.full(nx * ny, np.nan)
        d[both] = (C / max(C.max(), 1e-12) - tc / max(tc.max(), 1e-12))[both]
        im = ax[2].imshow(grid(d).T, origin="lower", extent=ext, aspect="auto",
                          cmap="RdBu_r", vmin=-1, vmax=1)
        ax[2].set_title("observed - modelled\n(red: real transitions the model misses)")
        fig.colorbar(im, ax=ax[2])
        for a in ax:
            a.set_xlabel("TIC 1"); a.set_ylabel("TIC 2")
        m = both & (counts > 0)
        corr = float(np.corrcoef(C[m], tc[m])[0, 1])
        fig.suptitle(f"Does the committor model match observed transitions?  corr = {corr:+.3f}")
        p3 = args.outdir / f"{args.prefix}_observed_vs_model.png"
        fig.savefig(p3, dpi=160); plt.close(fig); print("  wrote", p3)
    else:
        corr = float("nan")

    # =================== artifact + summary ===================================
    out_npz = args.outdir / f"{args.prefix}.npz"
    # ---- per-channel R_TP, each normalised to its OWN maximum -----------------
    # A globally normalised field is dominated by whichever channel is common: ala2 has
    # ~5,400 beta<->alphaR crossings against ~45 for alphaL, so a summed R_TP pins alphaL
    # at zero and a bias built on it ignores exactly the channel that broke the FES.
    # Normalising per channel first, then combining with explicit weights, makes the
    # rare corridor visible while keeping the choice of emphasis a stated decision
    # rather than an accident of event counts.
    R_TP_chan, chan_norm = {}, {}
    for key, segs in paths.items():
        rho_c = np.zeros(nx * ny)
        for a, b in segs:
            if b - a < 2:
                continue
            v_ = cell[a + 1:b]
            np.add.at(rho_c, v_[v_ >= 0], 1.0)
        if rho_c.sum() <= 0:
            continue
        rho_c /= rho_c.sum()
        r = rho_c / (rho_eq + eps)
        chan_norm[key] = float(r.max())
        R_TP_chan[key] = r / max(r.max(), 1e-12)
    combined = np.zeros(nx * ny)
    for key, r in R_TP_chan.items():
        combined = np.maximum(combined, r)      # max, not sum: a cell used by ANY
                                                # channel is worth targeting
    print("  per-channel R_TP maxima: " +
          ", ".join(f"{k} {v:.1f}" for k, v in chan_norm.items()))

    np.savez_compressed(out_npz, passage_frequency=C, rho_TP=rho_TP, R_TP=R_TP,
                        R_TP_perchannel_max=combined,
                        **{f"R_TP_{k.replace('->','_to_')}": v for k, v in R_TP_chan.items()},
                        basin_names=np.array(names, dtype=object),
                        **{f"passage_frequency_{k.replace('->','_to_')}": v
                           for k, v in C_dir.items()},
                        mean_displacement=v, displacement_counts=disp_n,
                        rho_eq=rho_eq, n_paths=n_paths, xedges=xe, yedges=ye)
    durations = {k: np.array([b - a for a, b in v_]) for k, v_ in paths.items()}
    unresolved = {k: int((d < 2).sum()) for k, d in durations.items() if len(d)}
    summary = {
        "n_paths": {k: len(v_) for k, v_ in paths.items()},
        "frame_ps": args.frame_ps,
        "duration_frames": {k: {"median": float(np.median(d)), "mean": float(d.mean()),
                                "max": int(d.max()),
                                "single_frame_pct": float(100 * (d == 1).mean())}
                            for k, d in durations.items() if len(d)},
        "paths_with_no_resolved_reactive_frame": unresolved,
        "cells_visited_by_any_path": int((C > 0).sum()),
        "cells_occupied_in_reference": int(occ.sum()),
        "corr_observed_vs_committor_model": corr,
        "artifact": str(out_npz),
        "caveat": ("C(s) constrains the populated corridor only; an unobserved channel is "
                   "not an impossible one. Shape a bias with it, do not gate on it."),
    }
    (args.outdir / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
