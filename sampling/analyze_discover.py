#!/usr/bin/env python3
"""Stage 1 (DISCOVER) analysis: what did the wide MetaD + flow-bias run actually reach?

    python -m sampling.analyze_discover --campaign local_work/dhh_stage1_discover/flow \
        --reference local_work/input_data/ala2_cg_backbone_cb_6bead_200k.npz \
        --bias-npz <smooth_reference_bias.npz> --outdir <dir>

NOT the same question as `analyze_flow_md`. That one asks "does swapping the reference density
break acquisition behaviour", and needs a `control` arm to compare against. A discover run has
one arm and one question:

    **Did we reach (tic1,tic2) cells the atomistic reference never visited, and are the
    structures there physically valid?**

New coverage is the deliverable that stage 2 consumes. Everything else here exists to stop a
meaningless answer:

  - `--min-visits`: a cell touched by ONE frame of one walker is a fly-through, not a discovery.
    Cells are counted as reached only above this occupancy.
  - chirality: MetaD pushing on a TICA CV can find "new" regions by inverting a CA centre. Those
    are artifacts, not states. The colvar records chirality per frame, so they are counted and
    reported rather than silently harvested.
  - hill height decay: well-tempered MetaD deposits `HEIGHT * exp(-V/(kT(gamma-1)))`. If the
    height has not fallen, the run has not begun to converge anywhere and "coverage" is just
    the leading edge of a still-growing bias.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_colvar(path: Path):
    """(field name -> column) and the numeric block, tolerating PLUMED's `#! SET` lines."""
    fields = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#! FIELDS"):
                fields = line.split()[2:]
                break
    if fields is None:
        raise ValueError(f"{path}: no FIELDS header")
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    return {f: data[:, i] for i, f in enumerate(fields)}, data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", type=Path, required=True, help="the ARM dir holding replica_*")
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--bias-npz", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--discard-ps", type=float, default=10.0,
                    help="must match METAD UPDATE_FROM; before it there is no bias")
    ap.add_argument("--min-visits", type=int, default=5,
                    help="frames in a cell before it counts as reached; 1 is a fly-through")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = np.load(a.bias_npz, allow_pickle=True)
    coef, mean = b["tica_coefficients"], b["tica_mean"]
    pairs = b["pair_indices"]
    bounds = b["bounds"]

    # reference TICA projection, from the same frozen model the bias uses
    ref = np.load(a.reference)
    R = ref["R"] if "R" in ref else ref[ref.files[0]]
    d_ref = np.linalg.norm(R[:, pairs[:, 0], :] - R[:, pairs[:, 1], :], axis=-1)
    z_ref = (d_ref - mean) @ coef

    reps = sorted(p for p in a.campaign.glob("replica_*") if p.is_dir())
    if not reps:
        raise SystemExit(f"no replica_* under {a.campaign}")
    Z, PHI, PSI, CHI, BIAS = [], [], [], [], []
    for r in reps:
        c, _ = read_colvar(r / "colvar.dat")
        m = c["time"] >= a.discard_ps
        Z.append(np.column_stack([c["tic1"][m], c["tic2"][m]]))
        PHI.append(c["phi"][m]); PSI.append(c["psi"][m])
        if "chirality" in c:
            CHI.append(c["chirality"][m])
        BIAS.append(c.get("metad.bias", np.zeros(m.sum()))[m] if "metad.bias" in c
                    else np.zeros(int(m.sum())))
    z = np.concatenate(Z); phi = np.concatenate(PHI); psi = np.concatenate(PSI)
    chi = np.concatenate(CHI) if CHI else None

    # shared grid so "new" is a like-for-like cell comparison
    lo = np.minimum(z.min(0), z_ref.min(0)); hi = np.maximum(z.max(0), z_ref.max(0))
    edges = [np.linspace(lo[i], hi[i], a.bins + 1) for i in range(2)]
    H_new, _, _ = np.histogram2d(z[:, 0], z[:, 1], bins=edges)
    H_ref, _, _ = np.histogram2d(z_ref[:, 0], z_ref[:, 1], bins=edges)

    reached = H_new >= a.min_visits
    ref_has = H_ref >= a.min_visits
    novel = reached & ~ref_has
    cell_area = (edges[0][1] - edges[0][0]) * (edges[1][1] - edges[1][0])

    n_bad = int((np.abs(chi) < 0.2).sum()) if chi is not None else 0

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    a0 = ax[0, 0]
    a0.pcolormesh(edges[0], edges[1], np.where(H_ref > 0, H_ref, np.nan).T,
                  cmap="Greys", shading="auto")
    a0.scatter(z[::37, 0], z[::37, 1], s=1.5, c="#2b6cb0", alpha=.25, edgecolor="none")
    a0.add_patch(plt.Rectangle((bounds[0][0], bounds[1][0]),
                               bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0],
                               fill=False, ec="crimson", lw=1.2, ls="--"))
    a0.set(xlabel="tic1", ylabel="tic2",
           title=f"A  discovered (blue) over AA reference (grey)\n"
                 f"{len(reps)} walkers, {len(z):,} frames")

    a1 = ax[0, 1]
    a1.pcolormesh(edges[0], edges[1], np.where(novel, 1.0, np.nan).T,
                  cmap="autumn", shading="auto", vmin=0, vmax=1)
    a1.pcolormesh(edges[0], edges[1], np.where(ref_has, .3, np.nan).T,
                  cmap="Greys", shading="auto", vmin=0, vmax=1)
    a1.set(xlabel="tic1", ylabel="tic2",
           title=f"B  NEW cells (orange), >= {a.min_visits} visits\n"
                 f"{novel.sum()} new vs {ref_has.sum()} reference "
                 f"(+{100*novel.sum()/max(ref_has.sum(),1):.0f}%), "
                 f"{novel.sum()*cell_area:.2f} tic^2")

    a2 = ax[1, 0]
    a2.scatter(np.degrees(phi[::37]), np.degrees(psi[::37]), s=1.5, alpha=.25,
               c="#2b6cb0", edgecolor="none")
    a2.set(xlim=(-180, 180), ylim=(-180, 180), xlabel="phi (deg)", ylabel="psi (deg)",
           xticks=[-180, -90, 0, 90, 180], yticks=[-180, -90, 0, 90, 180],
           title=f"C  Ramachandran\nalphaL (phi>0) {100*(phi>0).mean():.1f}% of frames"
                 + (f"; {100*n_bad/len(chi):.2f}% near-planar CA" if chi is not None else ""))

    # D -- well-tempered height decay. HILLS is shared under WALKERS_MPI, so this is the
    # single global bias every walker felt, not one replica's private history.
    a3 = ax[1, 1]
    hills = None
    for r in reps:
        if (r / "HILLS").exists() and (r / "HILLS").stat().st_size > 0:
            hills = np.loadtxt(r / "HILLS", comments="#"); break
    if hills is not None and hills.ndim == 2:
        k = max(1, len(hills) // 400)
        t, h = hills[::k, 0], hills[::k, 5]
        a3.plot(t, h, lw=.8, color="#dd6b20")
        w = max(1, len(h) // 40)
        a3.plot(t, np.convolve(h, np.ones(w) / w, "same"), lw=2, color="#7b341e")
        a3.set(xlabel="time (ps)", ylabel="hill height (kcal/mol)",
               title=f"D  well-tempered decay, {len(hills):,} hills shared\n"
                     f"last/first {h[-len(h)//20:].mean()/h[:len(h)//20].mean():.2f}")
    else:
        a3.text(.5, .5, "no HILLS", ha="center")
    for x in ax.ravel():
        x.grid(alpha=.25, lw=.5)
    fig.tight_layout()
    fig.savefig(a.outdir / "discover_coverage.png", dpi=140)
    plt.close(fig)

    np.savez_compressed(a.outdir / "discover_coverage.npz", z=z.astype(np.float32),
                        phi=phi.astype(np.float32), psi=psi.astype(np.float32),
                        novel=novel, reached=reached, ref_has=ref_has,
                        edges0=edges[0], edges1=edges[1])
    print(f"{len(reps)} walkers, {len(z):,} frames after {a.discard_ps} ps")
    print(f"  cells reached {reached.sum()}  (reference {ref_has.sum()})")
    print(f"  NEW cells     {novel.sum()}  = +{100*novel.sum()/max(ref_has.sum(),1):.0f}% "
          f"of reference coverage, {novel.sum()*cell_area:.2f} tic^2")
    print(f"  alphaL (phi>0) {100*(phi>0).mean():.2f}% of frames")
    if chi is not None:
        print(f"  near-planar CA (|chirality|<0.2) {100*n_bad/len(chi):.3f}% "
              f"-- these are artifacts, not states")
    print(f"wrote {a.outdir}/discover_coverage.png")


if __name__ == "__main__":
    main()
