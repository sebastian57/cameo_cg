#!/usr/bin/env python3
"""Phase 3 analysis: does swapping the reference density break acquisition behaviour?

    python -m sampling.analyze_flow_md --campaign local_work/flow_phase3 \
        --bias-npz <b.npz> --reference <ref.npz> --outdir local_work/flow_phase3/analysis

Reads each arm's `colvar.dat` -- which records what the trajectories ACTUALLY experienced,
including the realised bias value -- rather than inferring behaviour from the tabulated field.
A rim well at -1.3 kcal/mol only matters if a trajectory ever visits it.

The comparison is against `control`, not against each other: both biases are enrichments over
the same unbiased system, so "did it enrich the transition region" and "did it leave the
basins and the geometry alone" are the two questions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ("control", "kde", "flow")


def load_arm(root: Path, arm: str) -> dict:
    cols, files = [], sorted((root / arm).glob("replica_*/colvar.dat"))
    for f in files:
        d = np.loadtxt(f, comments=("#", "@"))
        if d.ndim == 2 and len(d):
            cols.append(d)
    if not cols:
        raise SystemExit(f"no colvar data for {arm}")
    names = [l.split()[2:] for l in open(files[0]) if l.startswith("#! FIELDS")][0]
    return dict(data=np.concatenate(cols), names=names, n_replicas=len(cols),
                per_replica=cols)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--bias-npz", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--bins", type=int, default=60)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    from sampling.biases.tica_regional import SmoothTICABias
    from sampling.flow_bias import kde_density_and_grad, transition_weights

    bias = SmoothTICABias.load(a.bias_npz)
    kbt = float(bias.kbt_kcal_mol)
    z_ref = np.asarray(bias.projection.transform(np.load(a.reference)["R"]), np.float64)[:, :2]
    bl, bh = bias.bounds[:, 0], bias.bounds[:, 1]
    r_w = transition_weights(a.bias_npz)
    C, band = np.asarray(bias.centers), np.asarray(bias.bandwidth)

    arms = {k: load_arm(a.campaign, k) for k in ARMS}
    # PER-ARM column maps: the control has no `acq.bias` column, so one shared index is wrong
    for v in arms.values():
        v["idx"] = {n: i for i, n in enumerate(v["names"])}
    idx = arms["control"]["idx"]

    # shared TICA grid over the reference range, so coverage is comparable
    zz = np.concatenate([v["data"][:, [idx["tic1"], idx["tic2"]]] for v in arms.values()])
    lo = np.minimum(z_ref.min(0), zz.min(0))
    hi = np.maximum(z_ref.max(0), zz.max(0))
    edges = [np.linspace(lo[k], hi[k], a.bins + 1) for k in range(2)]
    Href, _, _ = np.histogram2d(z_ref[:, 0], z_ref[:, 1], bins=edges)
    ref_occ = Href > 0

    def basins(phi, psi):
        b = ((phi > -180) & (phi < -20) & ((psi > 90) | (psi < -150))).mean() * 100
        aR = ((phi > -160) & (phi < -20) & (psi > -120) & (psi < 50)).mean() * 100
        aL = ((phi > 20) & (phi < 100) & (psi > -20) & (psi < 100)).mean() * 100
        return b, aR, aL

    out = {}
    for arm, v in arms.items():
        d = v["data"]
        z = d[:, [idx["tic1"], idx["tic2"]]]
        phi = np.degrees(d[:, idx["phi"]]); psi = np.degrees(d[:, idx["psi"]])
        chi = np.degrees(d[:, idx["chirality"]])
        H, _, _ = np.histogram2d(z[:, 0], z[:, 1], bins=edges)
        # transition occupancy: r(z) evaluated on the visited points, i.e. how much of the
        # trajectory sat where the acquisition field actually asks for samples
        r_vis, _ = kde_density_and_grad(z, C, r_w, band)
        outside = ((z[:, 0] < bl[0]) | (z[:, 0] > bh[0]) |
                   (z[:, 1] < bl[1]) | (z[:, 1] > bh[1]))
        rec = dict(
            n_frames=int(len(d)), n_replicas=v["n_replicas"],
            coverage_pct=float((H[ref_occ] > 0).mean() * 100),
            cells_visited=int((H > 0).sum()),
            new_cells_vs_ref=int(((H > 0) & ~ref_occ).sum()),
            basins=dict(zip(("beta", "alphaR", "alphaL"), [round(x, 2) for x in basins(phi, psi)])),
            transition_occupancy=float(r_vis.mean()),
            outside_bounds_pct=float(outside.mean() * 100),
            chirality=dict(mean=float(chi.mean()), sd=float(chi.std()),
                           sign_purity=float((np.sign(chi) == np.sign(np.median(chi))).mean())),
        )
        if "acq.bias" in v["idx"]:
            b = d[:, v["idx"]["acq.bias"]]
            rec["realised_bias"] = dict(
                mean=float(b.mean()), min=float(b.min()), max=float(b.max()),
                p1=float(np.percentile(b, 1)), frac_below_minus1=float((b < -1).mean()))
        out[arm] = rec

    from sampling.mapping import dihedral_deg, get_mapping, wrap_deg
    m = get_mapping("ala2_backbone_cb_6")
    R_ref = np.load(a.reference)["R"].astype(np.float64)
    cv = lambda n: wrap_deg(dihedral_deg(R_ref, m.cvs[n].bead_indices) + m.cvs[n].shift_deg)
    ref_basins = dict(zip(("beta", "alphaR", "alphaL"),
                          [round(x, 2) for x in basins(cv("phi"), cv("psi"))]))
    summary = dict(kbt=kbt, arms=out,
                   reference=dict(n_frames=int(len(z_ref)), basins=ref_basins))
    (a.outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"reference basins: {ref_basins}\n")
    print(f"{'':22s}" + "".join(f"{k:>12s}" for k in ARMS))
    rows = [("frames", lambda r: r["n_frames"]),
            ("coverage of ref %", lambda r: round(r["coverage_pct"], 1)),
            ("cells visited", lambda r: r["cells_visited"]),
            ("NEW cells vs ref", lambda r: r["new_cells_vs_ref"]),
            ("beta %", lambda r: r["basins"]["beta"]),
            ("alphaR %", lambda r: r["basins"]["alphaR"]),
            ("alphaL %", lambda r: r["basins"]["alphaL"]),
            ("transition occ.", lambda r: round(r["transition_occupancy"], 5)),
            ("outside bounds %", lambda r: round(r["outside_bounds_pct"], 2)),
            ("chirality sd (deg)", lambda r: round(r["chirality"]["sd"], 1)),
            ("chirality purity", lambda r: round(r["chirality"]["sign_purity"], 4))]
    for lbl, fn in rows:
        print(f"{lbl:22s}" + "".join(f"{fn(out[k]):>12}" for k in ARMS))
    print(f"\nrealised bias (kcal/mol), from colvar.dat:")
    for lbl, key in (("  mean", "mean"), ("  min", "min"), ("  1st pct", "p1"),
                     ("  frac < -1", "frac_below_minus1")):
        print(f"{lbl:22s}" + "".join(
            f"{round(out[k]['realised_bias'][key], 4):>12}" if "realised_bias" in out[k]
            else f"{'-':>12}" for k in ARMS))

    enrich = out["flow"]["transition_occupancy"] / out["control"]["transition_occupancy"]
    enrich_k = out["kde"]["transition_occupancy"] / out["control"]["transition_occupancy"]
    print(f"\ntransition enrichment vs control:  KDE {enrich_k:.3f}x   flow {enrich:.3f}x")

    _figures(a, arms, idx, edges, z_ref, ref_occ, bias, out)
    print(f"wrote {a.outdir}/summary.json and figures")


def _figures(a, arms, idx, edges, z_ref, ref_occ, bias, out) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gx = 0.5 * (edges[0][:-1] + edges[0][1:])
    gy = 0.5 * (edges[1][:-1] + edges[1][1:])
    ext = [edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]]
    kw = dict(origin="lower", extent=ext, aspect="auto")
    bl, bh = bias.bounds[:, 0], bias.bounds[:, 1]

    # fig1: TICA occupancy per arm + reference
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))
    Hr, _, _ = np.histogram2d(z_ref[:, 0], z_ref[:, 1], bins=edges)
    panels = [("reference (mapped AA)", Hr)] + [
        (k, np.histogram2d(arms[k]["data"][:, arms[k]["idx"]["tic1"]],
                           arms[k]["data"][:, arms[k]["idx"]["tic2"]], bins=edges)[0])
        for k in ARMS]
    for i, (t, H) in enumerate(panels):
        P = H / max(H.sum(), 1)
        with np.errstate(divide="ignore"):
            F = -0.5922 * np.log(np.where(P > 0, P, np.nan))
        F = F - np.nanmin(F)
        im = ax[i].imshow(np.clip(F, 0, 6).T, cmap="viridis", vmin=0, vmax=6, **kw)
        ax[i].add_patch(plt.Rectangle((bl[0], bl[1]), bh[0] - bl[0], bh[1] - bl[1],
                                      fill=False, ec="r", lw=1.2, ls="--"))
        ax[i].set_title(f"{t}\n({int((H>0).sum())} cells)")
        ax[i].set_xlabel("tic1"); plt.colorbar(im, ax=ax[i], label="kcal/mol")
    ax[0].set_ylabel("tic2")
    fig.tight_layout(); fig.savefig(a.outdir / "fig1_tica_occupancy.png", dpi=130); plt.close(fig)

    # fig2: Ramachandran per arm
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    e = [np.linspace(-180, 180, 73)] * 2
    for i, k in enumerate(ARMS):
        phi = np.degrees(arms[k]["data"][:, arms[k]["idx"]["phi"]])
        psi = np.degrees(arms[k]["data"][:, arms[k]["idx"]["psi"]])
        H, _, _ = np.histogram2d(phi, psi, bins=e)
        P = H / H.sum()
        with np.errstate(divide="ignore"):
            F = -0.5922 * np.log(np.where(P > 0, P, np.nan))
        im = ax[i].imshow(np.clip(F - np.nanmin(F), 0, 6).T, cmap="viridis", vmin=0, vmax=6,
                          origin="lower", extent=[-180, 180, -180, 180], aspect="auto")
        ax[i].set_title(f"{k}   alphaL {out[k]['basins']['alphaL']}%")
        ax[i].set_xlabel("phi (deg)"); plt.colorbar(im, ax=ax[i], label="kcal/mol")
    ax[0].set_ylabel("psi (deg)")
    fig.tight_layout(); fig.savefig(a.outdir / "fig2_ramachandran.png", dpi=130); plt.close(fig)

    # fig3: realised bias + geometry sanity
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for k in ("kde", "flow"):
        b = arms[k]["data"][:, arms[k]["idx"]["acq.bias"]]
        ax[0].hist(b, bins=80, alpha=.6, density=True, label=f"{k} (min {b.min():.2f})")
    ax[0].set_xlabel("realised bias (kcal/mol)"); ax[0].legend()
    ax[0].set_title("what the trajectories ACTUALLY felt")
    for k in ARMS:
        ax[1].hist(np.degrees(arms[k]["data"][:, arms[k]["idx"]["chirality"]]), bins=90, alpha=.5,
                   density=True, label=k)
    ax[1].set_xlabel("chirality dihedral (deg)"); ax[1].legend()
    ax[1].set_title("geometry: no mirror images should appear")
    for k in ARMS:
        z = arms[k]["data"][:, [arms[k]["idx"]["tic1"], arms[k]["idx"]["tic2"]]]
        out_frac = np.cumsum(((z[:, 0] < bl[0]) | (z[:, 0] > bh[0]) |
                              (z[:, 1] < bl[1]) | (z[:, 1] > bh[1]))) / np.arange(1, len(z) + 1)
        ax[2].plot(np.linspace(0, 1, len(out_frac)), out_frac * 100, label=k)
    ax[2].set_xlabel("fraction of trajectory"); ax[2].set_ylabel("% outside bounds")
    ax[2].set_title("support excursions"); ax[2].legend()
    fig.tight_layout(); fig.savefig(a.outdir / "fig3_bias_and_geometry.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
