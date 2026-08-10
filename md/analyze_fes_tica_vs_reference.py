#!/usr/bin/env python3
"""Quantitative CG-MD evaluation against a mapped-AA reference.

Ramachandran and TICA free-energy surfaces, Jensen-Shannon divergence, and cell
coverage -- the standard comparison used for every wide160 MD run in this project.

    python -m md.analyze_fes_tica_vs_reference \
        --npz local_work/md_runs/<run>/traj_*_rep*.npz \
        --reference <mapped-AA reference>.npz \
        --mapping ala2_backbone_cb_6 \
        --outdir local_work/md_analysis/<run> --prefix <run>

Replaces the one-off `local_work/analyze_ala2_*_quantitative.py` scripts, which
hardcoded the 5-bead paper reference and could not be pointed at a 6-bead run.
TICA/FES primitives are imported from `md.analyze_traj` rather than reimplemented.

Protocol (matches the prior 8x3ns dt=4fs analyses so numbers stay comparable):
  * first `--discard-frac` of every replica dropped (equilibration)
  * TICA is fit on the REFERENCE ONLY and the CG trajectories are projected onto it,
    so the comparison never lets the model choose its own favourable coordinates
  * both ensembles are histogrammed on ONE shared grid per space
  * coverage counts reference-occupied cells that the CG ensemble also reaches
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

from md.analyze_traj import build_features, fit_tica            # noqa: E402
from sampling.mapping import get_mapping, normalized_signed_volume  # noqa: E402

KB_KCAL = 0.0019872042586


def all_pairs(n: int) -> np.ndarray:
    return np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=int)


def free_energy(H: np.ndarray, kT: float) -> np.ndarray:
    p = H / max(H.sum(), 1.0)
    with np.errstate(divide="ignore"):
        F = -kT * np.log(p)
    F[~np.isfinite(F)] = np.nan
    return F - np.nanmin(F)


def js_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits between two histograms."""
    p = P.ravel().astype(np.float64); p = p / max(p.sum(), 1.0)
    q = Q.ravel().astype(np.float64); q = q / max(q.sum(), 1.0)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def coverage(ref_H: np.ndarray, cg_H: np.ndarray) -> float:
    """Fraction of reference-occupied cells the CG ensemble also visits."""
    occ = ref_H > 0
    if not occ.any():
        return float("nan")
    return float((cg_H[occ] > 0).mean() * 100.0)


def _hist2d(x, y, xe, ye):
    H, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    return H


def load_replicas(paths, discard_frac: float, mapping, max_bond: float,
                  drop_replica_frac: float):
    """Load replicas, discard equilibration, and REJECT dissociated frames.

    A CG model can blow the molecule apart; one such replica in this project reached a
    2775 A "bond". Those frames are not merely noisy -- they dominate any distance-based
    feature, stretch the TICA grid until the reference collapses into a single cell, and
    silently corrupt basin populations. Filtering is mandatory, not optional.

    A replica losing more than `drop_replica_frac` of its frames is discarded entirely
    and reported: it did not simulate the intended system.
    """
    Rs, per = [], []
    for p in paths:
        with np.load(p) as d:
            R = np.asarray(d["R"], dtype=np.float64)
        n_total = len(R)
        R = R[int(n_total * discard_frac):]
        bl = np.stack([np.linalg.norm(R[:, i] - R[:, j], axis=-1)
                       for i, j in mapping.bonds], axis=1)
        ok = bl.max(axis=1) <= max_bond
        frac_bad = float(1.0 - ok.mean())
        rec = {"file": Path(p).name, "frames_total": int(n_total),
               "frames_after_discard": int(len(R)), "frames_dissociated": int((~ok).sum()),
               "dissociated_pct": round(100 * frac_bad, 3),
               "max_bond_A": float(bl.max())}
        if frac_bad > drop_replica_frac:
            rec["status"] = "DROPPED (dissociated)"
            per.append(rec)
            print(f"  !! {rec['file']}: {100*frac_bad:.1f}% dissociated "
                  f"(max bond {bl.max():.1f} A) -- REPLICA DROPPED")
            continue
        rec["status"] = "kept"
        rec["frames_kept"] = int(ok.sum())
        if (~ok).any():
            print(f"  {rec['file']}: dropped {int((~ok).sum())} dissociated frames "
                  f"({100*frac_bad:.2f}%)")
        Rs.append(R[ok])
        per.append(rec)
    if not Rs:
        raise SystemExit("every replica dissociated; nothing to analyse")
    return np.concatenate(Rs), per


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz", nargs="+", help="CG trajectory NPZ files (single ensemble)")
    src.add_argument("--ensemble", action="append", default=None, metavar="LABEL=GLOB",
                     help="named ensemble, repeatable: 'preREM=path/rep*.npz'. Every "
                          "ensemble is compared to the reference on ONE shared grid, so "
                          "models/stages are directly comparable (replaces the pre/post-REM "
                          "and narrow-vs-wide one-offs).")
    ap.add_argument("--reference", type=Path, required=True, help="mapped-AA reference NPZ")
    ap.add_argument("--mapping", type=str, default="ala2_backbone_cb_6")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--prefix", type=str, required=True)
    ap.add_argument("--discard-frac", type=float, default=0.20)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=298.0)
    ap.add_argument("--lagtime", type=int, default=20)
    ap.add_argument("--ref-stride", type=int, default=1)
    ap.add_argument("--max-bond", type=float, default=3.0,
                    help="frames with any CG bond longer than this are dissociated and "
                         "are rejected (default 3.0 A; physical bonds are ~1.3-1.6)")
    ap.add_argument("--drop-replica-frac", type=float, default=0.02,
                    help="a replica losing more than this fraction to dissociation is "
                         "dropped whole and reported (default 0.02)")
    ap.add_argument("--cv-x", type=str, default="phi",
                    help="mapping CV for the first FES axis (default phi). The 5-bead "
                         "legacy mapping has no true phi/psi -- use chirality/psi_proxy.")
    ap.add_argument("--cv-y", type=str, default="psi",
                    help="mapping CV for the second FES axis (default psi)")
    ap.add_argument("--enrichment", action="store_true",
                    help="emit log2(ensemble/reference) density maps -- WHERE a bias pulls "
                         "the ensemble. Red = over-sampled, blue = depleted. Read this "
                         "before building a dataset: it catches a sampler that flattens "
                         "basin ratios instead of adding transition information.")
    ap.add_argument("--coverage-vs-time", action="store_true",
                    help="also emit coverage as a function of aggregate simulated time "
                         "(log axis), separating 'more replicas' from 'longer chains'")
    ap.add_argument("--frame-ps", type=float, default=0.2,
                    help="ps per saved frame, for --coverage-vs-time (default 0.2)")
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()

    m = get_mapping(args.mapping)
    kT = KB_KCAL * args.temperature
    args.outdir.mkdir(parents=True, exist_ok=True)

    import glob as _glob
    if args.ensemble:
        ensembles = []
        for spec in args.ensemble:
            if "=" not in spec:
                raise SystemExit(f"--ensemble expects LABEL=GLOB, got {spec!r}")
            label, pattern = spec.split("=", 1)
            files = sorted(f for f in _glob.glob(pattern) if "partial" not in f)
            if not files:
                raise SystemExit(f"ensemble {label!r}: no files match {pattern!r}")
            ensembles.append((label, files))
    else:
        ensembles = [("CG MD", list(args.npz))]

    loaded = []
    for label, files in ensembles:
        print(f"[{label}] {len(files)} file(s)")
        Rl, perl = load_replicas(files, args.discard_frac, m, args.max_bond,
                                 args.drop_replica_frac)
        loaded.append({"label": label, "R": Rl, "per_replica": perl, "files": files})
    R_cg, per_rep = loaded[0]["R"], loaded[0]["per_replica"]
    with np.load(args.reference) as d:
        R_ref = np.asarray(d["R"], dtype=np.float64)[:: args.ref_stride]
    if R_cg.shape[1] != R_ref.shape[1]:
        raise SystemExit(f"bead mismatch: CG {R_cg.shape[1]} vs reference {R_ref.shape[1]}")
    print(f"{len(loaded)} ensemble(s), {sum(len(e['R']) for e in loaded)} CG frames "
          f"({args.discard_frac:.0%} discarded) | reference {len(R_ref)} frames")

    # ---------- Ramachandran ----------
    for cv in (args.cv_x, args.cv_y):
        if cv not in m.cvs:
            raise SystemExit(f"mapping {m.name} has no CV {cv!r}; available: {sorted(m.cvs)}")
    phi_r, psi_r = m.cvs[args.cv_x].evaluate(R_ref), m.cvs[args.cv_y].evaluate(R_ref)
    phi_c, psi_c = m.cvs[args.cv_x].evaluate(R_cg), m.cvs[args.cv_y].evaluate(R_cg)
    edges = np.linspace(-180, 180, args.bins + 1)
    Href = _hist2d(phi_r, psi_r, edges, edges)
    Hcg = _hist2d(phi_c, psi_c, edges, edges)
    rama_js, rama_cov = js_divergence(Href, Hcg), coverage(Href, Hcg)   # primary ensemble

    # ---------- TICA: fit on the REFERENCE only ----------
    pairs = all_pairs(R_ref.shape[1])
    Xref, Xcg = build_features(R_ref, pairs), build_features(R_cg, pairs)
    model, Yref = fit_tica(Xref, args.lagtime)
    Ycg = np.asarray(model.transform(Xcg), dtype=np.float64)
    lo = np.minimum(Yref.min(0), Ycg.min(0))[:2]
    hi = np.maximum(Yref.max(0), Ycg.max(0))[:2]
    xe = np.linspace(lo[0], hi[0], args.bins + 1)
    ye = np.linspace(lo[1], hi[1], args.bins + 1)
    Tref = _hist2d(Yref[:, 0], Yref[:, 1], xe, ye)
    Tcg = _hist2d(Ycg[:, 0], Ycg[:, 1], xe, ye)
    tica_js, tica_cov = js_divergence(Tref, Tcg), coverage(Tref, Tcg)

    # ---------- chirality / parity ----------
    ctr = m.inversion_centers()
    chi_ref = chi_cg = None
    if len(ctr) == 1:
        c, nb = next(iter(ctr.items()))
        chi_ref = normalized_signed_volume(R_ref, c, nb)
        chi_cg = normalized_signed_volume(R_cg, c, nb)

    # ---------- basins ----------
    def basins(phi, psi):
        """Only meaningful when the axes are a true Ramachandran (cv_x=phi, cv_y=psi)."""
        return {"alphaR_pct": float(((phi > -180) & (phi < 0) & (psi > -120) & (psi < 50)).mean() * 100),
                "alphaL_pct": float(((phi > 0) & (phi < 120) & (psi > -50) & (psi < 100)).mean() * 100),
                "phi_positive_pct": float((phi > 0).mean() * 100)}

    # ---------- per-ensemble metrics on the SHARED grids ----------
    for e in loaded:
        pr, ps = m.cvs[args.cv_x].evaluate(e["R"]), m.cvs[args.cv_y].evaluate(e["R"])
        Hh = _hist2d(pr, ps, edges, edges)
        Tt = _hist2d(*np.asarray(model.transform(build_features(e["R"], pairs)),
                                 dtype=np.float64)[:, :2].T, xe, ye)
        e.update(rama_H=Hh, tica_H=Tt,
                 rama_js=js_divergence(Href, Hh), rama_cov=coverage(Href, Hh),
                 tica_js=js_divergence(Tref, Tt), tica_cov=coverage(Tref, Tt),
                 cv_x=pr, cv_y=ps)

    # ---------- plots: reference + one panel per ensemble, one shared scale ----------
    for tag, refH, ax_lab, ex in (
        ("rama", Href, (f"{args.cv_x} (deg)", f"{args.cv_y} (deg)"),
         [edges[0], edges[-1], edges[0], edges[-1]]),
        ("tica", Tref, ("TIC 1", "TIC 2"), [xe[0], xe[-1], ye[0], ye[-1]]),
    ):
        panels = [("reference (mapped AA)", refH)] + [
            (f"{e['label']}  JS={e[tag + '_js']:.3f}", e[tag + "_H"]) for e in loaded]
        fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.4),
                                 constrained_layout=True, squeeze=False)
        for ax, (name, H) in zip(axes[0], panels):
            im = ax.imshow(free_energy(H, kT).T, origin="lower", extent=ex, aspect="auto",
                           cmap="turbo", vmin=0, vmax=8.0)
            ax.set_xlabel(ax_lab[0]); ax.set_ylabel(ax_lab[1]); ax.set_title(name, fontsize=10)
            fig.colorbar(im, ax=ax, label="F (kcal/mol)")
        fig.suptitle(args.title or args.prefix)
        out = args.outdir / f"{args.prefix}_{tag}_fes.png"
        fig.savefig(out, dpi=170); plt.close(fig)
        print("  wrote", out)

    # ---------- enrichment maps ----------
    if args.enrichment:
        for tag, refH, ax_lab, ex in (
            ("rama", Href, (f"{args.cv_x} (deg)", f"{args.cv_y} (deg)"),
             [edges[0], edges[-1], edges[0], edges[-1]]),
            ("tica", Tref, ("TIC 1", "TIC 2"), [xe[0], xe[-1], ye[0], ye[-1]]),
        ):
            fig, axes = plt.subplots(1, len(loaded), figsize=(5.6 * len(loaded), 4.4),
                                     constrained_layout=True, squeeze=False)
            pr = refH / max(refH.sum(), 1.0)
            for ax, e in zip(axes[0], loaded):
                H = e[tag + "_H"]; pc = H / max(H.sum(), 1.0)
                both = (pr > 0) | (pc > 0)
                floor = 0.5 / max(refH.sum(), H.sum(), 1.0)
                ratio = np.full(pr.shape, np.nan)
                ratio[both] = np.log2(np.maximum(pc[both], floor)
                                      / np.maximum(pr[both], floor))
                im = ax.imshow(ratio.T, origin="lower", extent=ex, aspect="auto",
                               cmap="RdBu_r", vmin=-4, vmax=4)
                ax.set_xlabel(ax_lab[0]); ax.set_ylabel(ax_lab[1])
                ax.set_title(f"{e['label']}: log2(ens/ref)", fontsize=10)
                fig.colorbar(im, ax=ax, label="log2 density ratio")
            fig.suptitle((args.title or args.prefix) + "  -  where the bias pulls")
            out = args.outdir / f"{args.prefix}_{tag}_enrichment.png"
            fig.savefig(out, dpi=170); plt.close(fig)
            print("  wrote", out)

    # ---------- coverage vs aggregate simulated time ----------
    cov_curves = {}
    if args.coverage_vs_time:
        fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
        for e in loaded:
            n = len(e["R"])
            pts = np.unique(np.geomspace(max(50, n // 500), n, 25).astype(int))
            rows = []
            for k in pts:
                Hk = _hist2d(e["cv_x"][:k], e["cv_y"][:k], edges, edges)
                Tk = _hist2d(*np.asarray(model.transform(
                    build_features(e["R"][:k], pairs)), dtype=np.float64)[:, :2].T, xe, ye)
                rows.append({"frames": int(k), "time_ns": k * args.frame_ps / 1000.0,
                             "rama_coverage_pct": coverage(Href, Hk),
                             "tica_coverage_pct": coverage(Tref, Tk)})
            cov_curves[e["label"]] = rows
            t = [r["time_ns"] for r in rows]
            ax.plot(t, [r["rama_coverage_pct"] for r in rows], "o-", label=f"{e['label']} Rama")
            ax.plot(t, [r["tica_coverage_pct"] for r in rows], "s--", label=f"{e['label']} TICA")
        ax.set_xscale("log"); ax.set_xlabel("aggregate simulated time (ns)")
        ax.set_ylabel("reference cells covered (%)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        ax.set_title("Coverage vs sampling time")
        out = args.outdir / f"{args.prefix}_coverage_vs_time.png"
        fig.savefig(out, dpi=170); plt.close(fig)
        print("  wrote", out)

    summary = {
        "status": "ok",
        "protocol": {"ensembles": {e["label"]: len(e["files"]) for e in loaded},
                     "discard_frac": args.discard_frac,
                     "bins": args.bins, "temperature_K": args.temperature,
                     "tica_lagtime_frames": args.lagtime,
                     "tica_fit_on": "reference only; CG projected",
                     "reference": str(args.reference), "mapping": args.mapping},
        "frames": {"cg": int(len(R_cg)), "reference": int(len(R_ref)),
                   "replicas_kept": sum(1 for r in per_rep if r["status"] == "kept"),
                   "replicas_dropped": [r["file"] for r in per_rep if r["status"] != "kept"],
                   "max_bond_A": args.max_bond, "per_replica": per_rep},
        "metrics": {e["label"]: {"rama_js_bits": e["rama_js"],
                                 "rama_coverage_pct": e["rama_cov"],
                                 "tica_js_bits": e["tica_js"],
                                 "tica_coverage_pct": e["tica_cov"],
                                 "frames": int(len(e["R"]))} for e in loaded},
        "coverage_vs_time": cov_curves,
        # Basin labels are Ramachandran-specific: emitting alphaR/alphaL for
        # chirality/psi_proxy axes would be a confidently wrong number.
        "basins": ({"reference": basins(phi_r, psi_r), "cg": basins(phi_c, psi_c)}
                   if (args.cv_x, args.cv_y) == ("phi", "psi")
                   else {"note": f"not computed: axes are {args.cv_x}/{args.cv_y}, "
                                 f"not a Ramachandran"}),
    }
    if chi_cg is not None:
        maj = np.sign(np.median(chi_ref))
        summary["chirality"] = {
            "center_bead": int(c), "neighbors": [int(x) for x in nb],
            "reference": {"mean": float(chi_ref.mean()), "min": float(chi_ref.min()),
                          "max": float(chi_ref.max()),
                          "sign_purity": float(max((chi_ref > 0).mean(), (chi_ref < 0).mean()))},
            "cg": {"mean": float(chi_cg.mean()), "min": float(chi_cg.min()),
                   "max": float(chi_cg.max()),
                   "sign_purity": float(max((chi_cg > 0).mean(), (chi_cg < 0).mean())),
                   "mirror_frames": int((np.sign(chi_cg) != maj).sum()),
                   "mirror_pct": float((np.sign(chi_cg) != maj).mean() * 100),
                   "near_planar_pct": float((np.abs(chi_cg) < 0.15).mean() * 100)},
        }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    for e in loaded:
        print(f"\n  [{e['label']}]  Rama JS={e['rama_js']:.4f} cov={e['rama_cov']:.1f}%"
              f"   TICA JS={e['tica_js']:.4f} cov={e['tica_cov']:.1f}%")
    if "reference" in summary["basins"]:
        print(f"  basins  reference {summary['basins']['reference']}")
        print(f"          CG        {summary['basins']['cg']}")
    else:
        print(f"  basins  {summary['basins']['note']}")
    if chi_cg is not None:
        ch = summary["chirality"]["cg"]
        print(f"  chirality  CG mirror {ch['mirror_pct']:.3f}%   near-planar "
              f"{ch['near_planar_pct']:.3f}%   sign purity {ch['sign_purity']:.4f}")
    print(f"\nwrote {args.outdir}/summary.json")


if __name__ == "__main__":
    main()
