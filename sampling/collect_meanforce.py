#!/usr/bin/env python3
"""Stage 3 estimator: conditional mean force per CG state, with its uncertainty.

    python -m sampling.collect_meanforce --campaign local_work/meanforce \
        --weights local_work/input_data/ala2_bb6_aggforce_weight_matrix.npz \
        --out local_work/meanforce/meanforce_dataset.npz

WHAT COMES OUT
    ONE ROW PER STATE, not one per frame:

        R_cg      (n_states, n_beads, 3)   the restrained CG configuration
        F_mean    (n_states, n_beads, 3)   <M F_AA>, the conditional mean force
        SE        (n_states, n_beads, 3)   standard error, from n_eff (NOT n_frames)
        n_eff     (n_states, n_beads)      effective independent samples

    **This is where "more AA samples != more training weight" is enforced.** 1,568 frames at
    one CG configuration collapse to a single training row carrying a mean and an uncertainty.
    Emitting them as 1,568 equally-weighted rows would multiply that region's influence on the
    force-matching objective by 1,568, which is a different experiment.

WHICH FORCES
    The BIAS-FREE rerun (`unbiased_forces.trr`), mapped with the aggforce weight matrix -- the
    same map `collect.py` applies, never a group sum. The restraint lives in PLUMED and the
    rerun carries no `-plumed`, so these forces are free of it by construction.

    In the stiff-restraint limit `<M F_AA>` and `-<F_restraint>` both converge to -grad PMF.
    This uses the former because it needs no assumption about the restraint being ideal, and
    because it reuses the existing, validated mapping.

n_eff, NOT n_frames
    Consecutive frames are correlated. Reporting `sigma/sqrt(n_frames)` would understate the
    error by `sqrt(n_frames/n_eff)`. The integrated autocorrelation time is estimated per bead
    per component with an initial-positive-sequence cutoff.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KJ_NM_TO_KCAL_A = 1.0 / 41.84


def integrated_act(x: np.ndarray, c_max: int = 200) -> float:
    """Integrated autocorrelation time in frames, via the initial-positive-sequence rule.

    Summing the empirical autocorrelation to large lag adds noise faster than signal; the
    standard fix is to truncate at the first non-positive value.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 8 or not np.any(x):
        return 1.0
    var = np.dot(x, x) / n
    if var <= 0:
        return 1.0
    tau = 0.5
    for k in range(1, min(c_max, n - 1)):
        c = np.dot(x[:-k], x[k:]) / ((n - k) * var)
        if c <= 0:
            break
        tau += c
    return float(max(1.0, 2.0 * tau))


def figure(path: Path, F, SE, SD, NEFF, DRIFT, width, ps_per_state, output_ps) -> None:
    """Four panels answering the four ways this stage can be silently wrong."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

    # A -- is the signal above its own noise? This is the whole point of the stage.
    a0 = ax[0, 0]
    a0.scatter(SE.ravel(), np.abs(F).ravel(), s=14, alpha=.55, edgecolor="none", c="#2b6cb0")
    lim = [min(SE.min(), np.abs(F).min()) * .8, max(SE.max(), np.abs(F).max()) * 1.2]
    a0.plot(lim, lim, "k--", lw=1, label="S/N = 1")
    a0.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           xlabel="SE  (kcal/mol/A)", ylabel="|F_mean|  (kcal/mol/A)",
           title=f"A  signal vs its own error\nmedian S/N "
                 f"{np.median(np.abs(F))/np.median(SE):.2f} : 1")
    a0.legend(frameon=False, fontsize=9)

    # B -- did the restraint actually hold? Drift beyond the thermal width means the
    # sampled ensemble is not the one the target labels claim.
    a1 = ax[0, 1]
    a1.hist(DRIFT.ravel(), bins=30, color="#805ad5", alpha=.8)
    a1.axvline(width, color="k", ls="--", lw=1.2, label=f"thermal width {width:.2f} A")
    a1.set(xlabel="|<R_bead> - target|  (A)", ylabel="beads",
           title=f"B  restraint drift\nmax {DRIFT.max():.3f} A")
    a1.legend(frameon=False, fontsize=9)

    # C -- correlation. n_eff, not n_frames, is what sets the error bar.
    a2 = ax[1, 0]
    n_frames = ps_per_state / output_ps
    a2.hist(n_frames / NEFF.ravel(), bins=25, color="#dd6b20", alpha=.8)
    a2.set(xlabel=f"tau  (frames of {output_ps} ps)", ylabel="beads",
           title=f"C  force autocorrelation\nmedian tau "
                 f"{np.median(n_frames/NEFF)*output_ps:.2f} ps")

    # D -- how long must a production state run? Extrapolate from the sigma measured here,
    # which is the only number in this plot that does not depend on the run length.
    a3 = ax[1, 1]
    sigma = float(np.median(SD))
    tau_ps = float(np.median(n_frames / NEFF)) * output_ps
    ps = np.logspace(np.log10(5), np.log10(3000), 200)
    a3.plot(ps, sigma / np.sqrt(ps / tau_ps), color="#2b6cb0", lw=2)
    a3.scatter([ps_per_state], [np.median(SE)], s=70, zorder=5, c="#c53030",
               label=f"this run: {ps_per_state:g} ps -> {np.median(SE):.2f}")
    for tgt in (1.0, 0.5):
        need = tau_ps * (sigma / tgt) ** 2
        a3.axhline(tgt, color="k", ls=":", lw=.9)
        a3.annotate(f"SE {tgt} needs {need:.0f} ps", (need, tgt), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    a3.set(xscale="log", yscale="log", xlabel="ps per state", ylabel="SE  (kcal/mol/A)",
           title=f"D  sizing production\nsigma_F = {sigma:.1f} kcal/mol/A")
    a3.legend(frameon=False, fontsize=9, loc="lower left")

    for x in ax.ravel():
        x.grid(alpha=.25, lw=.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True,
                    help="aggforce weight matrix W (n_beads x n_aa). NEVER re-derive this: "
                         "the fit is not bit-reproducible (4.66e-01 vs 4.17e-05).")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--discard-ps", type=float, default=20.0,
                    help="equilibration dropped per state; the restraint has to pull the seed "
                         "onto its target first")
    ap.add_argument("--output-ps", type=float, default=0.2)
    ap.add_argument("--mapping", default="ala2_backbone_cb_6")
    ap.add_argument("--gmx", default="gmx")
    ap.add_argument("--fast", action="store_true",
                    help="read the TRR directly instead of shelling out to gmx three times "
                         "per state (~0.3 s vs ~5.9 s). Requires the solute to be the first "
                         "atoms of the system and unbroken, which holds for frozen campaigns.")
    ap.add_argument("--state-start", type=int, default=0,
                    help="process states [start, stop); lets several processes split one "
                         "campaign, which is how a 42,000-state collection is parallelised")
    ap.add_argument("--state-stop", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes over states")
    ap.add_argument("--max-drift", type=float, default=0.5,
                    help="reject a state whose mean bead position sits further than this "
                         "(Angstrom) from its restraint target -- the restraint did not hold")
    a = ap.parse_args()

    # Reuse collect.py's readers verbatim. NOT mdtraj: it exposes no TRR force block
    # (`Trajectory` has no `.forces` at all), and `W` is (n_beads, 22) -- the solute only --
    # so the frames must come out of `gmx traj` on the Protein group, not the full 2,642-atom
    # system. Both traps are already documented there; do not re-solve them here.
    from .collect import _gmx_traj, _make_whole
    from .mapping import get_mapping

    mapping = get_mapping(a.mapping)
    bead_atoms0 = [i - 1 for i in mapping.aa_atom_indices_1based]

    W = np.load(a.weights)["W"]
    man = json.loads((a.campaign / "manifest.json").read_text())
    targets = {int(s["state"]): np.asarray(s["target"], np.float64) for s in man["states"]}
    width = float(man.get("restraint_width_A", 0.1))

    states = sorted(p for p in a.campaign.glob("state_*") if p.is_dir())
    if not states:
        raise SystemExit(f"no state_* under {a.campaign}")
    states = states[a.state_start:a.state_stop]
    if not states:
        raise SystemExit(f"empty slice [{a.state_start}, {a.state_stop})")

    R_out, F_out, SE_out, NEFF_out, keep, rejected = [], [], [], [], [], []
    DRIFT_out, SD_out = [], []
    n_skip = int(a.discard_ps / a.output_ps)

    for sd in states:
        k = int(sd.name.split("_")[1])
        trr = sd / "unbiased_forces.trr"
        if not trr.exists():
            rejected.append((sd.name, "no unbiased_forces.trr")); continue
        n_aa = W.shape[1]
        if a.fast:
            # DIRECT TRR READ. The GROMACS route costs ~5.9 s/state (three subprocesses, each
            # paying a module load); at 50,000 states that is 82 h of serial CPU, an order of
            # magnitude more than the MD it analyses. mdtraj's Trajectory has no force block,
            # but the FILE object does -- `_read(n, None, False, True)` returns forces as its
            # 7th element. ~0.3 s/state.
            # Solute atoms only, and NO make_whole: the beads are frozen and the solute cannot
            # straddle a boundary within one state (verified by the pbc-margin guard at build).
            try:
                from mdtraj.formats import TRRTrajectoryFile
                with TRRTrajectoryFile(str(trr)) as fh:
                    nfr = len(fh)
                    out = fh._read(nfr, None, False, True)
                xyz, tc, frc = out[0], out[1], out[6]
                if frc is None:
                    raise ValueError("trr carries no forces (nstfout=0?)")
                xyz = np.asarray(xyz, np.float64); frc = np.asarray(frc, np.float64)
                tc = np.asarray(tc, np.float64)
                # DROP PLACEHOLDER FRAMES. A GROMACS .trr carries entries that hold neither
                # coordinates nor forces (this run: 4 of them at t = 2.5/7.5/12.5/17.5 ps among
                # 106 real frames). mdtraj allocates the arrays for ALL frames but only fills
                # the ones that actually carry the field, so those rows are UNINITIALISED
                # MEMORY -- not zeros. Testing for zero is therefore unsound and
                # NON-DETERMINISTIC: two runs over the same 200 states rejected 1 and then 149,
                # with drifts like 6.1e18 (the 0x5555... bit pattern). `gmx traj` skips these
                # frames, which is why the slow path never saw it.
                # Test finiteness AND physical magnitude instead; garbage fails both.
                # SELECT ON THE TIME GRID. The frame TIMES are written correctly even for
                # placeholder entries (they appeared at 2.5/7.5/12.5/17.5 ps, i.e. off the
                # 0.2 ps output grid), whereas their coordinate/force buffers are
                # uninitialised. Testing the buffers is therefore non-deterministic -- three
                # sequential runs agreed, but 12 PARALLEL shards did not, admitting garbage
                # frames on 1,226 states and creating 0.1 ps gaps. The time grid is a
                # property of the file, not of whatever memory happened to be reused.
                grid = tc / float(a.output_ps)     # NOT `k` -- that is the state index
                on_grid = np.abs(grid - np.round(grid)) < 1e-2
                fin = np.isfinite(xyz).all(axis=(1, 2)) & np.isfinite(frc).all(axis=(1, 2))
                mag = (np.abs(xyz).max(axis=(1, 2)) < 100.0)      # nm; any real box is << 100
                good = on_grid & fin & mag
                if not good.all():
                    xyz, frc, tc = xyz[good], frc[good], tc[good]
                R_aa = xyz[:, :n_aa]
                F_aa = frc[:, :n_aa]
                tf = tc
                # The surviving frames must be evenly spaced at output_ps. If they are not,
                # the filter let something through and the average would be silently wrong.
                if len(tc) > 2:
                    dt = np.diff(tc)
                    if not np.allclose(dt, dt[0], rtol=1e-2, atol=1e-4):
                        rejected.append((sd.name, f"irregular frame spacing after filtering "
                                                  f"({dt.min():.3f}..{dt.max():.3f} ps)"))
                        continue
            except Exception as exc:
                rejected.append((sd.name, f"fast read failed: {exc}")); continue
        else:
            try:
                _make_whole(sd, a.gmx)
                tc, R_aa = _gmx_traj(sd, "whole.trr", "-ox", "aa_coords.xvg", n_aa, a.gmx)
                tf, F_aa = _gmx_traj(sd, "unbiased_forces.trr", "-of", "aa_forces.xvg",
                                     n_aa, a.gmx)
            except (RuntimeError, ValueError) as exc:
                rejected.append((sd.name, str(exc).splitlines()[0])); continue
        if len(F_aa) != len(R_aa) or not np.allclose(tc, tf, atol=1e-6):
            rejected.append((sd.name, f"{len(R_aa)} coord vs {len(F_aa)} force frames"))
            continue
        if len(F_aa) <= n_skip + 8:
            rejected.append((sd.name, f"only {len(F_aa)} frames")); continue

        # coordinates are a plain bead SELECTION; forces need the aggforce map
        Rb = R_aa[n_skip:][:, bead_atoms0, :] * 10.0                 # nm -> Angstrom
        Fb = np.einsum("bn,fnd->fbd", W, F_aa[n_skip:] * KJ_NM_TO_KCAL_A)

        drift = np.linalg.norm(Rb.mean(0) - targets[k], axis=-1).max()
        if drift > a.max_drift:
            rejected.append((sd.name, f"restraint drift {drift:.2f} A")); continue

        Fm = Fb.mean(0)
        tau = np.array([[integrated_act(Fb[:, b, c]) for c in range(3)]
                        for b in range(Fb.shape[1])])
        neff = len(Fb) / tau
        se = Fb.std(0, ddof=1) / np.sqrt(neff)

        R_out.append(targets[k]); F_out.append(Fm); SE_out.append(se)
        NEFF_out.append(neff.mean(1)); keep.append(k)
        DRIFT_out.append(np.linalg.norm(Rb.mean(0) - targets[k], axis=-1))
        SD_out.append(Fb.std(0, ddof=1))

    if not keep:
        raise SystemExit("no usable states")
    R_out = np.stack(R_out); F_out = np.stack(F_out)
    SE_out = np.stack(SE_out); NEFF_out = np.stack(NEFF_out)
    DRIFT_out = np.stack(DRIFT_out); SD_out = np.stack(SD_out)

    out = a.out or (a.campaign / "meanforce_dataset.npz")
    np.savez_compressed(out, R=R_out.astype(np.float32), F=F_out.astype(np.float32),
                        SE=SE_out.astype(np.float32), n_eff=NEFF_out.astype(np.float32),
                        drift=DRIFT_out.astype(np.float32), sd=SD_out.astype(np.float32),
                        state=np.asarray(keep, np.int32))
    summary = dict(
        n_states_kept=len(keep), n_states_rejected=len(rejected),
        rejected=rejected[:20],
        frames_per_state=int(len(Fb)),
        median_n_eff=float(np.median(NEFF_out)),
        median_tau_frames=float(len(Fb) / np.median(NEFF_out)),
        median_SE=float(np.median(SE_out)),
        median_abs_F=float(np.median(np.abs(F_out))),
        naive_SE_if_frames_were_independent=float(np.median(SE_out) *
                                                  np.sqrt(np.median(NEFF_out) / len(Fb))),
        restraint_width_A=width,
    )
    (Path(out).with_suffix(".summary.json")).write_text(json.dumps(summary, indent=1))

    print(f"kept {len(keep)}/{len(states)} states"
          + (f", rejected {len(rejected)}" if rejected else ""))
    for n, why in rejected[:5]:
        print(f"  rejected {n}: {why}")
    print(f"frames/state {summary['frames_per_state']}, "
          f"median n_eff {summary['median_n_eff']:.0f} "
          f"(tau ~ {summary['median_tau_frames']:.1f} frames)")
    print(f"median |F_mean| {summary['median_abs_F']:.3f}   "
          f"median SE {summary['median_SE']:.3f} kcal/mol/A")
    print(f"  -> S/N {summary['median_abs_F']/max(summary['median_SE'],1e-9):.1f} : 1"
          f"   (single-frame labels are ~1:10)")
    print(f"  naive SE ignoring correlation would have been "
          f"{summary['naive_SE_if_frames_were_independent']:.3f} -- "
          f"understated by {summary['median_SE']/max(summary['naive_SE_if_frames_were_independent'],1e-9):.1f}x")
    print(f"wrote {out}")
    figure(Path(out).with_suffix(".png"), F_out, SE_out, SD_out, NEFF_out, DRIFT_out,
           width, float(len(Fb)) * a.output_ps, a.output_ps)


if __name__ == "__main__":
    main()
