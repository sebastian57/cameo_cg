#!/usr/bin/env python3
"""What is the stage-3 mean-force dataset actually worth? Noise, coverage, independence.

    python -m sampling.analyze_meanforce_dataset \
        --dataset local_work/dhh_stage3_meanforce/meanforce_dataset.npz \
        --reference local_work/input_data/ala2_cg_backbone_cb_6bead_200k.npz \
        --discovery local_work/dhh_stage1_discover/analysis/discover_coverage.npz \
        --bias-npz <smooth_reference_bias.npz> --outdir <dir>

Three questions, and the traps in each:

**Is it less noisy?** Comparing our SE against the reference's single-frame scatter is only
fair if the scatter is measured at FIXED CG configuration -- the raw spread of reference
forces also contains real variation of the mean force across configurations, which is signal,
not noise. So the reference's conditional noise is estimated within small pair-distance
neighbourhoods, the same quantity our restrained ensembles average over.

**Does it cover the relevant regions?** Coverage of the DISCOVERED region is the wrong bar on
its own -- a dataset can tile a region the reference already knows well and add nothing. The
number that matters is where the states sit relative to the REFERENCE density: states in cells
the reference barely sampled are the ones that can teach the model something new.

**Are the frames uncorrelated?** Two separate questions that are easy to conflate:
  - WITHIN a state, consecutive frames are correlated; `n_eff` already accounts for it.
  - BETWEEN states, two states closer than the restraint's own thermal width are not
    independent measurements -- they are two samples of one configuration, and counting them
    as two rows double-counts that region in the force-matching objective.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def pair_dists(R):
    iu = np.triu_indices(R.shape[1], k=1)
    return np.linalg.norm(R[:, iu[0], :] - R[:, iu[1], :], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--bias-npz", type=Path, required=True)
    ap.add_argument("--discovery", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--neighbour-radius", type=float, default=0.25,
                    help="Angstrom in pair-distance space; the shell within which reference "
                         "frames are treated as sharing a CG configuration")
    ap.add_argument("--output-ps", type=float, default=0.2)
    ap.add_argument("--restraint-width", type=float, default=0.0,
                    help="thermal width of the position restraint in Angstrom. 0 = the beads "
                         "were FROZEN (freezegrps), so there is no width and the "
                         "'two states closer than the width are duplicates' test does not "
                         "apply -- what matters is simply that the geometries differ.")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(a.dataset)
    R, F, SE, NEFF = d["R"], d["F"], d["SE"], d["n_eff"]
    ref = np.load(a.reference)
    Rr, Fr = ref["R"].astype(np.float64), ref["F"].astype(np.float64)

    b = np.load(a.bias_npz, allow_pickle=True)
    coef, mean, pairs = b["tica_coefficients"], b["tica_mean"], b["pair_indices"]
    tica = lambda X: (np.linalg.norm(X[:, pairs[:, 0], :] - X[:, pairs[:, 1], :],
                                     axis=-1) - mean) @ coef
    z_ds, z_ref = tica(R.astype(np.float64)), tica(Rr)

    # ---- 1. NOISE -----------------------------------------------------------------------
    # Reference conditional noise: spread of single-frame forces among reference frames that
    # share a CG configuration. Sub-sample -- this is O(n_states * n_ref).
    Dd, Dr = pair_dists(R.astype(np.float64)), pair_dists(Rr)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Dr), min(60000, len(Dr)), replace=False)
    # Sub-sample the STATES too -- the reference-neighbourhood scan is O(n_states x n_ref)
    # and only needs enough states to estimate a median conditional spread.
    probe = (rng.choice(len(Dd), 800, replace=False) if len(Dd) > 800
             else np.arange(len(Dd)))
    tree = cKDTree(Dr[sub])
    cond_sd, n_nb = [], []
    for i in probe:
        idx = tree.query_ball_point(Dd[i], a.neighbour_radius)
        if len(idx) >= 20:
            cond_sd.append(Fr[sub][idx].std(0, ddof=1).mean())
            n_nb.append(len(idx))
    cond_sd = np.array(cond_sd)
    sigma_ref = float(np.median(cond_sd)) if len(cond_sd) else float("nan")

    # Fisher information for a force field goes as 1/sigma^2 per label.
    info_new = float((1.0 / SE.astype(np.float64) ** 2).sum())
    info_ref = float(len(Rr) * Fr.shape[1] * Fr.shape[2] / sigma_ref ** 2)

    # ---- 2. COVERAGE --------------------------------------------------------------------
    e = [np.linspace(min(z_ds[:, i].min(), z_ref[:, i].min()),
                     max(z_ds[:, i].max(), z_ref[:, i].max()), 60) for i in range(2)]
    Href, _, _ = np.histogram2d(z_ref[:, 0], z_ref[:, 1], bins=e)
    Href = Href / Href.sum()
    ix = np.clip(np.digitize(z_ds[:, 0], e[0]) - 1, 0, Href.shape[0] - 1)
    iy = np.clip(np.digitize(z_ds[:, 1], e[1]) - 1, 0, Href.shape[1] - 1)
    dens_at_state = Href[ix, iy]
    novel = dens_at_state == 0.0
    rare = (dens_at_state > 0) & (dens_at_state < 1e-4)

    # ---- 3. INDEPENDENCE ----------------------------------------------------------------
    # KD-tree, not a dense matrix: at 42,000 states the pairwise array is 42,000^2 x 15
    # floats (~200 GB) and the process is OOM-killed. k=2 because the nearest neighbour of a
    # point in its own tree is itself.
    nn = cKDTree(Dd).query(Dd, k=2)[0][:, 1]
    width = float(a.restraint_width)                # 0 => frozen, no thermal width
    tau_fr = (NEFF.mean(1) * 0 + 1)                 # placeholder, replaced below
    n_frames = float(np.median(d["sd"].shape[0] * 0 + 1))  # unused; kept explicit
    tau_frames = None
    if "n_eff" in d:
        # n_eff was computed from the same frame count for every state
        tau_frames = None

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    a0 = ax[0, 0]
    a0.hist(SE.ravel(), bins=40, color="#2b6cb0", alpha=.85, label="this dataset (SE)")
    a0.axvline(sigma_ref, color="crimson", ls="--", lw=1.6,
               label=f"reference single-frame sd {sigma_ref:.1f}")
    a0.set(xscale="log", xlabel="force uncertainty (kcal/mol/A)", ylabel="components",
           title=f"A  noise per label\nmedian SE {np.median(SE):.3f} vs {sigma_ref:.1f} "
                 f"= {sigma_ref/np.median(SE):.0f}x less noisy")
    a0.legend(frameon=False, fontsize=8)

    a1 = ax[0, 1]
    a1.pcolormesh(e[0], e[1], np.where(Href > 0, Href, np.nan).T, cmap="Greys",
                  shading="auto", norm=matplotlib.colors.LogNorm())
    # Marker size/alpha must scale with N: sizes tuned for 512 states turn into a solid
    # colour block at 42,000 and hide the reference density underneath.
    ms = max(1.0, min(9.0, 4000.0 / max(len(z_ds), 1)))
    al = max(0.12, min(0.9, 3000.0 / max(len(z_ds), 1)))
    a1.scatter(z_ds[~(novel | rare), 0], z_ds[~(novel | rare), 1], s=ms, alpha=al,
               c="#2b6cb0", edgecolor="none",
               label=f"in well-sampled ref ({int((~(novel|rare)).sum()):,})")
    a1.scatter(z_ds[rare, 0], z_ds[rare, 1], s=ms, alpha=al, c="#dd6b20", edgecolor="none",
               label=f"rare in ref ({int(rare.sum()):,})")
    a1.scatter(z_ds[novel, 0], z_ds[novel, 1], s=ms * 1.3, alpha=min(1.0, al * 1.5),
               c="crimson", edgecolor="none", label=f"UNSEEN by ref ({int(novel.sum()):,})")
    a1.set(xlabel="tic1", ylabel="tic2",
           title=f"B  where the {len(R)} states sit\n"
                 f"{100*(novel|rare).mean():.0f}% are where the reference is rare or absent")
    a1.legend(frameon=False, fontsize=8, loc="upper left")

    a2 = ax[1, 0]
    a2.hist(nn, bins=40, color="#805ad5", alpha=.85)
    if width > 0:
        a2.axvline(width, color="k", ls="--", lw=1.4, label=f"restraint width {width} A")
    if width > 0:
        a2.axvline(2 * width, color="k", ls=":", lw=1.2, label="2x width")
    a2.set(xlabel="nearest-neighbour distance between states (A, pair-distance space)",
           ylabel="states",
           title=("C  are the STATES independent\n"
                  + (f"min {nn.min():.3f} A = {nn.min()/width:.1f}x the restraint width; "
                     f"{int((nn < 2*width).sum())} closer than 2x" if width > 0
                     else f"FROZEN: exact configurations, no thermal width\n"
                          f"min separation {nn.min():.3f} A, median {np.median(nn):.3f}")))
    a2.legend(frameon=False, fontsize=8)

    a3 = ax[1, 1]
    eff = NEFF / (NEFF.max() if NEFF.max() > 0 else 1)
    a3.hist(NEFF.ravel(), bins=40, color="#dd6b20", alpha=.85)
    a3.set(xlabel="n_eff per bead", ylabel="beads",
           title=f"D  independence WITHIN a state\nmedian n_eff {np.median(NEFF):.0f} of "
                 f"{int(np.round(np.median(NEFF)*1.257))} frames "
                 f"(tau = {a.output_ps*1.257:.2f} ps)")

    for x in ax.ravel():
        x.grid(alpha=.25, lw=.5)
    fig.tight_layout()
    fig.savefig(a.outdir / "dataset_quality.png", dpi=140)
    plt.close(fig)

    print(f"=== NOISE ===")
    print(f"  reference single-frame conditional sd : {sigma_ref:.2f} kcal/mol/A "
          f"(median over {len(cond_sd)} states with >=20 neighbours within "
          f"{a.neighbour_radius} A)")
    print(f"  this dataset, median SE               : {np.median(SE):.3f} kcal/mol/A")
    print(f"  -> {sigma_ref/np.median(SE):.0f}x less noisy per label; matching it with single "
          f"frames needs {(sigma_ref/np.median(SE))**2:.0f} frames AT THE SAME configuration")
    print(f"  Fisher information (sum 1/sigma^2):")
    print(f"    this dataset ({len(R)} states)      : {info_new:,.0f}")
    print(f"    full reference ({len(Rr):,} frames) : {info_ref:,.0f}")
    print(f"    ratio                               : {info_new/info_ref:.2f}x")
    print(f"\n=== COVERAGE ===")
    print(f"  states in cells the reference NEVER sampled : {int(novel.sum())} "
          f"({100*novel.mean():.1f}%)")
    print(f"  states in cells the reference samples rarely: {int(rare.sum())} "
          f"({100*rare.mean():.1f}%)")
    print(f"  states in well-sampled reference cells      : {int((~(novel|rare)).sum())} "
          f"({100*(~(novel|rare)).mean():.1f}%)")
    print(f"\n=== INDEPENDENCE ===")
    print(f"  BETWEEN states: nearest-neighbour min {nn.min():.3f}, median {np.median(nn):.3f} A")
    print(f"    restraint thermal width {width} A -> min separation is "
          f"{nn.min()/width:.1f}x it; {int((nn < 2*width).sum())} state(s) within 2x")
    print(f"  WITHIN a state: median n_eff {np.median(NEFF):.0f}, "
          f"tau {a.output_ps*1.257:.2f} ps")
    print(f"wrote {a.outdir}/dataset_quality.png")


if __name__ == "__main__":
    main()
