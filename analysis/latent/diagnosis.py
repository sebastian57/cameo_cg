#!/usr/bin/env python3
"""Latent-space diagnosis of the beta/alphaR population imbalance in bb6 CG models.

THE PROBLEM THIS ATTACKS
    Every pure-FM bb6 model reproduces the alphaL-beta free energy difference well
    (v2_192k_fm1000: dF = -0.160 kcal/mol, the best on record) while getting the beta/alphaR
    SPLIT badly wrong: 38.4 / 52.4 against a truth of 65.2 / 30.0. No mechanism has been
    identified. Comparing raw FES histograms has not separated the candidate explanations
    because an unweighted histogram lets a large discrepancy in a nearly-empty region look as
    important as a small one in a heavily-populated region.

WHY THE FLOW LATENT HELPS
    The acquisition flow `u = f_theta(z)` was fit by maximum likelihood to the ALL-ATOM
    REFERENCE ensemble in frozen TICA coordinates, so by construction

        z ~ p_ref   ==>   u ~ N(0, I).

    Two consequences do the work here:

    1. EQUAL VOLUME = EQUAL PROBABILITY. Lebesgue measure in u is reference probability mass.
       Every distance, density and divergence computed in u is automatically weighted by how
       much the reference ensemble actually cares, which is exactly what the raw-histogram
       comparison fails to do.

    2. THE REFERENCE SIDE BECOMES ANALYTIC. If u ~ N(0, I) then the marginal along ANY unit
       direction e is exactly N(0, 1), so the reference free energy along e is exactly
       F(s) = 1/2 kT s^2 -- no histogram, no binning noise, no finite-sample error. Model error
       is therefore measured against a NOISELESS baseline.

    The flow is deliberately small and slightly underfit (an expressive flow turns finite-sample
    density fluctuations into artificial energy corrugations). So the empirical u_ref is NOT
    exactly N(0, I), and the gap between empirical-reference and analytic-N(0,I) is the FLOW
    MISFIT FLOOR. Every model-vs-reference number below is reported alongside that floor; a
    model discrepancy smaller than the floor means nothing.

THE THREE HYPOTHESES, AND HOW THEY SEPARATE IN LATENT SPACE
    Let mu_beta and mu_alphaR be the reference basin centroids in u, and let
    e = (mu_alphaR - mu_beta)/|mu_alphaR - mu_beta| be the "beta->alphaR axis". Project
    s = u . e. Then:

      (a) SHIFTED DIVIDING SURFACE -- the saddle (density minimum between the two modes along
          s) moves. Mass reassigns wholesale without either well changing shape.
      (b) WIDENED alphaR WELL -- the alphaR mode's width along s (and its generalized variance
          in the orthogonal direction) grows, absorbing mass from beta while the saddle stays.
      (c) WRONG RELATIVE DEPTH -- modes and saddle stay put, the relative peak heights change.

    These are mutually exclusive readings of the same 1D profile, which is why the projection
    is the diagnostic and the 2D maps are the supporting evidence.

WHAT THIS CANNOT SEE
    The flow consumes only the 2D frozen TICA projection, never Cartesian coordinates. Two
    structurally distinct configurations at the same (tic1, tic2) are the SAME point in u --
    chirality inversion in particular is invisible here and stays with the signed-volume
    diagnostic. "Typical" also means typical OF THE AA REFERENCE, inheriting the reference's
    own sampling limitations.

USAGE
    python latent_diagnosis.py --outdir <dir> \
        --ensemble fm1000=<glob of traj_*.npz> [--ensemble rem400=<glob>] ...
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
from pathlib import Path

import numpy as np

from analysis.common.cli import add_project_root_argument
from analysis.common.paths import repo_root, resolve_glob, resolve_input, resolve_output
from analysis.common.provenance import write_manifest

# Canonical analysis inputs are supplied through the CLI.
# --- design tokens (dataviz reference palette, light surface) ---------------------------
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"
# Categorical slots in FIXED order. beta and alphaR take slots 1 and 2 -- the maximally
# separated pair -- because they are the pair under investigation.
BASIN_COLOR = {"beta": "#2a78d6", "alphaR": "#eb6834", "alphaL": "#1baf7a", "other": "#eda100"}
BASIN_MARKER = {"beta": "o", "alphaR": "s", "alphaL": "^", "other": "D"}  # secondary encoding
BASINS = ["beta", "alphaR", "alphaL", "other"]
SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_LO, DIV_MID, DIV_HI = "#2a78d6", "#f2f1ec", "#eb6834"   # two hues + NEUTRAL midpoint


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------------------
# Basin assignment -- verbatim from md/analyze_model_residuals_by_region.py so that every
# region number in this file is directly comparable with the force-residual probe. Order
# matters: later assignments overwrite earlier ones.
# ---------------------------------------------------------------------------------------
def assign_basins(R: np.ndarray, mapping) -> np.ndarray:
    from sampling.mapping import dihedral_deg, wrap_deg
    cv = lambda n: wrap_deg(dihedral_deg(np.asarray(R, np.float64),
                                         mapping.cvs[n].bead_indices) + mapping.cvs[n].shift_deg)
    phi, psi = cv("phi"), cv("psi")
    region = np.full(len(R), "other", dtype=object)
    region[(phi > -180) & (phi < -20) & ((psi > 90) | (psi < -150))] = "beta"
    region[(phi > -160) & (phi < -20) & (psi > -120) & (psi < 50)] = "alphaR"
    region[(phi > 20) & (phi < 100) & (psi > -20) & (psi < 100)] = "alphaL"
    region[(phi > -15) & (phi < 15)] = "alphaL_corridor"
    # The corridor is the dividing surface, not a basin; fold it into `other` for population
    # bookkeeping but keep phi/psi so it can be recovered.
    region[region == "alphaL_corridor"] = "other"
    return region, phi, psi


def load_replicas(paths, discard_frac: float, max_bond: float):
    """Same equilibration discard and dissociation rejection as md/analyze_fes_tica_vs_reference.py."""
    keep, per = [], []
    for p in sorted(paths):
        d = np.load(p)
        R = np.asarray(d["R"], np.float64)
        n_total = len(R)
        R = R[int(n_total * discard_frac):]
        bond = np.linalg.norm(R[:, 1:, :] - R[:, :-1, :], axis=-1).max(axis=-1)
        ok = bond <= max_bond
        per.append(dict(file=Path(p).name, frames_total=int(n_total),
                        frames_after_discard=int(len(R)),
                        frames_dissociated=int((~ok).sum()), frames_kept=int(ok.sum())))
        keep.append(R[ok])
    return np.concatenate(keep, axis=0), per


# ---------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------
def hist2d_density(u, edges):
    H, _, _ = np.histogram2d(u[:, 0], u[:, 1], bins=[edges, edges])
    area = (edges[1] - edges[0]) ** 2
    return H / (H.sum() * area)


def kl_hist(p, q, area, floor=1e-12):
    """KL(p||q) over a shared grid. Cells where p is empty contribute nothing."""
    m = p > 0
    return float(np.sum(p[m] * np.log(np.maximum(p[m], floor) / np.maximum(q[m], floor))) * area)


def gaussian_2d_on_grid(edges):
    c = 0.5 * (edges[1:] + edges[:-1])
    X, Y = np.meshgrid(c, c, indexing="ij")
    return np.exp(-0.5 * (X ** 2 + Y ** 2)) / (2.0 * np.pi)


def cov_stats(u):
    mu = u.mean(axis=0)
    C = np.cov(u.T)
    ev = np.linalg.eigvalsh(C)
    return dict(mean=mu.tolist(), cov=C.tolist(),
                eigenvalues=ev.tolist(),
                generalized_variance=float(np.linalg.det(C)),
                anisotropy=float(ev.max() / max(ev.min(), 1e-12)),
                mean_norm=float(np.linalg.norm(mu)))


def profile_along(u, e, edges):
    s = u @ e
    h, _ = np.histogram(s, bins=edges, density=True)
    return s, h


def free_energy(hist, kT, floor=1e-9):
    F = -kT * np.log(np.maximum(hist, floor))
    return F - np.nanmin(F[np.isfinite(F)])


def find_saddle(centers, F, lo, hi):
    """Highest free energy (density minimum) strictly between two mode positions."""
    m = (centers > lo) & (centers < hi)
    if m.sum() < 3:
        return float("nan"), float("nan")
    i = np.argmax(F[m])
    return float(centers[m][i]), float(F[m][i])


# ---------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--bias-npz", type=Path, required=True)
    ap.add_argument("--flow-dir", type=Path, required=True)
    ap.add_argument("--flow-primary", default="small_seed0")
    ap.add_argument("--flow-seed", action="append", default=None,
                    help="flow seed suffix, repeatable; default: six small_seed variants")
    ap.add_argument("--flow-crosscheck", default="base_seed0")
    ap.add_argument("--ensemble", action="append", required=True,
                    help="LABEL=glob  (repeatable)")
    ap.add_argument("--discard-frac", type=float, default=0.20)
    ap.add_argument("--max-bond", type=float, default=3.0)
    ap.add_argument("--mapping", default="ala2_backbone_cb_6")
    ap.add_argument("--grid-lim", type=float, default=4.5, help="latent plot/grid half-width")
    ap.add_argument("--bins", type=int, default=80)
    add_project_root_argument(ap)
    a = ap.parse_args()
    project_root = repo_root(a.project_root)
    a.outdir = resolve_output(a.outdir, base=project_root)
    reference = resolve_input(a.reference, base=project_root, label="reference")
    bias_npz = resolve_input(a.bias_npz, base=project_root, label="bias artifact")
    flow_dir = resolve_input(a.flow_dir, base=project_root, label="flow directory")
    flow_primary = str(a.flow_primary)
    flow_seeds = list(a.flow_seed or [f"small_seed{s}" for s in range(6)])
    if flow_primary not in flow_seeds:
        flow_seeds.insert(0, flow_primary)
    flow_crosscheck = str(a.flow_crosscheck)

    from sampling.biases.tica_regional import SmoothTICABias
    from sampling.flow_density import load_flow, to_latent, log_prob
    from sampling.mapping import get_mapping

    mapping = get_mapping(a.mapping)
    bias = SmoothTICABias.load(bias_npz)
    kT = float(bias.kbt_kcal_mol)
    _log(f"[setup] kT = {kT:.6f} kcal/mol; TICA {bias.projection.coefficients.shape} "
         f"({len(bias.projection.pairs)} pair distances -> "
         f"{bias.projection.coefficients.shape[1]} TICs)")

    flows = {}
    for tag in dict.fromkeys(flow_seeds + [flow_crosscheck]):
        p = flow_dir / f"flow_{tag}.npz"
        if p.exists():
            flows[tag] = load_flow(p)
    if flow_primary not in flows:
        raise SystemExit(f"primary flow {flow_primary} not found in {flow_dir}")
    _log(f"[setup] flows loaded: {sorted(flows)} (primary {flow_primary})")

    n_dims = flows[flow_primary][1].n_dims

    def latent(R, tag=flow_primary):
        z = np.asarray(bias.projection.transform(R), np.float64)[:, :n_dims]
        params, cfg = flows[tag]
        return np.asarray(to_latent(params, cfg, z), np.float64), z

    # ---------------- reference ----------------
    _log("[load] reference")
    R_ref = np.asarray(np.load(reference, allow_pickle=False)["R"], np.float64)
    u_ref, z_ref = latent(R_ref)
    reg_ref, phi_ref, psi_ref = assign_basins(R_ref, mapping)
    _log(f"[load] reference {len(R_ref)} frames -> u mean {u_ref.mean(0)} "
         f"sd {u_ref.std(0)}")

    # ---------------- model ensembles ----------------
    ens = {}
    for spec in a.ensemble:
        label, _, pattern = str(spec).partition("=")
        try:
            files = resolve_glob(pattern, base=project_root, label=f"ensemble {label}")
        except FileNotFoundError:
            files = []
        if not files:
            _log(f"[skip] {label}: no files match {pattern}")
            continue
        R, per = load_replicas(files, a.discard_frac, a.max_bond)
        u, z = latent(R)
        reg, phi, psi = assign_basins(R, mapping)
        ens[label] = dict(R=R, u=u, z=z, region=reg, phi=phi, psi=psi, per_replica=per,
                          n_files=len(files))
        _log(f"[load] {label}: {len(files)} replicas, {len(R)} frames kept")
    if not ens:
        raise SystemExit("no model ensembles loaded")

    edges = np.linspace(-a.grid_lim, a.grid_lim, a.bins + 1)
    area = (edges[1] - edges[0]) ** 2
    centers = 0.5 * (edges[1:] + edges[:-1])
    p_ref = hist2d_density(u_ref, edges)
    p_gauss = gaussian_2d_on_grid(edges)

    summary = {
        "provenance": {
            "bias_npz": str(bias_npz), "reference": str(reference),
            "flow_dir": str(flow_dir), "flow_primary": flow_primary,
            "flow_seeds": [t for t in flow_seeds if t in flows],
            "flow_crosscheck": flow_crosscheck if flow_crosscheck in flows else None,
            "n_dims": int(n_dims), "kT_kcal_mol": kT,
            "discard_frac": a.discard_frac, "max_bond_A": a.max_bond,
            "basin_definitions": "md/analyze_model_residuals_by_region.py (verbatim)",
        },
        "flow_misfit_floor": {}, "ensembles": {}, "mechanism": {},
    }

    # ---------------- flow misfit floor ----------------
    # How far is the EMPIRICAL reference latent from the analytic N(0,I) it is supposed to be?
    # Nothing below this level in the model numbers is interpretable.
    floor = cov_stats(u_ref)
    floor["kl_to_gaussian"] = kl_hist(p_ref, p_gauss, area)
    floor["seed_spread_kl"] = {}
    for tag in flow_seeds:
        if tag in flows and tag != FLOW_PRIMARY:
            u_s, _ = latent(R_ref, tag)
            floor["seed_spread_kl"][tag] = kl_hist(hist2d_density(u_s, edges), p_gauss, area)
    if flow_crosscheck in flows:
        u_c, _ = latent(R_ref, flow_crosscheck)
        floor["crosscheck_kl"] = {flow_crosscheck:
                                  kl_hist(hist2d_density(u_c, edges), p_gauss, area)}
    summary["flow_misfit_floor"] = floor
    _log(f"[floor] reference latent mean |mu| = {floor['mean_norm']:.4f}, "
         f"KL(ref||N(0,I)) = {floor['kl_to_gaussian']:.4f} nats")

    # ---------------- reference basin geometry (defines the beta->alphaR axis) ----------
    ref_basin = {}
    for b in BASINS:
        m = reg_ref == b
        if m.sum() < 50:
            continue
        ref_basin[b] = cov_stats(u_ref[m])
        ref_basin[b]["pct"] = float(100.0 * m.mean())
    mu_b = np.asarray(ref_basin["beta"]["mean"])
    mu_a = np.asarray(ref_basin["alphaR"]["mean"])
    axis = (mu_a - mu_b)
    axis_len = float(np.linalg.norm(axis))
    e_axis = axis / axis_len
    summary["mechanism"]["axis"] = {
        "beta_centroid_u": mu_b.tolist(), "alphaR_centroid_u": mu_a.tolist(),
        "separation": axis_len, "unit_vector": e_axis.tolist(),
        "note": ("beta->alphaR axis defined by REFERENCE centroids so that every ensemble is "
                 "projected onto the same fixed direction"),
    }
    _log(f"[axis] beta {mu_b.round(3)} -> alphaR {mu_a.round(3)}, separation {axis_len:.3f}")

    # 1D profiles along the axis. The reference marginal along ANY unit direction is exactly
    # N(0,1) if the flow were perfect, so the analytic curve is the noiseless baseline.
    s_edges = np.linspace(-a.grid_lim, a.grid_lim, 121)
    s_centers = 0.5 * (s_edges[1:] + s_edges[:-1])
    s_ref, h_ref = profile_along(u_ref, e_axis, s_edges)
    h_analytic = np.exp(-0.5 * s_centers ** 2) / np.sqrt(2.0 * np.pi)
    F_ref, F_analytic = free_energy(h_ref, kT), free_energy(h_analytic, kT)
    s_b, s_a = float(mu_b @ e_axis), float(mu_a @ e_axis)
    sad_ref = find_saddle(s_centers, F_ref, min(s_b, s_a), max(s_b, s_a))
    summary["mechanism"]["reference_profile"] = {
        "beta_mode_s": s_b, "alphaR_mode_s": s_a,
        "saddle_s": sad_ref[0], "saddle_F": sad_ref[1],
    }

    profiles = {"reference": (s_ref, h_ref, F_ref)}

    # ---------------- per-ensemble ----------------
    for label, e in ens.items():
        u = e["u"]
        rec = {"n_frames": int(len(u)), "n_replicas": e["n_files"],
               "per_replica": e["per_replica"], "global": cov_stats(u)}
        p_mod = hist2d_density(u, edges)
        rec["kl_to_gaussian"] = kl_hist(p_mod, p_gauss, area)
        rec["kl_to_reference"] = kl_hist(p_mod, p_ref, area)
        rec["excess_over_floor"] = rec["kl_to_gaussian"] - floor["kl_to_gaussian"]

        rec["basins"] = {}
        for b in BASINS:
            m = e["region"] == b
            entry = {"pct": float(100.0 * m.mean()),
                     "pct_reference": ref_basin.get(b, {}).get("pct", float("nan"))}
            if m.sum() >= 50:
                cs = cov_stats(u[m])
                entry.update(cs)
                if b in ref_basin:
                    d = np.asarray(cs["mean"]) - np.asarray(ref_basin[b]["mean"])
                    entry["centroid_shift"] = float(np.linalg.norm(d))
                    entry["centroid_shift_along_axis"] = float(d @ e_axis)
                    entry["gv_ratio_model_over_ref"] = float(
                        cs["generalized_variance"] / ref_basin[b]["generalized_variance"])
            rec["basins"][b] = entry

        s_m, h_m = profile_along(u, e_axis, s_edges)
        F_m = free_energy(h_m, kT)
        profiles[label] = (s_m, h_m, F_m)
        sad_m = find_saddle(s_centers, F_m, min(s_b, s_a), max(s_b, s_a))
        rec["profile"] = {
            "saddle_s": sad_m[0], "saddle_F": sad_m[1],
            "saddle_shift_vs_reference": (sad_m[0] - sad_ref[0]
                                          if np.isfinite(sad_m[0]) and np.isfinite(sad_ref[0])
                                          else float("nan")),
            "mean_s": float(s_m.mean()), "sd_s": float(s_m.std()),
            "mean_s_reference": float(s_ref.mean()), "sd_s_reference": float(s_ref.std()),
        }
        summary["ensembles"][label] = rec
        _log(f"[{label}] KL(model||N) {rec['kl_to_gaussian']:.4f} "
             f"(floor {floor['kl_to_gaussian']:.4f}, excess {rec['excess_over_floor']:+.4f}); "
             f"beta {rec['basins']['beta']['pct']:.1f}% alphaR {rec['basins']['alphaR']['pct']:.1f}%")

    # ---------------- figures ----------------
    make_figures(a.outdir, edges, centers, u_ref, reg_ref, ens, ref_basin, p_ref, p_gauss,
                 s_centers, profiles, h_analytic, F_analytic, e_axis, s_b, s_a, kT, summary)

    (a.outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    manifest = write_manifest(
        a.outdir,
        inputs={"reference": reference, "bias_npz": bias_npz, "flow_dir": flow_dir,
                "ensembles": [str(item) for item in a.ensemble]},
        parameters=vars(a),
        module="analysis.latent.diagnosis",
        extra={"summary": str(a.outdir / "summary.json")},
    )
    _log(f"\nwrote {a.outdir}/summary.json and 6 figures; manifest={manifest}")


# ---------------------------------------------------------------------------------------
def make_figures(outdir, edges, centers, u_ref, reg_ref, ens, ref_basin, p_ref, p_gauss,
                 s_centers, profiles, h_analytic, F_analytic, e_axis, s_b, s_a, kT, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Ellipse

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
        "axes.grid": True, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.dpi": 150,
    })
    seq = LinearSegmentedColormap.from_list("seq", SEQ_STEPS)
    div = LinearSegmentedColormap.from_list("div", [DIV_LO, DIV_MID, DIV_HI])
    lim = edges[-1]
    labels = list(ens)

    def sigma_rings(ax):
        for r in (1, 2, 3):
            ax.add_patch(plt.Circle((0, 0), r, fill=False, ec=MUTED, lw=0.8, ls=":", zorder=5))
            ax.annotate(f"{r}$\\sigma$", (r * 0.707, r * 0.707), color=MUTED, fontsize=7,
                        ha="left", va="bottom", zorder=5)

    # ---- fig1: latent density, reference vs each model -------------------------------
    n = 1 + len(labels)
    fig, axes = plt.subplots(1, n, figsize=(4.1 * n, 4.0), constrained_layout=True)
    axes = np.atleast_1d(axes)
    vmax = max([p_ref.max()] + [hist2d_density(ens[l]["u"], edges).max() for l in labels])
    for ax, (name, u) in zip(axes, [("AA reference", u_ref)] + [(l, ens[l]["u"]) for l in labels]):
        H = hist2d_density(u, edges)
        im = ax.pcolormesh(edges, edges, H.T, cmap=seq, vmin=0, vmax=vmax, rasterized=True)
        sigma_rings(ax)
        ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("$u_1$"); ax.set_title(name)
    axes[0].set_ylabel("$u_2$")
    fig.colorbar(im, ax=axes, shrink=0.85, label="latent density")
    fig.suptitle("Latent ensembles. A perfect model is exactly $\\mathcal{N}(0,I)$ — "
                 "the dotted rings.", fontsize=10)
    fig.savefig(outdir / "fig1_latent_density.png", bbox_inches="tight"); plt.close(fig)

    # ---- fig2: log-ratio maps (diverging, neutral midpoint) ---------------------------
    fig, axes = plt.subplots(1, len(labels) + 1, figsize=(4.1 * (len(labels) + 1), 4.0),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    panels = [("AA reference vs $\\mathcal{N}(0,I)$\n(flow misfit floor)", p_ref, p_gauss)]
    panels += [(f"{l} vs AA reference", hist2d_density(ens[l]["u"], edges), p_ref) for l in labels]
    lo = 1e-4
    for ax, (name, pa, pb) in zip(axes, panels):
        m = (pa > lo) & (pb > lo)
        Rr = np.full_like(pa, np.nan); Rr[m] = np.log(pa[m] / pb[m])
        v = np.nanpercentile(np.abs(Rr), 99) or 1.0
        im = ax.pcolormesh(edges, edges, Rr.T, cmap=div,
                           norm=TwoSlopeNorm(vcenter=0.0, vmin=-v, vmax=v), rasterized=True)
        sigma_rings(ax)
        ax.plot([s_b * e_axis[0], s_a * e_axis[0]], [s_b * e_axis[1], s_a * e_axis[1]],
                color=INK, lw=1.4, ls="--", zorder=6)
        ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("$u_1$"); ax.set_title(name, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8, label="log density ratio")
    axes[0].set_ylabel("$u_2$")
    fig.suptitle("Where mass is misplaced, weighted by reference probability. "
                 "Dashed line: the $\\beta\\rightarrow\\alpha_R$ axis.", fontsize=10)
    fig.savefig(outdir / "fig2_latent_logratio.png", bbox_inches="tight"); plt.close(fig)

    # ---- fig3: radial |u| distribution vs analytic chi_2 ------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    r_edges = np.linspace(0, lim, 60); r_c = 0.5 * (r_edges[1:] + r_edges[:-1])
    ax1.plot(r_c, r_c * np.exp(-0.5 * r_c ** 2), color=INK, lw=2.0, ls="--",
             label="analytic $\\chi_2$ (perfect flow)")
    ax1.plot(r_c, np.histogram(np.linalg.norm(u_ref, axis=1), bins=r_edges, density=True)[0],
             color=INK2, lw=2.0, label="AA reference (empirical)")
    for l, c in zip(labels, [BASIN_COLOR["beta"], BASIN_COLOR["alphaR"], BASIN_COLOR["alphaL"]]):
        ax1.plot(r_c, np.histogram(np.linalg.norm(ens[l]["u"], axis=1),
                                   bins=r_edges, density=True)[0], color=c, lw=2.0, label=l)
    ax1.set_xlabel("$|u|$  (reference typicality, $\\sigma$ units)")
    ax1.set_ylabel("density"); ax1.set_title("Radial latent profile"); ax1.legend()

    ax2.plot(r_c, np.exp(-0.5 * r_c ** 2), color=INK, lw=2.0, ls="--", label="analytic")
    ax2.plot(r_c, [np.mean(np.linalg.norm(u_ref, axis=1) > x) for x in r_c],
             color=INK2, lw=2.0, label="AA reference")
    for l, c in zip(labels, [BASIN_COLOR["beta"], BASIN_COLOR["alphaR"], BASIN_COLOR["alphaL"]]):
        ax2.plot(r_c, [np.mean(np.linalg.norm(ens[l]["u"], axis=1) > x) for x in r_c],
                 color=c, lw=2.0, label=l)
    ax2.set_yscale("log"); ax2.set_ylim(1e-5, 1.2)
    ax2.set_xlabel("$|u|$"); ax2.set_ylabel("$P(|u| > x)$")
    ax2.set_title("Tail occupancy — exact for $d=2$: $P=e^{-x^2/2}$"); ax2.legend()
    fig.savefig(outdir / "fig3_radial.png", bbox_inches="tight"); plt.close(fig)

    # ---- fig4: per-basin centroids + covariance ellipses ------------------------------
    fig, axes = plt.subplots(1, len(labels) + 1, figsize=(4.1 * (len(labels) + 1), 4.2),
                             constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    def draw(ax, stats, filled):
        for b in BASINS:
            if b not in stats or "cov" not in stats[b]:
                continue

            mu = np.asarray(stats[b]["mean"]); C = np.asarray(stats[b]["cov"])
            ev, evec = np.linalg.eigh(C)
            ang = np.degrees(np.arctan2(evec[1, -1], evec[0, -1]))
            for k in (1, 2):
                ax.add_patch(Ellipse(mu, 2 * k * np.sqrt(ev[-1]), 2 * k * np.sqrt(ev[0]),
                                     angle=ang, fc=BASIN_COLOR[b] if filled else "none",
                                     alpha=0.16 if filled else 1.0,
                                     ec=BASIN_COLOR[b], lw=1.6, ls="-" if filled else "--"))
            ax.plot(*mu, BASIN_MARKER[b], color=BASIN_COLOR[b], ms=9, mec=SURFACE, mew=1.5,
                    label=f"{b}  {stats[b]['pct']:.1f}%")

    draw(axes[0], ref_basin, True)
    axes[0].set_title("AA reference"); axes[0].legend(loc="upper left", fontsize=8)
    for ax, l in zip(axes[1:], labels):
        draw(ax, ref_basin, False)
        draw(ax, summary["ensembles"][l]["basins"], True)
        ax.set_title(f"{l}\n(dashed = reference)"); ax.legend(loc="upper left", fontsize=8)
    for ax in axes:
        sigma_rings(ax); ax.set_aspect("equal")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xlabel("$u_1$")
    axes[0].set_ylabel("$u_2$")
    fig.suptitle("Basin geometry in latent coordinates: 1$\\sigma$/2$\\sigma$ covariance "
                 "ellipses and centroids", fontsize=10)
    fig.savefig(outdir / "fig4_basin_geometry.png", bbox_inches="tight"); plt.close(fig)

    # ---- fig5: THE MECHANISM PLOT -- profile along the beta->alphaR axis --------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    ax1.plot(s_centers, h_analytic, color=INK, lw=2.0, ls="--",
             label="analytic $\\mathcal{N}(0,1)$")
    ax1.plot(s_centers, profiles["reference"][1], color=INK2, lw=2.0, label="AA reference")
    for l, c in zip(labels, [BASIN_COLOR["beta"], BASIN_COLOR["alphaR"], BASIN_COLOR["alphaL"]]):
        ax1.plot(s_centers, profiles[l][1], color=c, lw=2.0, label=l)
    for pos, nm, c in ((s_b, "$\\beta$", BASIN_COLOR["beta"]),
                       (s_a, "$\\alpha_R$", BASIN_COLOR["alphaR"])):
        ax1.axvline(pos, color=c, lw=1.0, ls=":")
        ax1.annotate(nm, (pos, ax1.get_ylim()[1]), color=c, ha="center", va="top", fontsize=9)
    ax1.set_xlabel("$s = u\\cdot e_{\\beta\\rightarrow\\alpha_R}$"); ax1.set_ylabel("density")
    ax1.set_title("Projection onto the $\\beta\\rightarrow\\alpha_R$ axis"); ax1.legend(fontsize=8)

    ax2.plot(s_centers, F_analytic, color=INK, lw=2.0, ls="--", label="analytic $\\frac{1}{2}kTs^2$")
    ax2.plot(s_centers, profiles["reference"][2], color=INK2, lw=2.0, label="AA reference")
    for l, c in zip(labels, [BASIN_COLOR["beta"], BASIN_COLOR["alphaR"], BASIN_COLOR["alphaL"]]):
        ax2.plot(s_centers, profiles[l][2], color=c, lw=2.0, label=l)
    for pos, c in ((s_b, BASIN_COLOR["beta"]), (s_a, BASIN_COLOR["alphaR"])):
        ax2.axvline(pos, color=c, lw=1.0, ls=":")
    ax2.set_ylim(0, 5.0)
    ax2.set_xlabel("$s$"); ax2.set_ylabel("free energy  (kcal/mol)")
    ax2.set_title("Free energy along the axis — reference side is ANALYTIC")
    ax2.legend(fontsize=8)
    # NOTE: an earlier version of this caption framed the panel as a "saddle shift" test. That
    # reading is RETRACTED -- this very figure disproves it: the AA reference (grey) is
    # essentially UNIMODAL along the axis, tracking the analytic N(0,1), so there is no genuine
    # double well and "the saddle" was the argmax of a flat parabola bottom (it swung from +0.41
    # to +1.09 across flow checkpoints). See FINDINGS.md and sharpness.py.
    fig.suptitle("The reference (grey) is UNIMODAL here — there is no double well, so there is "
                 "no saddle to shift.\nOver-sharpening shows up as a narrow SPIKE on a depleted "
                 "$\\beta$ shoulder, not as a moved boundary.", fontsize=9.5)
    fig.savefig(outdir / "fig5_mechanism_axis.png", bbox_inches="tight"); plt.close(fig)

    # ---- fig6: basin populations ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 4.0), constrained_layout=True)
    x = np.arange(len(BASINS)); w = 0.8 / (len(labels) + 1)
    ax.bar(x, [ref_basin.get(b, {}).get("pct", 0.0) for b in BASINS], w,
           color=INK2, label="AA reference")
    for i, l in enumerate(labels):
        ax.bar(x + (i + 1) * w, [summary["ensembles"][l]["basins"][b]["pct"] for b in BASINS],
               w, color=BASIN_COLOR[BASINS[i % len(BASINS)]], label=l)
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(BASINS)
    ax.set_ylabel("population (%)"); ax.set_title("Basin populations — the quantity being explained")
    ax.legend()
    fig.savefig(outdir / "fig6_populations.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
