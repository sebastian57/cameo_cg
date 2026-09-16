"""Aligned static-energy, reference-FES, and model-MD-FES analysis."""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.landscapes.landscape import (
    aggregate_static,
    assign_basins,
    basin_thermodynamics,
    fit_landscape,
    free_energy,
    histogram2d_periodic,
    reference_zero_cell,
)


KB_KCAL = 0.0019872042586


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path.name}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _basin_free_energy(regions: np.ndarray, basin: str, kT: float) -> float:
    beta_count = int(np.sum(regions == "beta"))
    basin_count = int(np.sum(regions == basin))
    if beta_count == 0 or basin_count == 0:
        raise ValueError(f"cannot compute dF for empty basin {basin} or beta")
    return float(-kT * np.log(basin_count / beta_count))


def bootstrap_basin_gaps(
    label: str, regions: np.ndarray, energies: np.ndarray, replicates: int, seed: int,
) -> list[dict[str, Any]]:
    """Return seeded within-basin bootstrap intervals for both diagnostic gaps."""
    regions = np.asarray(regions).astype(str)
    energies = np.asarray(energies, dtype=np.float64)
    rng = np.random.default_rng(seed)
    beta = energies[regions == "beta"]
    if beta.size == 0:
        raise ValueError("reference contains no beta frames")
    rows = []
    for basin in ("alphaR", "alphaL"):
        values = energies[regions == basin]
        if values.size == 0:
            raise ValueError(f"reference contains no {basin} frames")
        observed = float(values.mean() - beta.mean())
        if replicates:
            draws = np.empty(replicates, dtype=np.float64)
            for index in range(replicates):
                draws[index] = (rng.choice(values, size=len(values), replace=True).mean()
                                - rng.choice(beta, size=len(beta), replace=True).mean())
            low, high = np.percentile(draws, [2.5, 97.5])
        else:
            low = high = float("nan")
        rows.append({"model": label, "basin": basin, "dU_vs_beta": observed,
                     "bootstrap_replicates": int(replicates),
                     "ci_low": float(low), "ci_high": float(high)})
    return rows


def matched_pair_metrics(
    pair_label: str, pre: str, post: str, regions: np.ndarray,
    energies: dict[str, np.ndarray], static_maps: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Compute post-minus-pre static-map and raw-frame basin-gap changes."""
    if pre not in energies or post not in energies:
        raise ValueError(f"pair {pair_label} references missing model {pre!r} or {post!r}")
    pre_map = np.asarray(static_maps[pre], dtype=np.float64)
    post_map = np.asarray(static_maps[post], dtype=np.float64)
    common = np.isfinite(pre_map) & np.isfinite(post_map)
    delta = np.full(pre_map.shape, np.nan, dtype=np.float64)
    delta[common] = post_map[common] - pre_map[common]
    regions = np.asarray(regions).astype(str)
    rows = []
    for basin in ("alphaR", "alphaL"):
        gaps = {}
        for name in (pre, post):
            energy = np.asarray(energies[name], dtype=np.float64)
            gaps[name] = float(energy[regions == basin].mean()
                               - energy[regions == "beta"].mean())
        rows.append({"pair": pair_label, "pre": pre, "post": post, "basin": basin,
                     "pre_dU": gaps[pre], "post_dU": gaps[post],
                     "delta_dU": gaps[post] - gaps[pre]})
    return rows, delta


def build_leaderboard(
    models: list[Any], correlations: list[dict[str, Any]],
    basins: list[dict[str, Any]], bootstrap: list[dict[str, Any]],
    frame_count: int, screen_version: str, *, evaluation_dataset: str = "",
    mapping: str = "", grid: str = "", evaluation_date: str = "",
    md_provenance: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build transparent static and MD ranks without a combined score."""
    corr_by_model = {row["model"]: row for row in correlations
                     if row["target"] == "reference"}
    basin_by_model = {(row["model"], row["basin"]): row for row in basins
                      if row["basin"] in ("alphaR", "alphaL")}
    boot_by_model = {(row["model"], row["basin"]): row for row in bootstrap}
    rows = []
    md_provenance = md_provenance or {}
    for model in models:
        model_md = md_provenance.get(model.label, {})
        if model_md.get("eligible_for_md_rank"):
            replicas = ",".join(
                str(item) for item in model_md.get("source_replica_ids", [])
            )
            md_provenance_text = (
                f"path={model_md.get('ensemble_path', '')}; "
                f"replicas={model_md.get('declared_replicas', '')}; "
                f"source_replica_ids={replicas}; "
                f"provenance=provenance.json#md_ensembles/{model.label}"
            )
        else:
            md_provenance_text = model_md.get("reason", "not evaluated")
        corr = corr_by_model[model.label]
        alpha_r = basin_by_model[(model.label, "alphaR")]
        alpha_l = basin_by_model[(model.label, "alphaL")]
        errors = [alpha_r["dF_md_error"], alpha_l["dF_md_error"]]
        has_md = all(value != "" and np.isfinite(float(value)) for value in errors)
        rms = (float(np.sqrt(np.mean(np.square(np.asarray(errors, dtype=float)))))
               if has_md else float("nan"))
        br = boot_by_model.get((model.label, "alphaR"), {})
        bl = boot_by_model.get((model.label, "alphaL"), {})
        rows.append({"model": model.label, "checkpoint": str(getattr(model, "checkpoint", "")),
                     "checkpoint_key": getattr(model, "checkpoint_key", ""), "config": str(getattr(model, "config", "")),
                     "screen_version": screen_version, "evaluation_frames": int(frame_count),
                     "evaluation_dataset": evaluation_dataset, "mapping": mapping,
                     "grid": grid, "evaluation_date": evaluation_date,
                     "lineage": getattr(model, "lineage", ""),
                     "optimization_dataset": getattr(model, "optimization_dataset", ""),
                     "stability_ood": getattr(model, "stability_ood", "not evaluated"),
                     "md_provenance": md_provenance_text,
                     "static_pearson_r": corr["pearson_r"],
                     "static_spearman_r": corr["spearman_r"],
                     "dU_alphaR_minus_beta": alpha_r["dU_vs_beta"],
                     "dU_alphaR_ci_low": br.get("ci_low", ""),
                     "dU_alphaR_ci_high": br.get("ci_high", ""),
                     "dU_alphaL_minus_beta": alpha_l["dU_vs_beta"],
                     "dU_alphaL_ci_low": bl.get("ci_low", ""),
                     "dU_alphaL_ci_high": bl.get("ci_high", ""),
                     "md_dF_alphaR_error": alpha_r["dF_md_error"] if has_md else "",
                     "md_dF_alphaL_error": alpha_l["dF_md_error"] if has_md else "",
                     "md_dF_rms_error": rms if has_md else "",
                     "md_status": "evaluated" if has_md else "not evaluated",
                     "static_rank": "", "md_rank": ""})
    static_order = sorted(range(len(rows)), key=lambda i: (
        not np.isfinite(float(rows[i]["static_pearson_r"])),
        -float(rows[i]["static_pearson_r"])
        if np.isfinite(float(rows[i]["static_pearson_r"])) else 0.0))
    for rank, index in enumerate(static_order, 1):
        rows[index]["static_rank"] = rank
    md_order = sorted((i for i, row in enumerate(rows)
                       if row["md_status"] == "evaluated"),
                      key=lambda i: float(rows[i]["md_dF_rms_error"]))
    for rank, index in enumerate(md_order, 1):
        rows[index]["md_rank"] = rank
    return rows


def analyze_arrays(
    config: Any,
    reference_phi: np.ndarray,
    reference_psi: np.ndarray,
    energies: dict[str, np.ndarray],
    md_cvs: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    md_provenance: dict[str, dict[str, Any]] | None = None,
    evaluation_timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Analyze already projected arrays and write the complete result bundle."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [model.label for model in config.models]
    if set(labels) != set(energies) or not set(md_cvs).issubset(labels):
        raise ValueError("energy keys must match models and MD-CV keys must be a subset")
    phi = np.asarray(reference_phi, dtype=np.float64)
    psi = np.asarray(reference_psi, dtype=np.float64)
    if phi.shape != psi.shape:
        raise ValueError("reference phi and psi must have the same shape")
    kT = KB_KCAL * float(config.temperature)
    bins = int(config.profile.bins)
    min_count = int(config.profile.min_count)
    zero_cell = reference_zero_cell(phi, psi, bins)
    reference_regions = assign_basins(phi, psi)
    reference_counts, edges = histogram2d_periodic(phi, psi, bins)
    reference_fes = free_energy(reference_counts, kT, zero_cell)

    arrays: dict[str, np.ndarray] = {
        "edges": edges,
        "reference_counts": reference_counts,
        "reference_fes": reference_fes,
        "zero_cell": np.asarray(zero_cell, dtype=np.int64),
    }
    cell_rows: list[dict[str, Any]] = []
    basin_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    model_maps: dict[str, dict[str, Any]] = {}
    centers = 0.5 * (edges[:-1] + edges[1:])

    for label in labels:
        energy = np.asarray(energies[label], dtype=np.float64)
        if energy.shape != phi.shape or not np.isfinite(energy).all():
            raise ValueError(f"energy array for {label} is invalid")
        static = aggregate_static(phi, psi, energy, bins, min_count, zero_cell)
        has_md = label in md_cvs and md_cvs[label] is not None
        if has_md:
            md_phi, md_psi = (np.asarray(part, dtype=np.float64) for part in md_cvs[label])
            md_regions = assign_basins(md_phi, md_psi)
            md_counts, _ = histogram2d_periodic(md_phi, md_psi, bins)
            md_fes = free_energy(md_counts, kT, zero_cell)
        else:
            md_regions = None
            md_counts = np.zeros((bins, bins), dtype=np.int64)
            md_fes = np.full((bins, bins), np.nan, dtype=np.float64)
        fits: dict[str, dict[str, Any]] = {}
        for target, target_map in (("model_md", md_fes), ("reference", reference_fes)):
            common = np.isfinite(static["mean"]) & np.isfinite(target_map)
            if target == "model_md" and not has_md:
                fit = {"n_cells": 0, "pearson_r": float("nan"),
                       "spearman_r": float("nan"), "slope": float("nan"),
                       "intercept": float("nan"), "r_squared": float("nan"),
                       "mask": common, "residual": np.full(common.shape, np.nan)}
                status = "not evaluated"
            else:
                try:
                    fit = fit_landscape(static["mean"], target_map, common)
                    status = "ok"
                except ValueError as error:
                    fit = {"n_cells": int(common.sum()), "pearson_r": float("nan"),
                           "spearman_r": float("nan"), "slope": float("nan"),
                           "intercept": float("nan"), "r_squared": float("nan"),
                           "mask": common, "residual": np.full(common.shape, np.nan)}
                    status = str(error)
            fits[target] = fit
            correlation_rows.append({
                "model": label, "target": target, "status": status,
                "n_cells": fit["n_cells"], "pearson_r": fit["pearson_r"],
                "spearman_r": fit["spearman_r"], "slope": fit["slope"],
                "intercept": fit["intercept"], "r_squared": fit["r_squared"],
            })
            arrays[f"{label}__{target}_residual"] = fit["residual"]

        beta_u = float(energy[reference_regions == "beta"].mean())
        md_rows = ({str(row["basin"]): row for row in
                    basin_thermodynamics(reference_regions, energy, md_regions, kT)}
                   if has_md else {})
        for basin in ("beta", "alphaR", "alphaL"):
            ref_sel = reference_regions == basin
            row = {
                "model": label, "basin": basin,
                "reference_count": int(ref_sel.sum()),
                "md_count": md_rows.get(basin, {}).get("md_count", ""),
                "mean_U": float(energy[ref_sel].mean()),
                "dU_vs_beta": float(energy[ref_sel].mean() - beta_u),
                "dF_md_vs_beta": md_rows.get(basin, {}).get("dF_md_vs_beta", ""),
                "dF_ref_vs_beta": _basin_free_energy(reference_regions, basin, kT),
                "md_status": "ok" if has_md else "not evaluated",
            }
            row["dF_md_error"] = (float(row["dF_md_vs_beta"] - row["dF_ref_vs_beta"])
                                  if has_md else "")
            basin_rows.append(row)

        for i in range(bins):
            for j in range(bins):
                cell_rows.append({
                    "model": label, "i_phi": i, "i_psi": j,
                    "phi_center_deg": centers[i], "psi_center_deg": centers[j],
                    "count": int(static["count"][i, j]),
                    "mean_delta_U": static["mean"][i, j],
                    "median_delta_U": static["median"][i, j],
                    "std_U": static["std"][i, j], "sem_U": static["sem"][i, j],
                    "supported": bool(np.isfinite(static["mean"][i, j])),
                })
        arrays[f"{label}__static_mean"] = static["mean"]
        arrays[f"{label}__static_median"] = static["median"]
        arrays[f"{label}__static_std"] = static["std"]
        arrays[f"{label}__static_sem"] = static["sem"]
        arrays[f"{label}__static_count"] = static["count"]
        arrays[f"{label}__md_counts"] = md_counts
        arrays[f"{label}__md_fes"] = md_fes
        model_maps[label] = {"static": static["mean"], "md_fes": md_fes,
                             "fits": fits, "has_md": has_md}

    bootstrap_rows = []
    bootstrap_replicates = int(getattr(config.profile, "bootstrap_replicates", 0))
    for offset, label in enumerate(labels):
        bootstrap_rows.extend(bootstrap_basin_gaps(
            label, reference_regions, energies[label], bootstrap_replicates,
            int(getattr(config, "seed", 0)) + offset,
        ))
    pair_rows = []
    pair_maps = {}
    static_maps = {label: model_maps[label]["static"] for label in labels}
    for pair in getattr(config, "pairs", []):
        rows, delta = matched_pair_metrics(
            pair.label, pair.pre, pair.post, reference_regions, energies, static_maps
        )
        pair_rows.extend(rows)
        pair_maps[pair.label] = delta
        arrays[f"pair__{pair.label}__delta_static"] = delta
    leaderboard_rows = build_leaderboard(
        config.models, correlation_rows, basin_rows, bootstrap_rows, len(phi), "ala2-bb6-v2",
        evaluation_dataset=str(getattr(config, "reference", "")), mapping=str(getattr(config, "mapping", "")),
        grid=f"{bins}x{bins}",
        evaluation_date=(evaluation_timestamp_utc or
                         datetime.now(timezone.utc).isoformat())[:10],
        md_provenance=md_provenance,
    )

    _write_csv(output_dir / "cell_statistics.csv", cell_rows)
    _write_csv(output_dir / "basin_summary.csv", basin_rows)
    _write_csv(output_dir / "correlations.csv", correlation_rows)
    _write_csv(output_dir / "basin_bootstrap.csv", bootstrap_rows)
    if pair_rows:
        _write_csv(output_dir / "matched_pairs.csv", pair_rows)
    _write_csv(output_dir / "leaderboard.csv", leaderboard_rows)
    _write_leaderboard_fragment(output_dir / "leaderboard_fragment.md", leaderboard_rows)
    np.savez_compressed(output_dir / "landscapes.npz", **arrays)
    _plot_landscapes(config, labels, reference_fes, model_maps, output_dir)
    _plot_correlations(config, labels, model_maps, output_dir)
    if pair_maps:
        _plot_matched_pairs(config, pair_maps, output_dir)
    summary = {
        "profile": config.profile.name,
        "scientific_result": bool(config.profile.scientific_result),
        "temperature_K": float(config.temperature),
        "kT_kcal_mol": kT,
        "bins": bins,
        "min_count": min_count,
        "reference_frames": int(len(phi)),
        "zero_cell": list(zero_cell),
        "models": labels,
        "basins": basin_rows,
        "correlations": correlation_rows,
        "bootstrap": bootstrap_rows,
        "matched_pairs": pair_rows,
        "leaderboard": leaderboard_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, allow_nan=True))
    return summary


def _write_leaderboard_fragment(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ("static_rank", "model", "static_pearson_r", "dU_alphaR_minus_beta",
               "dU_alphaL_minus_beta", "md_rank", "md_status")
    lines = ["# Model leaderboard fragment", "",
             "Evaluation provenance: checkpoint, key, config, screen version, and frame count are in leaderboard.csv.", "",
             "| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in sorted(rows, key=lambda item: int(item["static_rank"])):
        values = [f"{row[column]:.6g}" if isinstance(row[column], float)
                  else str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def _plot_matched_pairs(config: Any, pair_maps: dict[str, np.ndarray], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(pair_maps), figsize=(5.0 * len(pair_maps), 4.5),
                             constrained_layout=True, squeeze=False)
    for ax, item in zip(axes[0], pair_maps.items()):
        label, delta = item
        im = ax.imshow(delta.T, origin="lower", extent=(-180, 180, -180, 180),
                       cmap="RdBu_r", vmin=-3, vmax=3, aspect="equal")
        ax.set(title=f"{label}: post - pre dU", xlabel="phi (deg)", ylabel="psi (deg)")
        fig.colorbar(im, ax=ax, label="delta dU (kcal/mol)")
    prefix = "SMOKE ONLY - " if not config.profile.scientific_result else ""
    fig.suptitle(prefix + "Matched pre/post REM energy changes")
    fig.savefig(output_dir / "matched_pair_landscapes.png", dpi=170)
    plt.close(fig)


def _plot_landscapes(config: Any, labels: list[str], reference_fes: np.ndarray,
                     model_maps: dict[str, dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(len(labels), 3, figsize=(13.5, 4.2 * len(labels)),
                             constrained_layout=True, squeeze=False)
    for row, label in enumerate(labels):
        maps = (reference_fes, model_maps[label]["static"], model_maps[label]["md_fes"])
        titles = ("Mapped-AA reference FES", f"{label} static ΔU",
                  f"{label} MD FES" if model_maps[label]["has_md"] else f"{label} MD not evaluated")
        for ax, grid, title in zip(axes[row], maps, titles):
            im = ax.imshow(grid.T, origin="lower", extent=(-180, 180, -180, 180),
                           cmap="turbo", vmin=-2, vmax=6, aspect="equal")
            ax.set(title=title, xlabel="phi (deg)", ylabel="psi (deg)")
            fig.colorbar(im, ax=ax, label="kcal/mol")
    prefix = "SMOKE ONLY — " if not config.profile.scientific_result else ""
    fig.suptitle(prefix + "MD-less learned-energy landscape comparison")
    fig.savefig(output_dir / "landscape_comparison.png", dpi=170)
    plt.close(fig)


def _plot_correlations(config: Any, labels: list[str], model_maps: dict[str, dict[str, Any]],
                       output_dir: Path) -> None:
    fig, axes = plt.subplots(len(labels), 2, figsize=(10.5, 4.2 * len(labels)),
                             constrained_layout=True, squeeze=False)
    for row, label in enumerate(labels):
        static = model_maps[label]["static"]
        md_fes = model_maps[label]["md_fes"]
        fit = model_maps[label]["fits"]["model_md"]
        mask = fit["mask"]
        ax = axes[row, 0]
        ax.scatter(static[mask], md_fes[mask], s=18, alpha=0.75)
        if np.isfinite(fit["slope"]):
            line_x = np.linspace(np.nanmin(static[mask]), np.nanmax(static[mask]), 100)
            ax.plot(line_x, fit["slope"] * line_x + fit["intercept"], color="black")
        ax.set(title=f"{label}: static ΔU vs MD FES", xlabel="static ΔU (kcal/mol)",
               ylabel="MD ΔF (kcal/mol)")
        im = axes[row, 1].imshow(fit["residual"].T, origin="lower",
                                 extent=(-180, 180, -180, 180), cmap="RdBu_r",
                                 vmin=-3, vmax=3, aspect="equal")
        axes[row, 1].set(title=f"{label}: fit residual", xlabel="phi (deg)", ylabel="psi (deg)")
        fig.colorbar(im, ax=axes[row, 1], label="kcal/mol")
    prefix = "SMOKE ONLY — " if not config.profile.scientific_result else ""
    fig.suptitle(prefix + "Static-to-MD correlations and residuals")
    fig.savefig(output_dir / "correlation_and_residuals.png", dpi=170)
    plt.close(fig)
