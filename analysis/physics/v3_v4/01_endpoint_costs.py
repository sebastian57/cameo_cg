"""Measure actual +/- displacement costs from collected mean-force labels.

This is a local line-integral diagnostic, not a replacement for a free-energy calculation.
With only anchor and endpoint labels it is a two-point trapezoid estimate; intermediate control
states make the estimate less sensitive to path curvature.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analysis.physics.v3_v4.common import (
    available_label_rows,
    bootstrap_mean,
    load_label_table,
    load_state_table,
    signed_path_cost,
    write_json,
)


def _label_positions(labels: dict[str, np.ndarray]) -> dict[int, int]:
    state = np.asarray(labels["state"], dtype=np.int64)
    if len(np.unique(state)) != len(state):
        raise ValueError("duplicate label state IDs")
    return {int(value): i for i, value in enumerate(state)}


def collect_paths(
    states: dict[str, np.ndarray], labels: dict[str, np.ndarray]
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    """Collect labeled anchor and signed-radius points by anchor/direction."""
    positions = _label_positions(labels)
    state_ids = np.asarray(states["state"], dtype=np.int64)
    groups: dict[int, dict[str, Any]] = {}
    anchor_rows: dict[int, dict[str, Any]] = {}
    missing = 0

    for i, state_id in enumerate(state_ids):
        label_i = positions.get(int(state_id))
        if label_i is None:
            missing += 1
            continue
        anchor = int(states["anchor"][i])
        direction = int(states["direction"][i])
        point = {
            "state": int(state_id),
            "row": i,
            "label_row": label_i,
            "R": np.asarray(labels["R"][label_i], dtype=np.float64),
            "F": np.asarray(labels["F"][label_i], dtype=np.float64),
            "multiplier": float(states.get("multiplier", np.zeros(len(state_ids)))[i]),
            "eps": float(states.get("eps_state", np.zeros(len(state_ids)))[i]),
            "lam": float(states.get("lam_state", np.zeros(len(state_ids)))[i]),
            "kind": int(states.get("kind", np.zeros(len(state_ids), dtype=np.int8))[i]),
        }
        if direction < 0:
            anchor_rows[anchor] = point
            continue
        groups.setdefault(anchor, {}).setdefault(direction, []).append(point)

    for anchor, directions in groups.items():
        directions["anchor"] = anchor_rows.get(anchor)
    return groups, {"n_states": len(state_ids), "n_labeled": len(positions), "n_missing": missing}


def _pair_rows(points: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    positive: dict[float, dict[str, Any]] = {}
    negative: dict[float, dict[str, Any]] = {}
    for point in points:
        multiplier = float(point["multiplier"])
        if multiplier == 0:
            continue
        target = positive if multiplier > 0 else negative
        target.setdefault(round(abs(multiplier), 7), point)
    return [(positive[radius], negative[radius]) for radius in sorted(set(positive) & set(negative))]


def _path(anchor: dict[str, Any], endpoint: dict[str, Any], points: list[dict[str, Any]]) -> float:
    sign = 1.0 if endpoint["multiplier"] > 0 else -1.0
    selected = [point for point in points if point["multiplier"] * sign > 0]
    selected.sort(key=lambda point: abs(point["multiplier"]))
    R = [anchor["R"]] + [point["R"] for point in selected]
    F = [anchor["F"]] + [point["F"] for point in selected]
    return signed_path_cost(np.asarray(R), np.asarray(F))


def summarize_endpoint_costs(
    states: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    n_boot: int = 2000,
    seed: int = 20260821,
    eps_max: float = 0.16,
) -> dict[str, Any]:
    groups, counts = collect_paths(states, labels)
    rows: list[dict[str, Any]] = []
    incomplete = 0
    for anchor_id, directions in groups.items():
        anchor = directions.get("anchor")
        if anchor is None:
            incomplete += sum(1 for key in directions if key != "anchor")
            continue
        for direction, points in directions.items():
            if direction == "anchor":
                continue
            pairs = _pair_rows(points)
            if not pairs:
                incomplete += 1
                continue
            for plus, minus in pairs:
                dplus = _path(anchor, plus, points)
                dminus = _path(anchor, minus, points)
                lam = float(plus["lam"] or minus["lam"])
                eps = float(plus["eps"] or minus["eps"])
                radius = abs(float(plus["multiplier"]))
                nominal = 0.5 * lam * (radius * eps) ** 2
                rows.append(
                    {
                        "anchor": anchor_id,
                        "direction": direction,
                        "radius": radius,
                        "dA_plus": dplus,
                        "dA_minus": dminus,
                        "asymmetry": dplus - dminus,
                        "nominal_quadratic": nominal,
                        "lambda": lam,
                        "eps": eps,
                        "kind": int(plus["kind"]),
                        "at_eps_cap": bool(eps >= eps_max - 1e-8),
                    }
                )

    asym = np.asarray([row["asymmetry"] for row in rows], dtype=np.float64)
    dplus = np.asarray([row["dA_plus"] for row in rows], dtype=np.float64)
    dminus = np.asarray([row["dA_minus"] for row in rows], dtype=np.float64)
    mean_asym, lo_asym, hi_asym = bootstrap_mean(asym, n_boot, seed)
    _, lo_plus, hi_plus = bootstrap_mean(dplus, n_boot, seed + 1)
    _, lo_minus, hi_minus = bootstrap_mean(dminus, n_boot, seed + 2)
    result = {
        **counts,
        "n_complete_pairs": len(rows),
        "n_incomplete_pairs": incomplete,
        "quadrature": "multi_point_trapezoid" if any(row["radius"] != 1.0 for row in rows) else "two_point_trapezoid",
        "median_asymmetry_kcal_mol": float(np.median(asym)) if len(asym) else float("nan"),
        "mean_asymmetry_kcal_mol": mean_asym,
        "asymmetry_bootstrap_95pct": [lo_asym, hi_asym],
        "mean_dA_plus_kcal_mol": float(dplus.mean()) if len(dplus) else float("nan"),
        "dA_plus_bootstrap_95pct": [lo_plus, hi_plus],
        "mean_dA_minus_kcal_mol": float(dminus.mean()) if len(dminus) else float("nan"),
        "dA_minus_bootstrap_95pct": [lo_minus, hi_minus],
        "n_at_eps_cap": int(sum(row["at_eps_cap"] for row in rows)),
        "bootstrap_seed": seed,
        "bootstrap_replicates": n_boot,
    }
    result["rows"] = rows
    return result


def _write_outputs(outdir: Path, result: dict[str, Any], states_path: Path, labels_path: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = result.pop("rows")
    fields = [
        "anchor", "direction", "radius", "dA_plus", "dA_minus", "asymmetry",
        "nominal_quadratic", "lambda", "eps", "kind", "at_eps_cap",
    ]
    with (outdir / "endpoint_costs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        np.savez_compressed(outdir / "endpoint_costs.npz", **{field: np.asarray([row[field] for row in rows]) for field in fields})
    result["states_path"] = str(states_path)
    result["labels_path"] = str(labels_path)
    write_json(outdir / "summary.json", result)
    print(f"complete +/- pairs: {result['n_complete_pairs']}")
    print(f"incomplete pairs:    {result['n_incomplete_pairs']}")
    print(f"median asymmetry:    {result['median_asymmetry_kcal_mol']:+.4f} kcal/mol")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--eps-max", type=float, default=0.16)
    args = parser.parse_args()
    states = load_state_table(args.states)
    labels = load_label_table(args.labels)
    result = summarize_endpoint_costs(states, labels, args.n_boot, args.seed, args.eps_max)
    _write_outputs(args.outdir, result, args.states, args.labels)


if __name__ == "__main__":
    main()
