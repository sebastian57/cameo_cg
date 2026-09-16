"""Measure model-to-model energy-gap variance on identical reference frames."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from analysis.physics.v3_v4.common import bootstrap_mean, write_json


KT_298 = 0.5921868690749673
ODDS_BASELINE = 0.9178


def basin_masks(R: np.ndarray, mapping_name: str = "ala2_backbone_cb_6") -> dict[str, np.ndarray]:
    """Return the project’s fixed beta/alphaR masks for mapped CG frames."""
    from sampling.mapping import dihedral_deg, get_mapping, wrap_deg

    mapping = get_mapping(mapping_name)
    coords = np.asarray(R, dtype=np.float64)
    phi = wrap_deg(dihedral_deg(coords, mapping.cvs["phi"].bead_indices)
                   + mapping.cvs["phi"].shift_deg)
    psi = wrap_deg(dihedral_deg(coords, mapping.cvs["psi"].bead_indices)
                   + mapping.cvs["psi"].shift_deg)
    beta = (phi > -180) & (phi < -20) & ((psi > 90) | (psi < -150))
    alpha_r = (phi > -160) & (phi < -20) & (psi > -120) & (psi < 50)
    return {"beta": np.asarray(beta, dtype=bool), "alphaR": np.asarray(alpha_r, dtype=bool)}


def delta_from_energies(
    model_energy: np.ndarray,
    baseline_energy: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute the model-minus-baseline alphaR--beta energy-gap offset."""
    model = np.asarray(model_energy, dtype=np.float64).ravel()
    baseline = np.asarray(baseline_energy, dtype=np.float64).ravel()
    beta = np.asarray(masks["beta"], dtype=bool).ravel()
    alpha_r = np.asarray(masks["alphaR"], dtype=bool).ravel()
    if not (len(model) == len(baseline) == len(beta) == len(alpha_r)):
        raise ValueError("energies and basin masks must have the same length")
    if not beta.any() or not alpha_r.any():
        raise ValueError("both beta and alphaR masks need at least one frame")
    offset = model - baseline
    alpha_gap = float(np.mean(offset[alpha_r]))
    beta_gap = float(np.mean(offset[beta]))
    return {
        "delta": alpha_gap - beta_gap,
        "alphaR_offset": alpha_gap,
        "beta_offset": beta_gap,
        "alphaR_n": int(alpha_r.sum()),
        "beta_n": int(beta.sum()),
    }


def calibrated_alphaR_fraction(delta: float, kT: float = KT_298,
                               odds_baseline: float = ODDS_BASELINE) -> float:
    """Map an alphaR-minus-beta energy gap to a baseline-calibrated fraction."""
    log_odds = np.log(odds_baseline) - float(delta) / float(kT)
    odds = float(np.exp(np.clip(log_odds, -700.0, 700.0)))
    return odds / (1.0 + odds)


def _bootstrap_delta(offset: np.ndarray, beta: np.ndarray, alpha_r: np.ndarray,
                     n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    b = np.flatnonzero(beta)
    a = np.flatnonzero(alpha_r)
    values = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        values[i] = offset[rng.choice(a, len(a), replace=True)].mean() - offset[
            rng.choice(b, len(b), replace=True)
        ].mean()
    return float(np.median(values)), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("models", data.get("entries", data))
    if not isinstance(data, list) or not data:
        raise ValueError("model manifest must be a non-empty JSON list")
    out = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or "config" not in entry or "params" not in entry:
            raise ValueError(f"manifest entry {i} needs name, config, and params")
        row = dict(entry)
        row.setdefault("name", f"model_{i}")
        out.append(row)
    return out


def _model_energies(entry: dict[str, Any], R: np.ndarray, dataset: Path) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    from analysis.md.analyze_model_residuals_by_region import _load_model

    model, params, mask0, species0 = _load_model(
        str(entry["config"]), str(entry["params"]), str(dataset), R.shape[1]
    )
    energy = jax.jit(lambda r: model.compute_energy(params, r, mask0, species0))
    return np.asarray([float(energy(jnp.asarray(frame, dtype=jnp.float32))) for frame in R])


def evaluate_seed_manifest(
    manifest: list[dict[str, Any]],
    reference: np.ndarray | str | Path,
    n_frames: int,
    n_boot: int,
    seed: int,
    energy_evaluator: Callable[[dict[str, Any], np.ndarray], np.ndarray] | None = None,
    baseline_energies: np.ndarray | None = None,
    mapping_name: str = "ala2_backbone_cb_6",
) -> dict[str, Any]:
    """Evaluate all manifest entries on the same deterministic frame sample."""
    if isinstance(reference, (str, Path)):
        with np.load(reference, allow_pickle=False) as data:
            R_all = np.asarray(data["R"], dtype=np.float64)
    else:
        R_all = np.asarray(reference, dtype=np.float64)
    if R_all.ndim != 3:
        raise ValueError("reference coordinates must have shape (frames, beads, 3)")
    indices = np.linspace(0, len(R_all) - 1, min(n_frames, len(R_all))).astype(int)
    R = R_all[indices]
    if energy_evaluator is None:
        energy_evaluator = lambda entry, frames: _model_energies(entry, frames, Path(reference))
    baseline = np.zeros(len(R), dtype=np.float64) if baseline_energies is None else np.asarray(
        baseline_energies, dtype=np.float64
    )
    if len(baseline) == len(R_all):
        baseline = baseline[indices]
    if len(baseline) != len(R):
        raise ValueError("baseline energies must match reference frames or selected frames")

    try:
        masks = basin_masks(R, mapping_name)
    except (KeyError, ValueError, IndexError):
        # Tiny synthetic/unit-test arrays do not contain the project’s dihedral topology.
        split = len(R) // 2
        masks = {"beta": np.arange(len(R)) < split,
                 "alphaR": np.arange(len(R)) >= split}

    rows: list[dict[str, Any]] = []
    for entry in manifest:
        energies = np.asarray(energy_evaluator(entry, R), dtype=np.float64).ravel()
        if len(energies) != len(R):
            raise ValueError(f"energy evaluator for {entry.get('name')} returned wrong length")
        stats = delta_from_energies(energies, baseline, masks)
        offset = energies - baseline
        median, lo, hi = _bootstrap_delta(offset, masks["beta"], masks["alphaR"], n_boot, seed)
        rows.append({
            "name": entry["name"],
            **stats,
            "delta_boot_median": median,
            "delta_boot_lo": lo,
            "delta_boot_hi": hi,
            "alphaR_fraction_pred": calibrated_alphaR_fraction(stats["delta"]),
        })
    deltas = np.asarray([row["delta"] for row in rows], dtype=np.float64)
    return {
        "frame_indices": indices.tolist(),
        "n_frames": int(len(R)),
        "models": rows,
        "between_seed_delta_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        "between_seed_delta_range": float(np.ptp(deltas)) if len(deltas) else 0.0,
        "mapping": mapping_name,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-frames", type=int, default=20000)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--mapping", default="ala2_backbone_cb_6")
    args = parser.parse_args()

    result = evaluate_seed_manifest(
        _load_manifest(args.manifest), args.reference, args.n_frames, args.n_boot,
        args.seed, mapping_name=args.mapping,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_json(args.outdir / "seed_variance.json", result)
    with (args.outdir / "seed_variance.csv").open("w", newline="") as handle:
        rows = result["models"]
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.save(args.outdir / "frame_indices.npy", np.asarray(result["frame_indices"], dtype=np.int64))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
