#!/usr/bin/env python3
"""Precompute directed type-pair distance bounds for Allegro edge gating."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

from config.manager import ConfigManager
from data.loader import load_npz
from training.edge_distance_gate import (
    build_edge_distance_gate_stats,
    save_edge_distance_gate_stats,
    select_clean_training_frames,
)


def _resolve(path: str | Path, *, base: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (base / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a directed amino-acid pair edge-distance gate artifact."
    )
    parser.add_argument("config", help="Training config YAML.")
    parser.add_argument("output", help="Output .npz artifact path.")
    parser.add_argument(
        "--falloff-percent-default",
        type=float,
        default=0.05,
        help="Default relative falloff width stored in the artifact.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = ConfigManager(config_path)
    config_base = config_path.parent
    data_path = _resolve(config.get_data_path(), base=config_base)
    dataset = load_npz(data_path)

    R = np.asarray(dataset["R"], dtype=np.float32)
    mask = np.asarray(dataset["mask"], dtype=np.float32)
    species = np.asarray(dataset["species"], dtype=np.int32)
    max_frames = config.get_max_frames()
    if max_frames is not None:
        max_frames = int(max_frames)
        R = R[:max_frames]
        mask = mask[:max_frames]
        species = species[:max_frames] if species.ndim == 2 else species

    train_R, train_mask, train_species = select_clean_training_frames(
        R=R,
        mask=mask,
        species=species,
        seed=int(config.get_seed()),
        val_fraction=float(config.get_val_fraction()),
    )
    model_species = int(config.get("model", "allegro", "num_types", default=0) or 0)
    observed_species = (
        int(np.max(train_species[train_mask > 0])) + 1 if np.any(train_mask > 0) else 0
    )
    stats = build_edge_distance_gate_stats(
        R=train_R,
        mask=train_mask,
        species=train_species,
        cutoff=float(config.get_cutoff()),
        n_species=max(model_species, observed_species),
        falloff_percent_default=float(args.falloff_percent_default),
        dataset_path=str(data_path),
        config_path=str(config_path),
    )
    output_path = _resolve(args.output, base=Path.cwd())
    save_edge_distance_gate_stats(stats, output_path)
    seen_pairs = int(np.sum(stats.count > 0))
    total_edges = int(np.sum(stats.count))
    print(
        f"Wrote {output_path} with {seen_pairs} directed type pairs, "
        f"{total_edges} directed edge observations, cutoff={stats.cutoff:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
