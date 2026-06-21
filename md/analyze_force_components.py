#!/usr/bin/env python3
"""Re-evaluate MD trajectory frames and decompose ML/prior forces.

This is a post-hoc diagnostic for trajectories written by scripts/run_md.py.
It samples explicitly selected frame indices plus optional random frames,
rebuilds the same CombinedModel from the MD YAML, and computes one force tensor
per energy component via a single JAX VJP per frame.

Example:
  python md/analyze_force_components.py \
      --md-config local_work/md_configs_prior_combo_aggforce_20260518/md_4zoh_local_bond_in_i3_i4.yaml \
      --traj local_work/md_runs/20260518_prior_combo_aggforce/local_bond_in_i3_i4/traj_4zoh_local_bond_in_i3_i4_nvt_50000steps.npz \
      --selected-frames 113 \
      --n-random 20

  # Scan the trajectory every 10 saved frames, keeping frame 113 as well.
  python md/analyze_force_components.py \
      --md-config local_work/md_configs_prior_combo_aggforce_20260518/md_4zoh_local_bond_in_i3_i4.yaml \
      --traj local_work/md_runs/20260518_prior_combo_aggforce/local_bond_in_i3_i4/traj_4zoh_local_bond_in_i3_i4_nvt_50000steps.npz \
      --scan-trajectory \
      --selected-frames 113
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Match scripts/run_md.py: avoid GPU initialization on login nodes unless the
# user explicitly asks for CUDA or is inside an allocation.
if "JAX_PLATFORMS" not in os.environ and not os.environ.get("SLURM_JOB_ID"):
    os.environ["JAX_PLATFORMS"] = "cpu"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.jax_setup import apply_jax_compat_shims

apply_jax_compat_shims()

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from config.manager import ConfigManager
from data.loader import DatasetLoader
from data.preprocessor import CoordinatePreprocessor
from models.combined_model import CombinedModel


@dataclass
class RuntimeContext:
    root: Path
    md_config_path: Path
    md_config: dict[str, Any]
    training_config_path: Path
    params_path: Path
    dataset_path: Path
    model: CombinedModel
    params: dict[str, Any]
    mask: jax.Array
    species: jax.Array


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--md-config", type=Path, required=True, help="MD YAML used for run_md.py")
    p.add_argument("--traj", type=Path, required=True, help="Trajectory NPZ from run_md.py")
    p.add_argument(
        "--selected-frames",
        type=str,
        default="",
        help="Comma-separated zero-based trajectory frame indices, e.g. 113,114",
    )
    p.add_argument("--n-random", type=int, default=0, help="Random frames to add")
    p.add_argument("--seed", type=int, default=42, help="Random frame seed")
    p.add_argument(
        "--scan-trajectory",
        action="store_true",
        help="Evaluate every Nth saved trajectory frame; set N with --frame-stride.",
    )
    p.add_argument(
        "--frame-stride",
        type=int,
        default=10,
        help="Saved-frame stride used by --scan-trajectory (default: 10).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Default: traj.parent/force_component_analysis_<timestamp>",
    )
    p.add_argument(
        "--recompute-components",
        action="store_true",
        help="Force post-hoc VJP recomputation even when run_md.py saved force_decomp arrays.",
    )
    return p.parse_args()


def _project_root(config_file: Path) -> Path:
    env = os.environ.get("CAMEO_CG_PROJECT_ROOT", "").strip()
    if env:
        return Path(env)
    # MD configs often live under local_work/md_configs*/..., so parent-based
    # guessing is brittle. This script itself lives in <repo>/md/.
    return REPO_ROOT


def _resolve(path_str: str | Path, root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else root / p


def _parse_index_list(raw: str) -> list[int]:
    values: list[int] = []
    for piece in raw.replace("[", "").replace("]", "").split(","):
        text = piece.strip()
        if not text:
            continue
        value = int(text)
        if value < 0:
            raise ValueError(f"Frame indices must be >= 0, got {value}")
        values.append(value)
    return list(dict.fromkeys(values))


def select_frames(
    n_frames: int,
    selected: list[int],
    n_random: int,
    seed: int,
    scan_trajectory: bool = False,
    frame_stride: int = 10,
) -> tuple[list[int], list[int], list[int]]:
    if n_random < 0:
        raise ValueError("--n-random must be >= 0")
    if frame_stride <= 0:
        raise ValueError("--frame-stride must be > 0")
    for idx in selected:
        if idx >= n_frames:
            raise ValueError(f"Selected frame {idx} out of range for {n_frames} frames")

    scan_frames = list(range(0, n_frames, frame_stride)) if scan_trajectory else []
    excluded = set(selected) | set(scan_frames)
    remaining = np.asarray([i for i in range(n_frames) if i not in excluded], dtype=np.int32)
    n_random = min(n_random, int(remaining.size))
    rng = np.random.default_rng(seed)
    random_frames = (
        [int(x) for x in rng.choice(remaining, size=n_random, replace=False)]
        if n_random
        else []
    )
    final = list(dict.fromkeys(scan_frames + selected + random_frames))
    if not final:
        raise ValueError(
            "No frames selected; pass --scan-trajectory, --selected-frames, and/or --n-random"
        )
    return selected, random_frames, final


def _load_params(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        payload = pickle.load(fh)

    params = payload
    if isinstance(payload, dict):
        if isinstance(payload.get("params"), dict):
            params = payload["params"]
        elif isinstance(payload.get("best_params"), dict):
            params = payload["best_params"]
        elif (
            isinstance(payload.get("trainer_state"), dict)
            and isinstance(payload["trainer_state"].get("params"), dict)
        ):
            params = payload["trainer_state"]["params"]

    if isinstance(params, dict) and "ml" not in params:
        params = {"ml": params}
    if not isinstance(params, dict):
        raise TypeError(f"Unsupported params payload type: {type(params)}")
    return params


def build_runtime(md_config_path: Path) -> RuntimeContext:
    md_config_path = md_config_path.resolve()
    root = _project_root(md_config_path)
    with md_config_path.open() as fh:
        raw = yaml.safe_load(fh)
    md_cfg = raw.get("md")
    if md_cfg is None:
        raise ValueError(f"{md_config_path} has no 'md:' section")

    training_config_path = _resolve(md_cfg["training_config_path"], root)
    params_path = _resolve(md_cfg["params_path"], root)
    dataset_path = _resolve(md_cfg["dataset_path"], root)

    config = ConfigManager(str(training_config_path))
    if md_cfg.get("override_use_priors", False):
        config.set("model", "use_priors", True)
        config.set("model", "train_priors", False)
    if "ml_energy_scale" in md_cfg:
        config.set("model", "ml_energy_scale", float(md_cfg["ml_energy_scale"]))
    if "prior_energy_scale" in md_cfg:
        config.set("model", "prior_energy_scale", float(md_cfg["prior_energy_scale"]))
    if "robustness_gate" in md_cfg:
        config.set("model", "robustness_gate", dict(md_cfg.get("robustness_gate") or {}))
    if md_cfg.get("cell_list", False):
        config.set("model", "neighbor_disable_cell_list", False)

    loader = DatasetLoader(str(dataset_path))
    frame_idx = int(md_cfg.get("frame_idx", 0))
    R0 = jnp.asarray(loader.R[frame_idx], dtype=jnp.float32)
    mask = jnp.asarray(loader.mask[frame_idx], dtype=jnp.float32)
    species = jnp.asarray(loader.species[frame_idx], dtype=jnp.int32)

    preprocessor = CoordinatePreprocessor(
        cutoff=config.get_cutoff(),
        buffer_multiplier=config.get_buffer_multiplier(),
        park_multiplier=config.get_park_multiplier(),
    )
    box, R_shift = preprocessor.compute_box_extent(loader.R, loader.mask)
    R0 = jnp.asarray(
        preprocessor.center_and_park(
            np.asarray(R0)[None], np.asarray(mask)[None], box, R_shift
        )[0],
        dtype=jnp.float32,
    )

    data_n_species = int(np.max(loader.species)) + 1
    config_n_species = config.get("model", "allegro", "num_types", default=None)
    n_species = max(data_n_species, int(config_n_species or 0))

    model = CombinedModel(
        config=config,
        R0=R0,
        box=box,
        species=species,
        N_max=loader.N_max,
        n_species_override=n_species,
    )

    return RuntimeContext(
        root=root,
        md_config_path=md_config_path,
        md_config=md_cfg,
        training_config_path=training_config_path,
        params_path=params_path,
        dataset_path=dataset_path,
        model=model,
        params=_load_params(params_path),
        mask=mask,
        species=species,
    )


def _to_float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64))


def _component_force_key(energy_key: str) -> str:
    return f"F_{energy_key[2:]}" if energy_key.startswith("E_") else f"F_{energy_key}"


def _neighbor_for_frame(model: CombinedModel, R: jax.Array, mask: jax.Array) -> Any:
    nbrs = model.ml_model.nneigh_fn.update(R, model.ml_model.nbrs_init, mask=mask)
    return nbrs


def evaluate_components(
    ctx: RuntimeContext,
    R_np: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    R = jnp.asarray(R_np, dtype=jnp.float32)
    nbrs = _neighbor_for_frame(ctx.model, R, ctx.mask)

    base = ctx.model.compute_components(
        ctx.params, R, ctx.mask, ctx.species, neighbor=nbrs
    )
    energy_keys = list(base.keys())

    def component_tuple(R_: jax.Array) -> tuple[jax.Array, ...]:
        nbrs_ = _neighbor_for_frame(ctx.model, R_, ctx.mask)
        comps = ctx.model.compute_components(
            ctx.params, R_, ctx.mask, ctx.species, neighbor=nbrs_
        )
        return tuple(comps[key] for key in energy_keys)

    _, vjp_fn = jax.vjp(component_tuple, R)
    energies = {key: _to_float(base[key]) for key in energy_keys}
    forces: dict[str, np.ndarray] = {}
    for idx, key in enumerate(energy_keys):
        cotangent = tuple(1.0 if i == idx else 0.0 for i in range(len(energy_keys)))
        forces[_component_force_key(key)] = np.asarray(-vjp_fn(cotangent)[0], dtype=np.float32)
    return energies, forces




def evaluate_components_from_saved_decomp(
    *,
    traj: dict[str, np.ndarray],
    frame_index: int,
    step: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]] | None:
    if "decomp_step" not in traj or "decomp_F_ml" not in traj:
        return None
    decomp_steps = np.asarray(traj["decomp_step"], dtype=np.int64)
    matches = np.where(decomp_steps == int(step))[0]
    if matches.size == 0:
        return None
    j = int(matches[0])
    energies: dict[str, float] = {}
    if "decomp_E_ml" in traj:
        energies["E_ml"] = float(np.asarray(traj["decomp_E_ml"])[j])
    if "decomp_E_prior" in traj:
        energies["E_prior_total"] = float(np.asarray(traj["decomp_E_prior"])[j])
    if "E_ml" in energies and "E_prior_total" in energies:
        energies["E_total"] = energies["E_ml"] + energies["E_prior_total"]

    f_ml = np.asarray(traj["decomp_F_ml"][j], dtype=np.float32)
    f_prior = np.asarray(
        traj.get("decomp_F_prior", np.zeros_like(traj["decomp_F_ml"])),
        dtype=np.float32,
    )[j]
    forces = {
        "F_ml": f_ml,
        "F_prior_total": f_prior,
        "F_total": f_ml + f_prior,
    }
    return energies, forces


def min_pair_distance(R: np.ndarray, mask: np.ndarray) -> tuple[float, int, int]:
    valid = np.where(mask > 0.5)[0]
    coords = R[valid]
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist[np.eye(dist.shape[0], dtype=bool)] = np.inf
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    return float(dist[i, j]), int(valid[i]), int(valid[j])


def write_outputs(
    *,
    outdir: Path,
    ctx: RuntimeContext,
    traj_path: Path,
    selected_frames: list[int],
    random_frames: list[int],
    scan_trajectory: bool,
    frame_stride: int,
    frame_rows: list[dict[str, Any]],
    payload: dict[str, np.ndarray],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "md_config": str(ctx.md_config_path),
        "training_config": str(ctx.training_config_path),
        "params": str(ctx.params_path),
        "dataset": str(ctx.dataset_path),
        "traj": str(traj_path),
        "selected_frames": selected_frames,
        "random_frames": random_frames,
        "scan_trajectory": scan_trajectory,
        "frame_stride": frame_stride,
        "evaluated_frames": [int(row["frame_index"]) for row in frame_rows],
        "outputs": {
            "summary_json": str((outdir / "summary.json").resolve()),
            "frame_metrics_csv": str((outdir / "frame_metrics.csv").resolve()),
            "force_components_npz": str((outdir / "force_components.npz").resolve()),
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    fieldnames = list(frame_rows[0].keys())
    with (outdir / "frame_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_rows)

    np.savez_compressed(outdir / "force_components.npz", **payload)


def main() -> None:
    args = parse_args()
    ctx = build_runtime(args.md_config)

    traj_path = _resolve(args.traj, ctx.root).resolve()
    with np.load(traj_path, allow_pickle=False) as traj_npz:
        traj = {key: np.asarray(traj_npz[key]) for key in traj_npz.files}
    R_all = np.asarray(traj["R"], dtype=np.float32)
    steps = np.asarray(traj["step"], dtype=np.int64)
    saved_F = np.asarray(traj["F"], dtype=np.float32) if "F" in traj else None
    traj_mask = np.asarray(traj["mask"], dtype=np.float32) if "mask" in traj else np.asarray(ctx.mask)

    selected, random_frames, final_frames = select_frames(
        R_all.shape[0],
        _parse_index_list(args.selected_frames),
        int(args.n_random),
        int(args.seed),
        bool(args.scan_trajectory),
        int(args.frame_stride),
    )

    outdir = args.outdir
    if outdir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = traj_path.parent / f"force_component_analysis_{stamp}"
    else:
        outdir = _resolve(outdir, ctx.root)

    frame_rows: list[dict[str, Any]] = []
    payload_lists: dict[str, list[np.ndarray]] = {
        "coords": [],
        "mask": [],
        "species": [],
    }
    frame_indices: list[int] = []
    timesteps: list[int] = []
    roles: list[str] = []

    selected_set = set(selected)
    random_set = set(random_frames)
    scan_set = (
        set(range(0, R_all.shape[0], int(args.frame_stride)))
        if args.scan_trajectory
        else set()
    )
    used_saved_decomp = 0
    for frame_index in final_frames:
        R = R_all[frame_index]
        saved_components = None if args.recompute_components else evaluate_components_from_saved_decomp(
            traj=traj, frame_index=frame_index, step=int(steps[frame_index])
        )
        if saved_components is None:
            energies, forces = evaluate_components(ctx, R)
            component_source = "recomputed_vjp"
        else:
            energies, forces = saved_components
            component_source = "saved_decomp"
            used_saved_decomp += 1
        role_parts: list[str] = []
        if frame_index in scan_set:
            role_parts.append("scan")
        if frame_index in selected_set:
            role_parts.append("selected")
        if frame_index in random_set:
            role_parts.append("random")
        role = "+".join(role_parts) if role_parts else "manual"

        min_d, min_i, min_j = min_pair_distance(R, traj_mask)
        row: dict[str, Any] = {
            "frame_index": int(frame_index),
            "step": int(steps[frame_index]),
            "role": role,
            "component_source": component_source,
            "min_pair_distance": min_d,
            "min_pair_i": min_i,
            "min_pair_j": min_j,
        }
        row.update(energies)

        for force_key, force in forces.items():
            valid_force = np.asarray(force)[traj_mask > 0.5]
            norms = np.linalg.norm(valid_force, axis=1)
            row[f"{force_key}_rms"] = float(np.sqrt(np.mean(np.sum(valid_force ** 2, axis=1))))
            row[f"{force_key}_max_norm"] = float(np.max(norms))
            row[f"{force_key}_argmax_atom"] = int(np.where(traj_mask > 0.5)[0][int(np.argmax(norms))])
            payload_lists.setdefault(force_key, []).append(np.asarray(force, dtype=np.float32))

        if saved_F is not None and "F_total" in forces:
            diff = np.asarray(forces["F_total"], dtype=np.float32) - saved_F[frame_index]
            row["saved_total_force_rmse"] = float(np.sqrt(np.mean(diff ** 2)))
            row["saved_total_force_max_error"] = float(np.max(np.linalg.norm(diff, axis=1)))
            payload_lists.setdefault("saved_F", []).append(saved_F[frame_index])

        frame_rows.append(row)
        frame_indices.append(int(frame_index))
        timesteps.append(int(steps[frame_index]))
        roles.append(role)
        payload_lists["coords"].append(R)
        payload_lists["mask"].append(traj_mask.astype(np.float32))
        payload_lists["species"].append(np.asarray(ctx.species, dtype=np.int32))

    payload: dict[str, np.ndarray] = {
        "frame_indices": np.asarray(frame_indices, dtype=np.int32),
        "timesteps": np.asarray(timesteps, dtype=np.int64),
        "role": np.asarray(roles, dtype=str),
    }
    for key, values in payload_lists.items():
        payload[key] = np.stack(values, axis=0)

    write_outputs(
        outdir=outdir,
        ctx=ctx,
        traj_path=traj_path,
        selected_frames=selected,
        random_frames=random_frames,
        scan_trajectory=bool(args.scan_trajectory),
        frame_stride=int(args.frame_stride),
        frame_rows=frame_rows,
        payload=payload,
    )
    print(json.dumps(
        {
            "outdir": str(outdir.resolve()),
            "evaluated_frames": frame_indices,
            "steps": timesteps,
            "saved_decomp_frames_used": used_saved_decomp,
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
