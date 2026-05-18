#!/usr/bin/env python3
"""FES/TICA/PCA analysis for cameo_cg JAX-MD NPZ trajectories.

Reads one or more trajectory NPZ files produced by run_md.py (or optionally a
LAMMPS dump), computes pair-distance features, fits or projects TICA/PCA, and
produces a 2D free-energy surface.

Modes:
  fit   — fit a new TICA/PCA model on the input trajectory (default)
  project — project onto a previously saved model (supply --reference-model
             and --reference-pairs)

Multiple NPZ files (--npz a.npz b.npz …) are concatenated in the order given
before fitting/projection.

Output (in --outdir):
  <prefix>_tica_projection.csv   — per-frame TIC 1 / TIC 2 coordinates
  <prefix>_pair_indices.csv      — atom pair definitions used as features
  <prefix>_tica_model.pkl        — fitted model (fit mode only)
  <prefix>_fes_grid.npz          — F, xedges, yedges arrays
  <prefix>_fes.png               — 2D FES plot
  <prefix>_metadata.json         — provenance + parameters

Usage examples:
  # Fit on a single trajectory
  python md/analyze_traj.py --npz local_work/md_runs/foo/traj.npz \\
      --outdir results/fes --prefix foo

  # Fit TICA, save model, then project a second trajectory onto it
  python md/analyze_traj.py --npz traj2.npz \\
      --reference-model results/fes/foo_tica_model.pkl \\
      --reference-pairs results/fes/foo_pair_indices.csv \\
      --outdir results/fes --prefix foo_run2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_KB_KJ_MOL_K = 0.00831446261815324  # kJ / (mol · K)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--npz",
        type=Path,
        nargs="+",
        metavar="FILE",
        help="One or more cameo_cg trajectory NPZ files (concatenated)",
    )
    src.add_argument(
        "--dump",
        type=Path,
        metavar="FILE",
        help="LAMMPS custom dump trajectory (alternative to --npz)",
    )

    p.add_argument("--outdir", type=Path, required=True, help="Output directory")
    p.add_argument("--prefix", type=str, default="traj", help="Output filename prefix")

    p.add_argument(
        "--method",
        choices=("tica", "pca"),
        default="tica",
        help="Dimensionality-reduction method (default: tica)",
    )
    p.add_argument(
        "--lagtime",
        type=int,
        default=10,
        help="TICA lagtime in frames (ignored for PCA)",
    )
    p.add_argument("--bins", type=int, default=80, help="Histogram bins per FES axis")
    p.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature for FES in K (default: 300)",
    )
    p.add_argument(
        "--n-pairs",
        type=int,
        default=200,
        help="Number of pair-distance features (default: 200)",
    )
    p.add_argument(
        "--pair-seed", type=int, default=42, help="RNG seed for random pair selection"
    )
    p.add_argument(
        "--pair-mode",
        choices=("random", "sequential"),
        default="random",
        help="How to select pairs (default: random)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Keep at most this many frames per file (0 = all)",
    )
    p.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Use every Nth frame (default: 1, i.e. all frames)",
    )
    p.add_argument(
        "--standardize",
        action="store_true",
        help="Z-score standardize pair-distance features before fitting",
    )
    p.add_argument(
        "--reference-model",
        type=Path,
        default=None,
        help="Pre-fitted model PKL — activates projection mode",
    )
    p.add_argument(
        "--reference-pairs",
        type=Path,
        default=None,
        help="Pair CSV from the reference model (required with --reference-model)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_npz_coords(
    path: Path,
    frame_stride: int = 1,
    max_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load valid-atom coordinates from a cameo_cg trajectory NPZ.

    Returns:
        coords: (n_frames, n_valid_atoms, 3) float64, Angstrom
        frame_steps: (n_frames,) int64 simulation step indices
    """
    with np.load(str(path), allow_pickle=False) as data:
        R = data["R"].astype(np.float64)          # (n_frames, N, 3)
        steps = data["step"].astype(np.int64)      # (n_frames,)
        if "mask" in data:
            valid = np.asarray(data["mask"]) > 0   # (N,)
        else:
            valid = np.ones(R.shape[1], dtype=bool)

    stride = max(frame_stride, 1)
    indices = list(range(0, R.shape[0], stride))
    if max_frames > 0:
        indices = indices[:max_frames]

    R = R[indices][:, valid, :]
    steps = steps[indices]
    return R, steps


def _choose_dump_coord_columns(header: list[str]) -> tuple[str, str, str]:
    x = "xu" if "xu" in header else "x"
    y = "yu" if "yu" in header else "y"
    z = "zu" if "zu" in header else "z"
    for c in (x, y, z):
        if c not in header:
            raise ValueError(f"Missing coordinate column '{c}' in dump header: {header}")
    return x, y, z


def load_dump_coords(
    path: Path,
    frame_stride: int = 1,
    max_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load coordinates from a LAMMPS custom dump.

    Returns:
        coords: (n_frames, n_atoms, 3) float64
        timesteps: (n_frames,) int64
    """
    frames: list[np.ndarray] = []
    timesteps: list[int] = []
    parsed_idx = 0

    with path.open("r") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            ts_line = fh.readline()
            if not ts_line:
                break
            timestep = int(ts_line.strip())

            if not fh.readline().startswith("ITEM: NUMBER OF ATOMS"):
                continue
            n_atoms = int(fh.readline().strip())

            if not fh.readline().startswith("ITEM: BOX BOUNDS"):
                continue
            fh.readline(); fh.readline(); fh.readline()  # skip 3 box lines

            atom_line = fh.readline().strip()
            if not atom_line.startswith("ITEM: ATOMS"):
                continue
            header = atom_line.split()[2:]

            if "id" not in header:
                raise ValueError("Dump must include atom id column")
            id_idx = header.index("id")
            x_col, y_col, z_col = _choose_dump_coord_columns(header)
            xi, yi, zi = header.index(x_col), header.index(y_col), header.index(z_col)

            rows: list[tuple[int, float, float, float]] = []
            ok = True
            for _ in range(n_atoms):
                row = fh.readline()
                if not row:
                    ok = False
                    break
                cols = row.split()
                try:
                    rows.append((int(cols[id_idx]), float(cols[xi]), float(cols[yi]), float(cols[zi])))
                except Exception:
                    ok = False
                    break
            if not ok:
                continue

            take = parsed_idx % max(frame_stride, 1) == 0
            parsed_idx += 1
            if not take:
                continue

            rows.sort(key=lambda r: r[0])
            frame = np.array([[x, y, z] for _, x, y, z in rows], dtype=np.float64)
            frames.append(frame)
            timesteps.append(timestep)
            if max_frames > 0 and len(frames) >= max_frames:
                break

    if not frames:
        raise ValueError(f"No complete frames parsed from {path}")
    return np.stack(frames, axis=0), np.asarray(timesteps, dtype=np.int64)


# ---------------------------------------------------------------------------
# Pair-distance features
# ---------------------------------------------------------------------------

def choose_pairs(n_atoms: int, n_pairs: int, mode: str, seed: int) -> np.ndarray:
    if n_atoms < 2:
        raise ValueError("Need at least 2 atoms to build pair features")
    if mode == "sequential":
        pairs = np.array([(i, i + 1) for i in range(n_atoms - 1)], dtype=np.int64)
        return pairs[: min(n_pairs, len(pairs))] if n_pairs > 0 else pairs
    rng = np.random.default_rng(seed)
    max_pairs = n_atoms * (n_atoms - 1) // 2
    target = min(max(n_pairs, 1), max_pairs)
    selected: set[tuple[int, int]] = set()
    while len(selected) < target:
        i = int(rng.integers(0, n_atoms - 1))
        j = int(rng.integers(i + 1, n_atoms))
        selected.add((i, j))
    return np.asarray(sorted(selected), dtype=np.int64)


def load_pairs_csv(path: Path) -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            i = int(row["atom_i_1based"]) - 1
            j = int(row["atom_j_1based"]) - 1
            if i > j:
                i, j = j, i
            pairs.append((i, j))
    if not pairs:
        raise ValueError(f"No pairs found in {path}")
    return np.asarray(pairs, dtype=np.int64)


def build_features(coords: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Compute pair Euclidean distances.  Returns (n_frames, n_pairs) float64."""
    diff = coords[:, pairs[:, 0], :] - coords[:, pairs[:, 1], :]
    return np.linalg.norm(diff, axis=-1).astype(np.float64)


def pair_hash(pairs: np.ndarray) -> str:
    payload = ";".join(f"{int(i)}-{int(j)}" for i, j in pairs).encode()
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def fit_tica(X: np.ndarray, lagtime: int) -> tuple[Any, np.ndarray]:
    try:
        from deeptime.decomposition import TICA
    except ImportError as exc:
        raise RuntimeError("deeptime is required for TICA. Install it in your env.") from exc
    if lagtime < 1:
        raise ValueError("TICA lagtime must be >= 1")
    if X.shape[0] <= lagtime:
        raise ValueError(
            f"Need n_frames > lagtime, got n_frames={X.shape[0]}, lagtime={lagtime}"
        )
    model = TICA(lagtime=lagtime, dim=2).fit(X).fetch_model()
    Y = np.asarray(model.transform(X), dtype=np.float64)
    return model, Y


def fit_pca(X: np.ndarray) -> tuple[Any, np.ndarray]:
    try:
        from deeptime.decomposition import PCA
    except ImportError as exc:
        raise RuntimeError("deeptime is required for PCA. Install it in your env.") from exc
    model = PCA(dim=2).fit(X).fetch_model()
    Y = np.asarray(model.transform(X), dtype=np.float64)
    return model, Y


def project_onto_model(X: np.ndarray, model_path: Path) -> tuple[Any, np.ndarray]:
    with model_path.open("rb") as f:
        model = pickle.load(f)
    Y = np.asarray(model.transform(X), dtype=np.float64)
    if Y.ndim != 2 or Y.shape[1] < 2:
        raise ValueError(f"Unexpected projected output shape: {Y.shape}")
    return model, Y


# ---------------------------------------------------------------------------
# FES
# ---------------------------------------------------------------------------

def compute_fes_2d(
    Y: np.ndarray,
    bins: int,
    temperature: float,
    xedges: np.ndarray | None = None,
    yedges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D free-energy surface from a (n_frames, 2) projection.

    Returns F [kJ/mol], xedges, yedges.  Unvisited bins are NaN.
    """
    hist_bins: Any = bins if (xedges is None or yedges is None) else [xedges, yedges]
    H, xe, ye = np.histogram2d(Y[:, 0], Y[:, 1], bins=hist_bins)
    occupied = H > 0
    P = np.zeros_like(H, dtype=np.float64)
    P[occupied] = H[occupied] / H.sum()
    F = np.full_like(P, np.nan, dtype=np.float64)
    F[occupied] = -_KB_KJ_MOL_K * temperature * np.log(P[occupied])
    F[occupied] -= np.nanmin(F[occupied])
    return F, xe, ye


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fes(
    F: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    out_png: Path,
    title: str = "Free Energy Surface",
    method: str = "TICA",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.colormaps["turbo"].copy()
    cmap.set_bad("white")
    mesh = ax.pcolormesh(
        xedges, yedges, np.ma.masked_invalid(F).T, cmap=cmap, shading="flat", vmin=0
    )
    label1 = "PC 1" if method.upper() == "PCA" else "TIC 1"
    label2 = "PC 2" if method.upper() == "PCA" else "TIC 2"
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax).set_label("F [kJ/mol]")
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_projection_csv(path: Path, steps: np.ndarray, Y: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "step", "dim1", "dim2"])
        for i in range(Y.shape[0]):
            w.writerow([i, int(steps[i]), float(Y[i, 0]), float(Y[i, 1])])


def write_pairs_csv(path: Path, pairs: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_index", "atom_i_1based", "atom_j_1based"])
        for k, (i, j) in enumerate(pairs):
            w.writerow([k, int(i + 1), int(j + 1)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load coordinates
    # ------------------------------------------------------------------
    all_coords: list[np.ndarray] = []
    all_steps: list[np.ndarray] = []

    if args.npz is not None:
        for npz_path in args.npz:
            coords, steps = load_npz_coords(
                npz_path, frame_stride=args.frame_stride, max_frames=args.max_frames
            )
            all_coords.append(coords)
            all_steps.append(steps)
            print(f"Loaded {coords.shape[0]} frames × {coords.shape[1]} atoms from {npz_path.name}")
    else:
        coords, steps = load_dump_coords(
            args.dump, frame_stride=args.frame_stride, max_frames=args.max_frames
        )
        all_coords.append(coords)
        all_steps.append(steps)
        print(f"Loaded {coords.shape[0]} frames × {coords.shape[1]} atoms from {args.dump.name}")

    # Consistency check: all files must have the same atom count.
    n_atoms_list = [c.shape[1] for c in all_coords]
    if len(set(n_atoms_list)) > 1:
        raise ValueError(f"Atom count mismatch across input files: {n_atoms_list}")

    coords_cat = np.concatenate(all_coords, axis=0)    # (T, N, 3)
    steps_cat = np.concatenate(all_steps, axis=0)      # (T,)
    n_frames_total, n_atoms, _ = coords_cat.shape
    print(f"Total frames: {n_frames_total}, atoms per frame: {n_atoms}")

    # ------------------------------------------------------------------
    # 2. Pair features
    # ------------------------------------------------------------------
    project_mode = args.reference_model is not None
    if project_mode:
        if args.reference_pairs is None:
            raise SystemExit("--reference-pairs is required when --reference-model is provided")
        pairs = load_pairs_csv(args.reference_pairs)
        if args.standardize:
            raise SystemExit("--standardize is not supported in projection mode")
        print(f"Using {pairs.shape[0]} pairs from reference: {args.reference_pairs.name}")
    else:
        pairs = choose_pairs(n_atoms, args.n_pairs, args.pair_mode, args.pair_seed)
        print(f"Using {pairs.shape[0]} pair-distance features (mode={args.pair_mode})")

    X = build_features(coords_cat, pairs)   # (T, n_pairs)

    if args.standardize and not project_mode:
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0
        X = (X - mu) / sigma
    else:
        mu = sigma = None

    # ------------------------------------------------------------------
    # 3. Fit or project
    # ------------------------------------------------------------------
    method_upper = args.method.upper()
    if project_mode:
        model, Y = project_onto_model(X, args.reference_model)
        print(f"Projected {n_frames_total} frames onto reference {method_upper} model")
    elif args.method == "tica":
        model, Y = fit_tica(X, lagtime=args.lagtime)
        print(f"Fitted TICA (lagtime={args.lagtime}) on {n_frames_total} frames")
    else:
        model, Y = fit_pca(X)
        print(f"Fitted PCA on {n_frames_total} frames")

    # ------------------------------------------------------------------
    # 4. FES
    # ------------------------------------------------------------------
    F, xedges, yedges = compute_fes_2d(Y, bins=args.bins, temperature=args.temperature)
    print(
        f"FES: {args.bins}×{args.bins} bins, T={args.temperature} K, "
        f"F_max={float(np.nanmax(F)):.2f} kJ/mol"
    )

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    args.outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix
    method_tag = args.method  # "tica" or "pca"

    proj_csv = args.outdir / f"{prefix}_{method_tag}_projection.csv"
    pairs_csv = args.outdir / f"{prefix}_pair_indices.csv"
    model_pkl = args.outdir / f"{prefix}_{method_tag}_model.pkl"
    fes_npz = args.outdir / f"{prefix}_fes_grid.npz"
    fes_png = args.outdir / f"{prefix}_fes.png"
    meta_json = args.outdir / f"{prefix}_metadata.json"

    write_projection_csv(proj_csv, steps_cat, Y)
    write_pairs_csv(pairs_csv, pairs)

    if not project_mode:
        with model_pkl.open("wb") as f:
            pickle.dump(model, f)

    np.savez(str(fes_npz), F=F, xedges=xedges, yedges=yedges)

    title_parts = [
        args.prefix,
        "projected" if project_mode else "fitted",
        method_upper,
    ]
    plot_fes(F, xedges, yedges, fes_png, title=" — ".join(title_parts), method=args.method)

    sources = [str(p.resolve()) for p in args.npz] if args.npz else [str(args.dump.resolve())]
    metadata: dict[str, Any] = {
        "sources": sources,
        "method": args.method,
        "mode": "project" if project_mode else "fit",
        "n_frames": int(n_frames_total),
        "n_atoms": int(n_atoms),
        "n_pairs": int(pairs.shape[0]),
        "pair_mode": "from_reference_pairs" if project_mode else args.pair_mode,
        "pair_hash": pair_hash(pairs),
        "lagtime": None if (project_mode or args.method == "pca") else int(args.lagtime),
        "bins": int(args.bins),
        "temperature": float(args.temperature),
        "frame_stride": int(args.frame_stride),
        "max_frames_per_file": int(args.max_frames),
        "standardize": bool(args.standardize),
        "reference_model": str(args.reference_model.resolve()) if project_mode else None,
        "reference_pairs": str(args.reference_pairs.resolve()) if project_mode else None,
        "outputs": {
            "fes_png": str(fes_png),
            "projection_csv": str(proj_csv),
            "pairs_csv": str(pairs_csv),
            "model_pkl": str(model_pkl) if not project_mode else None,
            "fes_grid_npz": str(fes_npz),
        },
    }
    meta_json.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    print(
        json.dumps(
            {
                "mode": "project" if project_mode else "fit",
                "method": args.method,
                "n_frames": int(n_frames_total),
                "outdir": str(args.outdir),
                "fes_png": str(fes_png),
                "metadata": str(meta_json),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
