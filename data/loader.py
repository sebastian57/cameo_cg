"""
Dataset Loading for Coarse-Grained Protein Simulations

Loads NPZ datasets with coordinates, forces, species, and masks.
Handles amino acid to species ID mapping.
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Sequence
from config.types import PathLike, as_path


def load_npz(path: PathLike) -> Dict[str, Any]:
    """
    Load dataset from NPZ file.

    Args:
        path: Path to NPZ file

    Returns:
        Dictionary with keys:
            - R: Coordinates (N_frames, N_atoms, 3)
            - F: Forces (N_frames, N_atoms, 3)
            - Z: Atomic numbers (N_frames, N_atoms) or (N_atoms,)
            - resid: Residue IDs (N_frames, N_atoms) or (N_atoms,)
            - resname: Residue names (N_frames, N_atoms) or (N_atoms,)
            - species: Species IDs (N_frames, N_atoms)
            - N_max: Maximum number of atoms (scalar or array)
            - mask: Valid atom mask (N_frames, N_atoms)
            - aa_to_id: Amino acid to species ID mapping (optional)

    Raises:
        FileNotFoundError: If NPZ file doesn't exist
        KeyError: If required keys are missing
    """
    path = as_path(path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")

    if path.is_dir():
        files = sorted(path.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found in directory: {path}")

        datasets = [np.load(f, allow_pickle=True) for f in files]
        n_atoms_per_file = [int(d["R"].shape[1]) for d in datasets]
        unique_n_atoms = sorted(set(n_atoms_per_file))
        if len(unique_n_atoms) > 1:
            is_bucketed = all(f.name.startswith("bucket_N") for f in files)
            details = ", ".join(str(n) for n in unique_n_atoms)
            if is_bucketed:
                raise ValueError(
                    f"Directory '{path}' contains bucketed NPZ files with mixed N_max "
                    f"({details}). Use scripts/train.py with "
                    f"--multi-protein-dir {path} instead of data.path."
                )
            raise ValueError(
                f"Directory '{path}' contains NPZ files with mixed N_max ({details}), "
                "which cannot be concatenated into a single DatasetLoader."
            )

        def cat(key):
            return np.concatenate([d[key] for d in datasets], axis=0)

        result = {
            "R":       cat("R").astype(np.float32),
            "F":       cat("F").astype(np.float32),
            "mask":    cat("mask").astype(np.float32),
            "species": cat("species").astype(np.int32),
            "Z":       datasets[0]["Z"].astype(np.int32) if "Z" in datasets[0] else None,
            "resid":   datasets[0]["resid"].astype(np.int32) if "resid" in datasets[0] else None,
            "resname": datasets[0]["resname"] if "resname" in datasets[0] else None,
            "N_max":   int(datasets[0]["N_max"][0]) if "N_max" in datasets[0] else datasets[0]["R"].shape[1],
            "aa_to_id": datasets[0]["aa_to_id"].item() if "aa_to_id" in datasets[0] else None,
        }

        return result

    data = np.load(path, allow_pickle=True)

    # Required keys
    required = ["R", "F"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing required keys in NPZ: {missing}")

    result = {
        "R": data["R"].astype(np.float32),
        "F": data["F"].astype(np.float32),
        "Z": data["Z"].astype(np.int32) if "Z" in data else None,
        "resid": data["resid"].astype(np.int32) if "resid" in data else None,
        "resname": data["resname"] if "resname" in data else None,
        "species": data["species"].astype(np.int32) if "species" in data else None,
        "N_max": data["N_max"] if "N_max" in data else data["R"].shape[1],
        "mask": data["mask"] if "mask" in data else np.ones(data["R"].shape[:2], dtype=np.float32),
        "aa_to_id": data["aa_to_id"].item() if "aa_to_id" in data else None,
    }

    # Handle N_max being an array
    if isinstance(result["N_max"], np.ndarray):
        result["N_max"] = int(result["N_max"][0])

    return result


def create_species_mapping(resnames: np.ndarray) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create bidirectional amino acid to species ID mapping.

    Args:
        resnames: Array of amino acid names (e.g., ['ALA', 'VAL', ...])

    Returns:
        aa_to_id: Dictionary mapping amino acid name to species ID
        id_to_aa: Dictionary mapping species ID to amino acid name

    Example:
        >>> resnames = np.array(['ALA', 'VAL', 'LEU', 'ALA'])
        >>> aa_to_id, id_to_aa = create_species_mapping(resnames)
        >>> aa_to_id
        {'ALA': 0, 'LEU': 1, 'VAL': 2}
    """
    # Get unique amino acids in sorted order
    unique_aas = sorted(set(resnames.flatten()))

    aa_to_id = {aa: i for i, aa in enumerate(unique_aas)}
    id_to_aa = {i: aa for aa, i in aa_to_id.items()}

    return aa_to_id, id_to_aa


def _choose_tile_capacity(
    n_valid: int,
    target_beads: int,
    bucket_beads: Optional[Sequence[int]],
) -> int:
    """Choose padded tile capacity for a packed tile."""
    if bucket_beads:
        for bucket in bucket_beads:
            if n_valid <= int(bucket):
                return int(bucket)
        raise ValueError(
            f"Tile with {n_valid} valid beads does not fit into configured "
            f"data.tile_bucket_beads={list(bucket_beads)}."
        )
    return max(int(target_beads), int(n_valid))


def _pack_structures_greedy(
    sorted_indices: np.ndarray,
    valid_counts: np.ndarray,
    target_beads: int,
    drop_incomplete: bool,
) -> list[list[int]]:
    """Greedily pack structures into tiles by valid bead budget."""
    tiles: list[list[int]] = []
    current: list[int] = []
    current_beads = 0

    for idx in sorted_indices:
        n_valid = int(valid_counts[idx])
        if n_valid <= 0:
            continue
        if current and (current_beads + n_valid > target_beads):
            tiles.append(current)
            current = []
            current_beads = 0
        current.append(int(idx))
        current_beads += n_valid

    if current:
        if not drop_incomplete or current_beads >= target_beads or not tiles:
            tiles.append(current)

    return tiles


def compute_force_loss_weights(split: Dict[str, np.ndarray]) -> np.ndarray:
    """Build per-particle weights that average force MSE uniformly by structure."""
    mask = np.asarray(split["mask"], dtype=np.float32)
    weights = np.zeros_like(mask, dtype=np.float32)

    if "segment_id" in split:
        segment_id = np.asarray(split["segment_id"], dtype=np.int32)
        for batch_idx in range(mask.shape[0]):
            valid = mask[batch_idx] > 0
            if not np.any(valid):
                continue
            valid_segments = segment_id[batch_idx][valid]
            if valid_segments.size == 0:
                continue
            counts = np.bincount(
                valid_segments, minlength=int(valid_segments.max()) + 1
            ).astype(np.float32)
            weights[batch_idx, valid] = 1.0 / np.maximum(counts[valid_segments], 1.0)
        return weights

    n_valid = np.sum(mask > 0, axis=1, dtype=np.float32)
    safe_n_valid = np.maximum(n_valid[:, None], 1.0)
    weights = np.where(mask > 0, mask / safe_n_valid, 0.0)
    return np.asarray(weights, dtype=np.float32)


def attach_batch_metadata(split: Dict[str, np.ndarray], sample_ids: np.ndarray) -> Dict[str, np.ndarray]:
    """Attach profiling and loss-normalization metadata to a batch split."""
    annotated = dict(split)
    sample_ids = np.asarray(sample_ids, dtype=np.int32)
    n_items = int(sample_ids.shape[0])
    n_valid = np.asarray(np.sum(annotated["mask"] > 0, axis=1), dtype=np.int32)
    capacity = int(annotated["R"].shape[1])

    annotated.setdefault("force_loss_mask", np.asarray(annotated["mask"], dtype=np.float32))
    annotated.setdefault("force_loss_weights", compute_force_loss_weights(annotated))
    annotated.setdefault("n_valid", n_valid)
    annotated.setdefault("n_segments", np.ones((n_items,), dtype=np.int32))
    annotated.setdefault("meta_batch_item_id", sample_ids)
    annotated.setdefault("meta_capacity", np.full((n_items,), capacity, dtype=np.int32))
    annotated.setdefault("meta_fill_ratio", n_valid.astype(np.float32) / max(capacity, 1))
    annotated.setdefault("meta_n_force_components", np.asarray(n_valid * 3, dtype=np.int32))
    annotated.setdefault("meta_source_structure_ids", sample_ids[:, None])
    annotated.setdefault("meta_source_structure_n_valid", n_valid[:, None])
    annotated.setdefault("meta_structure_size_min", n_valid)
    annotated.setdefault("meta_structure_size_mean", n_valid.astype(np.float32))
    annotated.setdefault("meta_structure_size_max", n_valid)
    annotated.setdefault("meta_structure_size_std", np.zeros((n_items,), dtype=np.float32))
    return annotated


def build_tiled_dataset(
    R: np.ndarray,
    F: np.ndarray,
    mask: np.ndarray,
    species: np.ndarray,
    structure_ids: Optional[np.ndarray] = None,
    target_beads: int = 1000,
    bucket_beads: Optional[Sequence[int]] = None,
    shuffle_structures: bool = False,
    sort_by_size: bool = True,
    drop_incomplete: bool = False,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Pack many small structures into disconnected tiled pseudo-structures.

    Each tile is represented as a single dense padded sample with unchanged core
    fields (`R`, `F`, `mask`, `species`) plus `segment_id` metadata identifying
    the original structure segment per valid node.

    Profiling metadata is included with a `meta_` prefix so later training
    diagnostics can attribute optimizer updates back to tile composition.
    """
    if target_beads <= 0:
        raise ValueError(f"target_beads must be > 0, got {target_beads}.")

    if R.ndim != 3 or F.ndim != 3:
        raise ValueError("R and F must have shape (n_structures, n_atoms, 3).")
    if mask.ndim != 2 or species.ndim != 2:
        raise ValueError("mask and species must have shape (n_structures, n_atoms).")
    if not (R.shape[0] == F.shape[0] == mask.shape[0] == species.shape[0]):
        raise ValueError("R/F/mask/species must share the same leading dimension.")

    n_structures = int(R.shape[0])
    valid_counts = np.asarray(np.sum(mask > 0, axis=1), dtype=np.int32)
    if structure_ids is None:
        structure_ids = np.arange(n_structures, dtype=np.int32)
    else:
        structure_ids = np.asarray(structure_ids, dtype=np.int32)
        if structure_ids.shape != (n_structures,):
            raise ValueError(
                "structure_ids must have shape (n_structures,), "
                f"got {structure_ids.shape}."
            )

    order = np.arange(n_structures, dtype=np.int32)
    if shuffle_structures:
        rng = np.random.RandomState(seed)
        rng.shuffle(order)
    if sort_by_size:
        # First-fit decreasing improves fill ratio and keeps behavior deterministic.
        order = order[np.argsort(valid_counts[order])[::-1]]

    tiles = _pack_structures_greedy(
        order, valid_counts, target_beads=target_beads, drop_incomplete=drop_incomplete
    )
    if not tiles:
        raise ValueError(
            "No tiles were constructed. Check mask/target_beads/drop_incomplete settings."
        )
    max_segments = max(len(tile) for tile in tiles)

    tile_R = []
    tile_F = []
    tile_mask = []
    tile_species = []
    tile_segment_id = []
    tile_n_valid = []
    tile_n_segments = []
    meta_batch_item_id = []
    meta_capacity = []
    meta_fill_ratio = []
    meta_n_force_components = []
    meta_source_structure_ids = []
    meta_source_structure_n_valid = []
    meta_structure_size_min = []
    meta_structure_size_mean = []
    meta_structure_size_max = []
    meta_structure_size_std = []

    for tile_id, tile_indices in enumerate(tiles):
        tile_indices_arr = np.asarray(tile_indices, dtype=np.int32)
        struct_valid = valid_counts[tile_indices_arr]
        n_valid_tile = int(np.sum(struct_valid))
        capacity = _choose_tile_capacity(
            n_valid=n_valid_tile,
            target_beads=target_beads,
            bucket_beads=bucket_beads,
        )

        R_out = np.zeros((capacity, 3), dtype=np.float32)
        F_out = np.zeros((capacity, 3), dtype=np.float32)
        mask_out = np.zeros((capacity,), dtype=np.float32)
        species_out = np.zeros((capacity,), dtype=np.int32)
        segment_out = np.full((capacity,), -1, dtype=np.int32)

        cursor = 0
        for seg_id, struct_idx in enumerate(tile_indices):
            valid_idx = np.flatnonzero(mask[struct_idx] > 0)
            n_valid = int(valid_idx.shape[0])
            if n_valid == 0:
                continue
            if cursor + n_valid > capacity:
                raise ValueError(
                    f"Tile overflow: required {cursor + n_valid} > capacity {capacity}."
                )
            sl = slice(cursor, cursor + n_valid)
            R_out[sl] = R[struct_idx, valid_idx]
            F_out[sl] = F[struct_idx, valid_idx]
            mask_out[sl] = 1.0
            species_out[sl] = species[struct_idx, valid_idx]
            segment_out[sl] = np.int32(seg_id)
            cursor += n_valid

        tile_R.append(R_out)
        tile_F.append(F_out)
        tile_mask.append(mask_out)
        tile_species.append(species_out)
        tile_segment_id.append(segment_out)
        tile_n_valid.append(np.int32(cursor))
        tile_n_segments.append(np.int32(len(tile_indices)))
        meta_batch_item_id.append(np.int32(tile_id))
        meta_capacity.append(np.int32(capacity))
        meta_fill_ratio.append(np.float32(cursor / max(capacity, 1)))
        meta_n_force_components.append(np.int32(cursor * 3))

        source_ids = np.full((max_segments,), -1, dtype=np.int32)
        source_valid = np.zeros((max_segments,), dtype=np.int32)
        source_ids[: len(tile_indices)] = structure_ids[tile_indices_arr]
        source_valid[: len(tile_indices)] = struct_valid
        meta_source_structure_ids.append(source_ids)
        meta_source_structure_n_valid.append(source_valid)
        meta_structure_size_min.append(np.int32(np.min(struct_valid)))
        meta_structure_size_mean.append(np.float32(np.mean(struct_valid)))
        meta_structure_size_max.append(np.int32(np.max(struct_valid)))
        meta_structure_size_std.append(np.float32(np.std(struct_valid)))

    return {
        "R": np.stack(tile_R, axis=0),
        "F": np.stack(tile_F, axis=0),
        "mask": np.stack(tile_mask, axis=0),
        "species": np.stack(tile_species, axis=0),
        "segment_id": np.stack(tile_segment_id, axis=0),
        "n_valid": np.asarray(tile_n_valid, dtype=np.int32),
        "n_segments": np.asarray(tile_n_segments, dtype=np.int32),
        "meta_batch_item_id": np.asarray(meta_batch_item_id, dtype=np.int32),
        "meta_capacity": np.asarray(meta_capacity, dtype=np.int32),
        "meta_fill_ratio": np.asarray(meta_fill_ratio, dtype=np.float32),
        "meta_n_force_components": np.asarray(meta_n_force_components, dtype=np.int32),
        "meta_source_structure_ids": np.stack(meta_source_structure_ids, axis=0),
        "meta_source_structure_n_valid": np.stack(
            meta_source_structure_n_valid, axis=0
        ),
        "meta_structure_size_min": np.asarray(meta_structure_size_min, dtype=np.int32),
        "meta_structure_size_mean": np.asarray(meta_structure_size_mean, dtype=np.float32),
        "meta_structure_size_max": np.asarray(meta_structure_size_max, dtype=np.int32),
        "meta_structure_size_std": np.asarray(meta_structure_size_std, dtype=np.float32),
    }


class DatasetLoader:
    """
    Loader for coarse-grained protein simulation datasets.

    Handles loading NPZ files, species mapping, and provides
    convenient access to dataset components.

    Example:
        >>> loader = DatasetLoader("dataset.npz")
        >>> R, F, mask, species = loader.get_batch(0, 32)
        >>> print(f"Loaded {loader.n_frames} frames, {loader.n_atoms} atoms")
    """

    def __init__(self, npz_path: PathLike, max_frames: Optional[int] = None, seed: int = 42):
        """
        Initialize dataset loader.

        Args:
            npz_path: Path to NPZ dataset file
            max_frames: Maximum number of frames to use (None = use all)
            seed: Random seed for frame shuffling
        """
        self.npz_path = as_path(npz_path)
        self.seed = seed

        # Load raw data
        raw_data = load_npz(npz_path)

        # Shuffle frames if max_frames is specified
        n_total = raw_data["R"].shape[0]
        if max_frames is not None and max_frames < n_total:
            rng = np.random.RandomState(seed)
            indices = rng.permutation(n_total)[:max_frames]
        else:
            indices = np.arange(n_total)

        # Store dataset as NumPy arrays to avoid unnecessary device transfers.
        # JAX arrays are created on-demand via .R_jax etc. when needed on device.
        self.R = np.asarray(raw_data["R"][indices], dtype=np.float32)
        self.F = np.asarray(raw_data["F"][indices], dtype=np.float32)
        self.mask = np.asarray(raw_data["mask"][indices], dtype=np.float32)

        # Species: optional — if absent, default to zeros (all atoms same type).
        # Also handle 1-D species arrays (shape (N_atoms,)) by broadcasting to
        # (N_frames, N_atoms) so that per-frame indexing works correctly.
        raw_species = raw_data["species"]
        if raw_species is None:
            self.species = np.zeros_like(self.mask, dtype=np.int32)
        else:
            raw_species = np.asarray(raw_species, dtype=np.int32)
            if raw_species.ndim == 1:
                # Per-atom species (no frame dimension): tile to all selected frames.
                self.species = np.tile(raw_species[None, :], (len(indices), 1))
            else:
                self.species = raw_species[indices]

        # Store metadata
        self.N_max = raw_data["N_max"]
        self.resid = raw_data["resid"]
        self.resname = raw_data["resname"]
        self.Z = raw_data["Z"]

        # Species mapping
        if raw_data["aa_to_id"] is not None:
            self.aa_to_id = raw_data["aa_to_id"]
            self.id_to_aa = {v: k for k, v in self.aa_to_id.items()}
        elif self.resname is not None:
            # Create mapping from resnames
            self.aa_to_id, self.id_to_aa = create_species_mapping(self.resname)
        else:
            # No mapping available (e.g. non-AA CG datasets with pre-assigned species IDs)
            self.aa_to_id = {}
            self.id_to_aa = {}

    @property
    def n_frames(self) -> int:
        """Number of frames in the dataset."""
        return self.R.shape[0]

    @property
    def n_atoms(self) -> int:
        """Number of atoms (including padding)."""
        return self.R.shape[1]

    @property
    def n_species(self) -> int:
        """Number of unique species types."""
        return len(self.aa_to_id)

    def get_frame(self, idx: int) -> Dict[str, jax.Array]:
        """
        Get a single frame by index.

        Args:
            idx: Frame index

        Returns:
            Dictionary with R, F, mask, species for the frame
        """
        return {
            "R": self.R[idx],
            "F": self.F[idx],
            "mask": self.mask[idx],
            "species": self.species[idx],
        }

    def get_batch(self, start: int, end: int) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Get a batch of frames.

        Args:
            start: Start frame index
            end: End frame index (exclusive)

        Returns:
            R, F, mask, species arrays for the batch
        """
        return (
            self.R[start:end],
            self.F[start:end],
            self.mask[start:end],
            self.species[start:end],
        )

    def get_all(self) -> Dict[str, jax.Array]:
        """Get complete dataset as dictionary."""
        return {
            "R": self.R,
            "F": self.F,
            "mask": self.mask,
            "species": self.species,
        }

    def split_train_val(self, val_fraction: float = 0.1) -> Tuple["DatasetLoader", "DatasetLoader"]:
        """
        Split dataset into training and validation sets.

        Args:
            val_fraction: Fraction of data to use for validation

        Returns:
            train_loader, val_loader

        Note:
            Creates new loader instances with split data
        """
        n_train = int(np.round(self.n_frames * (1 - val_fraction)))
        n_val = self.n_frames - n_train

        # Create train loader
        train_loader = DatasetLoader.__new__(DatasetLoader)
        train_loader.npz_path = self.npz_path
        train_loader.seed = self.seed
        train_loader.R = self.R[:n_train]
        train_loader.F = self.F[:n_train]
        train_loader.mask = self.mask[:n_train]
        train_loader.species = self.species[:n_train]
        train_loader.N_max = self.N_max
        train_loader.resid = self.resid
        train_loader.resname = self.resname
        train_loader.Z = self.Z
        train_loader.aa_to_id = self.aa_to_id
        train_loader.id_to_aa = self.id_to_aa

        # Create val loader
        val_loader = DatasetLoader.__new__(DatasetLoader)
        val_loader.npz_path = self.npz_path
        val_loader.seed = self.seed
        val_loader.R = self.R[n_train:]
        val_loader.F = self.F[n_train:]
        val_loader.mask = self.mask[n_train:]
        val_loader.species = self.species[n_train:]
        val_loader.N_max = self.N_max
        val_loader.resid = self.resid
        val_loader.resname = self.resname
        val_loader.Z = self.Z
        val_loader.aa_to_id = self.aa_to_id
        val_loader.id_to_aa = self.id_to_aa

        return train_loader, val_loader

    def summary(self) -> str:
        """Get summary string of dataset."""
        real_atoms_per_frame = jnp.sum(self.mask, axis=1)
        avg_atoms = float(jnp.mean(real_atoms_per_frame))
        min_atoms = int(jnp.min(real_atoms_per_frame))
        max_atoms = int(jnp.max(real_atoms_per_frame))

        return (
            f"Dataset: {self.npz_path.name}\n"
            f"  Frames: {self.n_frames}\n"
            f"  Atoms: {self.n_atoms} (max), {avg_atoms:.1f} (avg), "
            f"{min_atoms}-{max_atoms} (range)\n"
            f"  Species: {self.n_species}\n"
            f"  Amino acids: {', '.join(sorted(self.aa_to_id.keys()))}"
        )

    def __repr__(self) -> str:
        return f"DatasetLoader('{self.npz_path}', n_frames={self.n_frames}, n_atoms={self.n_atoms})"

    def __len__(self) -> int:
        return self.n_frames


class BucketedDatasetLoader:
    """
    Loader for protein-aware bucketed datasets.

    Finds all bucket_N*.npz files in a directory (or accepts an explicit list)
    and wraps each in a DatasetLoader.  Buckets are sorted by N_max so training
    code can iterate from smallest to largest.

    Example:
        >>> loader = BucketedDatasetLoader("processed/03_bucketed_npz")
        >>> for n_max, bucket in loader.buckets:
        ...     print(f"N_max={n_max}: {len(bucket)} frames")
    """

    def __init__(
        self,
        bucket_dir_or_paths,
        max_frames: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Args:
            bucket_dir_or_paths: path to a directory containing bucket_N*.npz
                                 files, OR an explicit list/glob of NPZ paths.
            max_frames:          maximum frames per bucket (None = use all)
            seed:                random seed for frame shuffling
        """
        from pathlib import Path as _Path

        if isinstance(bucket_dir_or_paths, (str, _Path)):
            bucket_dir = _Path(bucket_dir_or_paths)
            npz_paths = sorted(bucket_dir.glob("bucket_N*.npz"))
            if not npz_paths:
                raise FileNotFoundError(
                    f"No bucket_N*.npz files found in {bucket_dir}"
                )
        else:
            npz_paths = sorted(_Path(p) for p in bucket_dir_or_paths)

        # Build (N_max, DatasetLoader) pairs sorted by N_max
        self._buckets = []
        for p in npz_paths:
            dl = DatasetLoader(str(p), max_frames=max_frames, seed=seed)
            self._buckets.append((dl.N_max, dl))

        self._buckets.sort(key=lambda x: x[0])

    @property
    def buckets(self):
        """List of (N_max, DatasetLoader) pairs sorted by N_max (ascending)."""
        return self._buckets

    @property
    def n_buckets(self) -> int:
        """Number of buckets."""
        return len(self._buckets)

    def summary(self) -> str:
        lines = [f"BucketedDatasetLoader: {self.n_buckets} buckets"]
        for n_max, dl in self._buckets:
            lines.append(f"  N_max={n_max:4d}: {dl.n_frames} frames  [{dl.npz_path.name}]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"BucketedDatasetLoader(n_buckets={self.n_buckets}, "
                f"N_max_range=[{self._buckets[0][0]}, {self._buckets[-1][0]}])"
                if self._buckets else "BucketedDatasetLoader(empty)")
