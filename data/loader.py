"""
Dataset Loading for Coarse-Grained Protein Simulations

Loads NPZ datasets with coordinates, forces, species, and masks.
Handles amino acid to species ID mapping.

Consolidated from:
- train_fm_multiple_proteins.py
- compute_single_multi.py
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from glob import glob

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
        print("It's a directory")

        files = sorted(path.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found in directory: {path}")

        datasets = [np.load(f, allow_pickle=True) for f in files]

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

        print(f"Loaded {len(files)} NPZ files, total frames: {result['R'].shape[0]}")
        return result

    elif path.is_file():
        print("It's a file")

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
        self.species = np.asarray(raw_data["species"][indices], dtype=np.int32)

        # Store metadata
        self.N_max = raw_data["N_max"]
        self.resid = raw_data["resid"]
        self.resname = raw_data["resname"]
        self.Z = raw_data["Z"]

        # Species mapping
        if raw_data["aa_to_id"] is not None:
            self.aa_to_id = raw_data["aa_to_id"]
            self.id_to_aa = {v: k for k, v in self.aa_to_id.items()}
        else:
            # Create mapping from resnames
            self.aa_to_id, self.id_to_aa = create_species_mapping(self.resname)

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
