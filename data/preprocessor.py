"""
Coordinate Preprocessing for CG Protein Simulations

Handles:
- Box extent computation based on coordinate range
- Centering coordinates in simulation box
- "Parking" padded atoms far from real atoms

Consolidated from:
- train_fm_multiple_proteins.py
- compute_single_multi.py
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Union


class CoordinatePreprocessor:
    """
    Preprocessor for protein coordinates in padded datasets.

    Handles centering coordinates in the simulation box and
    "parking" padded atoms far away to avoid spurious interactions.

    Example:
        >>> preprocessor = CoordinatePreprocessor(cutoff=12.0)
        >>> R_processed, box_extent, shift = preprocessor.process_dataset(R, mask)
    """

    def __init__(self, cutoff: float, buffer_multiplier: float = 2.0, park_multiplier: float = 0.95):
        """
        Initialize coordinate preprocessor.

        Args:
            cutoff: Neighbor list cutoff distance
            buffer_multiplier: Multiplier for buffer around coordinates (default: 2.0)
            park_multiplier: Multiplier for parking location (default: 0.95)
        """
        self.cutoff = cutoff
        self.buffer_multiplier = buffer_multiplier
        self.park_multiplier = park_multiplier

    def compute_box_extent(
        self,
        R: jax.Array,
        mask: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Compute simulation box extent and centering shift.

        The box is sized to fit all real atoms plus a buffer zone
        (2 * cutoff by default). The shift centers coordinates in the box.

        Args:
            R: Coordinates of shape (N_frames, N_atoms, 3) or (N_atoms, 3)
            mask: Validity mask of shape (N_frames, N_atoms) or (N_atoms,)

        Returns:
            extent: Box dimensions (3,)
            shift: Translation to center coordinates (3,)

        Example:
            >>> R = jnp.array([...])  # (100, 50, 3)
            >>> mask = jnp.ones((100, 50))
            >>> extent, shift = preprocessor.compute_box_extent(R, mask)
        """
        # Use large sentinel values for masked atoms
        big = 1e9
        m = mask[..., None]  # Add coordinate dimension

        # Replace masked atoms with sentinel values
        R_for_min = jnp.where(m > 0, R, +big)
        R_for_max = jnp.where(m > 0, R, -big)

        # Find min/max over all frames and atoms
        mins = jnp.min(R_for_min.reshape(-1, 3), axis=0)
        maxs = jnp.max(R_for_max.reshape(-1, 3), axis=0)

        # Add buffer for neighbor list
        buffer = self.buffer_multiplier * self.cutoff
        extent = (maxs - mins) + 2.0 * buffer

        # Compute shift to center coordinates
        center = 0.5 * (mins + maxs)
        shift = 0.5 * extent - center

        return extent.astype(jnp.float32), shift.astype(jnp.float32)

    def center_and_park(
        self,
        R: jax.Array,
        mask: jax.Array,
        extent: jax.Array,
        shift: jax.Array
    ) -> jax.Array:
        """
        Center coordinates and park padded atoms.

        Real atoms (mask=1) are shifted to be centered in the box.
        Padded atoms (mask=0) are placed at a "parking" location
        far from real atoms to avoid spurious neighbor list interactions.

        Args:
            R: Coordinates of shape (N_frames, N_atoms, 3) or (N_atoms, 3)
            mask: Validity mask of shape (N_frames, N_atoms) or (N_atoms,)
            extent: Box dimensions (3,)
            shift: Centering shift (3,)

        Returns:
            Processed coordinates with same shape as R

        Example:
            >>> R_centered = preprocessor.center_and_park(R, mask, extent, shift)
        """
        # Apply centering shift
        R_shifted = R + shift

        # Compute parking location (near box boundary)
        park_location = self.park_multiplier * extent

        # Expand mask dimensions to match coordinates
        if R.ndim == 3:  # (N_frames, N_atoms, 3)
            mask_expanded = mask[..., None]
        else:  # (N_atoms, 3)
            mask_expanded = mask[:, None]

        # Real atoms keep their shifted positions, padded atoms go to parking
        R_processed = mask_expanded * R_shifted + (1.0 - mask_expanded) * park_location[None, :]

        return R_processed

    def process_frame(
        self,
        R: jax.Array,
        mask: jax.Array,
        extent: jax.Array,
        shift: jax.Array
    ) -> jax.Array:
        """
        Process a single frame with given box parameters.

        Args:
            R: Coordinates (N_atoms, 3)
            mask: Validity mask (N_atoms,)
            extent: Box dimensions (3,)
            shift: Centering shift (3,)

        Returns:
            Processed coordinates (N_atoms, 3)
        """
        return self.center_and_park(R, mask, extent, shift)

    def process_dataset(
        self,
        R: jax.Array,
        mask: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Process entire dataset: compute box and preprocess coordinates.

        This is a convenience method that computes the box extent
        based on all frames and then processes all coordinates.

        Args:
            R: Coordinates (N_frames, N_atoms, 3)
            mask: Validity mask (N_frames, N_atoms)

        Returns:
            R_processed: Processed coordinates (N_frames, N_atoms, 3)
            extent: Box dimensions (3,)
            shift: Centering shift (3,)

        Example:
            >>> R_processed, extent, shift = preprocessor.process_dataset(R, mask)
        """
        # Compute box parameters from entire dataset
        extent, shift = self.compute_box_extent(R, mask)

        # Process all frames
        R_processed = self.center_and_park(R, mask, extent, shift)

        return R_processed, extent, shift

    def __repr__(self) -> str:
        return (
            f"CoordinatePreprocessor(cutoff={self.cutoff}, "
            f"buffer={self.buffer_multiplier}, park={self.park_multiplier})"
        )


def compute_extent_and_shift_masked(
    R: jax.Array,
    mask: jax.Array,
    cutoff: float
) -> Tuple[jax.Array, jax.Array]:
    """
    Convenience function to compute box extent and shift.

    Args:
        R: Coordinates (N_frames, N_atoms, 3) or (N_atoms, 3)
        mask: Validity mask (N_frames, N_atoms) or (N_atoms,)
        cutoff: Neighbor list cutoff distance

    Returns:
        extent: Box dimensions (3,)
        shift: Centering shift (3,)

    Example:
        >>> extent, shift = compute_extent_and_shift_masked(R, mask, cutoff=12.0)
    """
    preprocessor = CoordinatePreprocessor(cutoff=cutoff)
    return preprocessor.compute_box_extent(R, mask)


def shift_and_park_coords(
    R: jax.Array,
    mask: jax.Array,
    extent: jax.Array,
    shift: jax.Array
) -> jax.Array:
    """
    Convenience function to center and park coordinates.

    Args:
        R: Coordinates (N_frames, N_atoms, 3) or (N_atoms, 3)
        mask: Validity mask (N_frames, N_atoms) or (N_atoms,)
        extent: Box dimensions (3,)
        shift: Centering shift (3,)

    Returns:
        Processed coordinates with same shape as R

    Example:
        >>> R_processed = shift_and_park_coords(R, mask, extent, shift)
    """
    preprocessor = CoordinatePreprocessor(cutoff=10.0)  # cutoff not used in this function
    return preprocessor.center_and_park(R, mask, extent, shift)
