"""
MLIR Model Export for LAMMPS Integration

Exports trained ML models (Allegro, MACE, or PaiNN) to MLIR format for use
with chemtrain-deploy and LAMMPS.

Extracted from:
- model_exporters.py (Allegro-only, MACE and PaiNN removed)
"""

import jax
import jax.numpy as jnp
from jax_md import partition
from chemtrain.deploy import exporter, graphs
from pathlib import Path
from typing import Any, Optional

from config.types import PathLike, as_path
from utils.logging import export_logger


def _ensure_per_atom_energy(e: jax.Array, n_atoms: int) -> jax.Array:
    """
    Ensure energy has per-atom shape (n_atoms,) for LAMMPS compatibility.

    Args:
        e: Energy value (scalar or array)
        n_atoms: Number of atoms

    Returns:
        Per-atom energy array of shape (n_atoms,)

    Raises:
        ValueError: If array has incompatible shape
    """
    e = jnp.asarray(e)

    if e.ndim == 0:
        # Scalar energy: divide equally among atoms
        e = jnp.full((n_atoms,), e / n_atoms)
    elif e.ndim == 1 and e.shape[0] != n_atoms:
        raise ValueError(f"Unexpected per-atom energy length {e.shape[0]}, expected {n_atoms}")
    elif e.ndim > 1:
        # Multi-dimensional: flatten to (n_atoms,)
        e = e.reshape((n_atoms,))

    return e


class AllegroExporter(exporter.Exporter):
    """
    Exporter for Allegro models to MLIR format.

    Wraps trained Allegro models (with or without priors) for deployment
    to LAMMPS via chemtrain-deploy.

    Example:
        >>> from models.combined_model import CombinedModel
        >>> model = CombinedModel(config, R0, box, species, N_max)
        >>> params = trainer.get_best_params()
        >>>
        >>> exp = AllegroExporter.from_combined_model(model, params, box, species[0])
        >>> exp.export_to_file("model.mlir")
    """

    # chemtrain.deploy.exporter.Exporter configuration
    graph_type = graphs.SimpleSparseNeighborList
    unit_style = "real"
    nbr_order = [1, 1]

    def __init__(
        self,
        apply_fn,
        apply_model,
        nneigh_fn,
        displacement,
        box: jax.Array,
        species: jax.Array,
        bonds: jax.Array,
        angles: jax.Array,
        rep_pairs: jax.Array,
        dihedrals: jax.Array,
        prior_params: dict,
        params: Any,
        r_cutoff: float
    ):
        """
        Initialize Allegro exporter.

        Args:
            apply_fn: Energy function that takes (params, apply_model, nneigh_fn, ...)
            apply_model: Allegro model apply function
            nneigh_fn: Neighbor list function
            displacement: JAX-MD displacement function
            box: Simulation box dimensions
            species: Species IDs for atoms
            bonds: Bond pair indices
            angles: Angle triplet indices
            rep_pairs: Repulsive pair indices
            dihedrals: Dihedral quadruplet indices
            prior_params: Prior energy parameters
            params: Trained model parameters
            r_cutoff: Neighbor list cutoff distance
        """
        super().__init__()
        self.apply_fn = apply_fn
        self.apply_model = apply_model
        self.nneigh_fn = nneigh_fn
        self.displacement = displacement
        self.box = jnp.asarray(box, dtype=jnp.float32)
        self.species = jnp.asarray(species, dtype=jnp.int32)
        self.bonds = bonds
        self.angles = angles
        self.rep_pairs = rep_pairs
        self.dihedrals = dihedrals
        self.prior_params = prior_params
        self.params = params
        self.r_cutoff = r_cutoff

    def energy_fn(self, pos: jax.Array, species: jax.Array, graph) -> jax.Array:
        """
        Compute per-atom energies for LAMMPS.

        This method is called by chemtrain.deploy during MLIR export.

        Args:
            pos: Atomic positions, shape (n_atoms, 3)
            species: Species IDs (1-based indexing from LAMMPS)
            graph: Neighbor list graph from chemtrain.deploy

        Returns:
            Per-atom energies, shape (n_atoms,)
        """
        # Convert chemtrain.deploy graph to JAX-MD neighbor list
        neighbors = partition.NeighborList(
            jnp.stack((graph.senders, graph.receivers)),
            pos, None, None, graph.senders.size, partition.Sparse,
            None, None, None
        )

        # CRITICAL: LAMMPS uses 1-based species indexing, model uses 0-based
        species_model = species - 1
        species_model = jnp.maximum(species_model, 0)  # Clamp to valid range

        # Mask: all atoms are real (no padding in LAMMPS)
        mask = jnp.ones(pos.shape[0], dtype=jnp.float32)

        # Compute energy using the stored apply function
        e = self.apply_fn(
            self.params,
            self.apply_model,
            self.nneigh_fn,
            self.displacement,
            pos,
            mask,
            self.box,
            species_model,
            self.bonds,
            self.angles,
            self.rep_pairs,
            self.dihedrals,
            self.prior_params,
            neighbor=neighbors
        )

        return _ensure_per_atom_energy(e, pos.shape[0])

    def export_to_file(self, output_path: PathLike):
        """
        Export model to MLIR file.

        Args:
            output_path: Path to save MLIR file
        """
        output_path = as_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Call parent class export and save methods
        self.export()
        self.save(str(output_path))

        export_logger.info(f"Exported model to: {output_path}")

    @classmethod
    def from_combined_model(
        cls,
        model,  # CombinedModel instance
        params: Any,
        box: jax.Array,
        species: jax.Array,
        apply_fn: Optional[Any] = None
    ):
        """
        Create exporter from CombinedModel instance.

        Args:
            model: CombinedModel instance
            params: Trained model parameters
            box: Simulation box dimensions
            species: Species IDs for a representative structure
            apply_fn: Optional custom apply function (uses model's if None)

        Returns:
            AllegroExporter instance

        Example:
            >>> model = CombinedModel(config, R0, box, species, N_max)
            >>> params = trainer.get_best_params()
            >>> exporter = AllegroExporter.from_combined_model(
            ...     model, params, box, species
            ... )
            >>> exporter.export_to_file("model.mlir")
        """
        # Extract components from CombinedModel
        ml_model = model.ml_model
        topology = model.topology

        # Get topology arrays
        bonds, angles = topology.get_bonds_and_angles()
        dihedrals = topology.get_dihedrals()
        rep_pairs = topology.get_repulsive_pairs()

        # Get prior parameters (if priors are used)
        if model.use_priors:
            prior_params = params.get("prior", model.prior.params)
            if getattr(model.prior, "uses_splines", False):
                export_logger.info(
                    "Spline priors detected: spline arrays will be captured as constants "
                    "through model.compute_total_energy during MLIR tracing."
                )
        else:
            # Empty prior params if not using priors
            prior_params = {}

        # Use model's compute_total_energy as apply_fn if not provided
        if apply_fn is None:
            # Create wrapper function that matches expected signature
            def default_apply_fn(
                params_, apply_model_, nneigh_fn_, displacement_,
                R_, mask_, box_, species_,
                bonds_, angles_, rep_pairs_, dihedrals_, prior_params_,
                neighbor=None
            ):
                return model.compute_total_energy(params_, R_, mask_, species_, neighbor)

            apply_fn = default_apply_fn

        return cls(
            apply_fn=apply_fn,
            apply_model=ml_model.model_apply_fn,
            nneigh_fn=ml_model.nneigh_fn,
            displacement=ml_model.displacement,
            box=box,
            species=species,
            bonds=bonds,
            angles=angles,
            rep_pairs=rep_pairs,
            dihedrals=dihedrals,
            prior_params=prior_params,
            params=params,
            r_cutoff=ml_model.cutoff
        )

    def __repr__(self) -> str:
        return (
            f"AllegroExporter(r_cutoff={self.r_cutoff}, "
            f"n_atoms={self.species.shape[0]})"
        )
