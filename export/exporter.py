"""
MLIR Model Export for LAMMPS Integration

Exports trained ML models to MLIR format for use with chemtrain-deploy and LAMMPS.
"""

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
from chemtrain.deploy import exporter, graphs
from chemtrain.deploy._protobuf import model_pb2 as deploy_model_proto
from jax_md import partition

from config.types import PathLike, as_path
from utils.logging import export_logger

_INLINE_LOC_RANGE_RE = re.compile(rb'":(\d+):(\d+) to :(\d+)\)')


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
        e = jnp.full((n_atoms,), e / n_atoms)
    elif e.ndim == 1 and e.shape[0] != n_atoms:
        raise ValueError(f"Unexpected per-atom energy length {e.shape[0]}, expected {n_atoms}")
    elif e.ndim > 1:
        e = e.reshape((n_atoms,))

    return e


def _normalize_inline_location_ranges(mlir_path: Path) -> int:
    """Normalize new MLIR inline location shorthand for older StableHLO parsers."""
    model = deploy_model_proto.Model()
    raw = mlir_path.read_bytes()
    consumed = model.ParseFromString(raw)
    if consumed != len(raw):
        raise ValueError(
            f"Could not parse full protobuf payload in {mlir_path}; "
            f"consumed={consumed}, size={len(raw)}"
        )

    module_bytes = model.mlir_module.encode("utf-8")

    def _expand(match: re.Match[bytes]) -> bytes:
        start_line = match.group(1)
        start_col = match.group(2)
        end_col = match.group(3)
        return b'":' + start_line + b":" + start_col + b" to " + start_line + b":" + end_col + b")"

    patched, num_replacements = _INLINE_LOC_RANGE_RE.subn(_expand, module_bytes)
    if num_replacements > 0:
        model.mlir_module = patched.decode("utf-8")
        mlir_path.write_bytes(model.SerializeToString())
    return int(num_replacements)


def _normalize_export_mode(mode: Any) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in {"auto", "symbolic", "fixed_size"}:
        raise ValueError(
            f"Unsupported export.mode={mode!r}. Expected one of: auto, symbolic, fixed_size."
        )
    return normalized


def _normalize_species_input_convention(convention: Any) -> str:
    normalized = str(convention).strip().lower().replace("-", "_")
    aliases = {
        "model": "model",
        "model_species": "model",
        "zero_based": "model",
        "zero_indexed": "model",
        "connector": "model",
        "chemtrain_deploy": "model",
        "lammps": "lammps",
        "lammps_types": "lammps",
        "one_based": "lammps",
        "one_indexed": "lammps",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported export.species_input_convention={convention!r}. "
            "Expected one of: model, lammps."
        )
    return aliases[normalized]


class ModelExporter(exporter.Exporter):
    """
    Exporter for ML models to MLIR format.

    Wraps trained models (with or without priors) for deployment
    to LAMMPS via chemtrain-deploy.
    """

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
        r_cutoff: float,
        *,
        combined_model: Optional[Any] = None,
        ml_model: Optional[Any] = None,
        sample_positions: Optional[jax.Array] = None,
        sample_neighbors: Optional[Any] = None,
        sample_species_model: Optional[jax.Array] = None,
        export_mode: str = "auto",
        species_input_convention: str = "model",
        also_export_naive: bool = False,
        naive_equivalence_atol: Optional[float] = None,
        naive_equivalence_per_atom_atol: float = 1.0e-6,
    ):
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
        self.combined_model = combined_model
        self.ml_model = ml_model
        self.sample_positions = None if sample_positions is None else jnp.asarray(sample_positions)
        self.sample_neighbors = sample_neighbors
        self.sample_species_model = (
            None if sample_species_model is None else jnp.asarray(sample_species_model, dtype=jnp.int32)
        )
        self.export_mode = _normalize_export_mode(export_mode)
        self.species_input_convention = _normalize_species_input_convention(
            species_input_convention
        )
        self.also_export_naive = bool(also_export_naive)
        self.naive_equivalence_atol = (
            None if naive_equivalence_atol is None else float(naive_equivalence_atol)
        )
        self.naive_equivalence_per_atom_atol = float(naive_equivalence_per_atom_atol)
        self._export_debug_logged = False

    def energy_fn(
        self,
        pos: jax.Array,
        species: jax.Array,
        graph,
        valid_mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Compute per-atom energies for LAMMPS."""
        if not self._export_debug_logged:
            export_logger.info(
                "Export trace inputs: pos_shape=%s species_shape=%s senders_shape=%s receivers_shape=%s senders_size=%s receivers_size=%s",
                tuple(pos.shape),
                tuple(species.shape),
                tuple(graph.senders.shape),
                tuple(graph.receivers.shape),
                graph.senders.size,
                graph.receivers.size,
            )
            self._export_debug_logged = True

        neighbors = partition.NeighborList(
            jnp.stack((graph.receivers, graph.senders)),
            pos,
            None,
            None,
            graph.senders.size,
            partition.Sparse,
            None,
            None,
            None,
        )

        if self.species_input_convention == "lammps":
            species_model = jnp.maximum(species - 1, 0)
        else:
            species_model = jnp.maximum(species, 0)
        if valid_mask is None:
            mask = jnp.ones(pos.shape[0], dtype=jnp.float32)
        else:
            mask = jnp.asarray(valid_mask, dtype=jnp.float32)

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
            neighbor=neighbors,
        )

        return _ensure_per_atom_energy(e, pos.shape[0])

    def _energy_fn(self, position, species, n_local, n_ghost, newton, *graph_args):
        # Expects particles to be sorted by local, ghost, and padding atoms.
        valid_mask = jnp.arange(position.shape[0]) < (n_local + n_ghost)
        ghost_mask = jnp.arange(position.shape[0]) < n_local

        graph, build_statistics = self.graph_type.create_from_args(
            self.r_cutoff,
            self.nbr_order,
            position,
            species,
            ghost_mask,
            valid_mask,
            newton,
            *graph_args,
        )
        graph = jax.lax.stop_gradient(graph)

        def force_and_aux(pos):
            per_atom_energies = self.energy_fn(
                pos,
                species,
                graph,
                valid_mask=valid_mask,
            )

            assert per_atom_energies.shape == ghost_mask.shape, (
                f"Per particle energies have shape {per_atom_energies.shape}, "
                f"but should have shape {ghost_mask.shape}."
            )

            total_energy = exporter.md_util.high_precision_sum(
                jnp.where(valid_mask, per_atom_energies, jnp.float32(0.0))
            )
            local_energy = exporter.md_util.high_precision_sum(
                jnp.where(ghost_mask, per_atom_energies, jnp.float32(0.0))
            )

            force_energy = jnp.where(newton, local_energy, total_energy)
            force_energy = jnp.negative(force_energy)

            aux = local_energy, *build_statistics
            return force_energy, aux

        return jax.grad(force_and_aux, has_aux=True)(position)

    def _current_tp_methods(self) -> tuple[str, ...]:
        methods = getattr(self.ml_model, "export_tp_methods", ()) if self.ml_model is not None else ()
        return tuple(methods)

    def _uses_uniform_1d(self) -> bool:
        return any(method == "uniform_1d" for method in self._current_tp_methods())

    def _resolve_export_mode(self) -> str:
        if self.export_mode == "auto":
            return "fixed_size" if self._uses_uniform_1d() else "symbolic"
        return self.export_mode

    def _build_naive_apply_model(self):
        if self.ml_model is None or not hasattr(self.ml_model, "build_export_apply_fn"):
            raise ValueError(
                "This model does not expose build_export_apply_fn(); cannot create a symbolic naive export."
            )
        return self.ml_model.build_export_apply_fn(tp_method_override="naive")

    def _validate_naive_equivalence(self, naive_apply_model) -> None:
        compute_with_apply = getattr(self.ml_model, "_compute_energy_with_apply", None)
        if compute_with_apply is None:
            return
        if self.sample_positions is None or self.sample_neighbors is None or self.sample_species_model is None:
            return

        ml_params = self.params.get("ml", self.params) if isinstance(self.params, dict) else self.params
        mask = jnp.ones(self.sample_positions.shape[0], dtype=jnp.float32)
        ref_energy = self.ml_model.compute_energy(
            ml_params,
            self.sample_positions,
            mask,
            self.sample_species_model,
            neighbor=self.sample_neighbors,
        )
        naive_energy_raw = compute_with_apply(
            naive_apply_model,
            ml_params,
            self.sample_positions,
            mask,
            self.sample_species_model,
            neighbor=self.sample_neighbors,
        )
        # naive_apply_model may return per-atom energies (ndim==1); sum to scalar.
        naive_energy_arr = jnp.asarray(naive_energy_raw)
        naive_energy = (
            jnp.sum(naive_energy_arr) if naive_energy_arr.ndim > 0 else naive_energy_arr
        )
        diff = float(jnp.abs(ref_energy - naive_energy))
        n_atoms = int(self.sample_positions.shape[0])
        adaptive_atol = max(1.0e-4, self.naive_equivalence_per_atom_atol * n_atoms)
        atol = self.naive_equivalence_atol if self.naive_equivalence_atol is not None else adaptive_atol
        if diff > atol:
            raise ValueError(
                f"Cross-backend export sanity check failed: |dE|={diff:.3e} exceeds "
                f"{atol:.3e} for {n_atoms} atoms."
            )
        export_logger.info(
            "Cross-backend naive export sanity check passed: |dE|=%.2e <= %.2e for %d atoms",
            diff,
            atol,
            n_atoms,
        )

    def _neighborlist_to_sparse_buffers(self, neighbor, n_atoms: int):
        idx = jnp.asarray(neighbor.idx)
        invalid_idx = jnp.asarray(int(n_atoms), dtype=jnp.int32)
        if getattr(neighbor, "format", None) == partition.Sparse and idx.ndim == 2 and idx.shape[0] == 2:
            senders = jnp.asarray(idx[0].reshape(-1), dtype=jnp.int32)
            receivers = jnp.asarray(idx[1].reshape(-1), dtype=jnp.int32)
        else:
            if idx.ndim != 2:
                raise ValueError(
                    f"Cannot convert neighbor list with idx shape {idx.shape} to sparse export buffers."
                )
            n_rows, n_cols = idx.shape
            senders = jnp.repeat(jnp.arange(n_rows, dtype=jnp.int32), n_cols)
            receivers = jnp.asarray(idx.reshape(-1), dtype=jnp.int32)

        valid = (
            (senders >= 0)
            & (senders < invalid_idx)
            & (receivers >= 0)
            & (receivers < invalid_idx)
            & (senders != receivers)
        )
        senders = jnp.where(valid, senders, invalid_idx)
        receivers = jnp.where(valid, receivers, invalid_idx)
        edge_buffer = jnp.ones((senders.shape[0],), dtype=jnp.bool_)
        return senders, receivers, edge_buffer

    def _build_fixed_size_export_inputs(self):
        if self.sample_positions is None or self.sample_neighbors is None or self.sample_species_model is None:
            raise ValueError(
                "Fixed-size export requires representative sample positions, species, and a neighbor list."
            )

        positions = jnp.asarray(self.sample_positions, dtype=jnp.float32)
        species_model = jnp.asarray(self.sample_species_model, dtype=jnp.int32)
        if self.species_input_convention == "lammps":
            species_export = jnp.where(
                species_model >= 0,
                species_model + 1,
                jnp.ones_like(species_model),
            )
        else:
            species_export = jnp.maximum(species_model, 0)
        senders, receivers, edge_buffer = self._neighborlist_to_sparse_buffers(
            self.sample_neighbors,
            positions.shape[0],
        )
        return (
            positions,
            species_export,
            jnp.asarray(int(positions.shape[0]), dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False, dtype=jnp.bool_),
            senders,
            receivers,
            edge_buffer,
        )

    @contextmanager
    def _temporary_apply_model(self, apply_model):
        previous_apply_model = self.apply_model
        self.apply_model = apply_model
        try:
            yield
        finally:
            self.apply_model = previous_apply_model

    def _export_single(
        self,
        output_path: PathLike,
        *,
        export_inputs=None,
        disabled_checks=None,
        apply_model=None,
    ) -> Path:
        output_path = as_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._export_debug_logged = False
        apply_model = self.apply_model if apply_model is None else apply_model
        with self._temporary_apply_model(apply_model):
            self.export(export_inputs=export_inputs, disabled_checks=disabled_checks)
            self.save(str(output_path))
        replacements = _normalize_inline_location_ranges(output_path)
        if replacements:
            export_logger.info(
                "Normalized %d inline location ranges for StableHLO compatibility.",
                replacements,
            )
        export_logger.info("Exported model to: %s", output_path)
        return output_path

    def export_to_file(self, output_path: PathLike):
        """Export model to MLIR file(s)."""
        output_path = as_path(output_path)
        resolved_mode = self._resolve_export_mode()
        tp_methods = self._current_tp_methods()
        export_logger.info(
            "Export strategy: requested_mode=%s resolved_mode=%s species_input_convention=%s tp_methods=%s also_export_naive=%s",
            self.export_mode,
            resolved_mode,
            self.species_input_convention,
            tp_methods if tp_methods else ("n/a",),
            self.also_export_naive,
        )

        exported_paths: dict[str, Path] = {}
        uses_uniform_1d = self._uses_uniform_1d()

        if resolved_mode == "fixed_size":
            disabled_checks = None
            if uses_uniform_1d:
                disabled_checks = [
                    jax.export.DisabledSafetyCheck.custom_call("uniform_1d"),
                    jax.export.DisabledSafetyCheck.custom_call("uniform_1d_cuda"),
                ]
            exported_paths["primary"] = self._export_single(
                output_path,
                export_inputs=self._build_fixed_size_export_inputs(),
                disabled_checks=disabled_checks,
            )
            if self.also_export_naive and uses_uniform_1d:
                symbolic_path = output_path.with_name(
                    f"{output_path.stem}_symbolic{output_path.suffix}"
                )
                naive_apply_model = self._build_naive_apply_model()
                self._validate_naive_equivalence(naive_apply_model)
                exported_paths["symbolic"] = self._export_single(
                    symbolic_path,
                    apply_model=naive_apply_model,
                )

        elif resolved_mode == "symbolic":
            apply_model = self.apply_model
            if uses_uniform_1d:
                apply_model = self._build_naive_apply_model()
                self._validate_naive_equivalence(apply_model)
            exported_paths["primary"] = self._export_single(
                output_path,
                apply_model=apply_model,
            )
        else:
            raise ValueError(f"Unsupported resolved export mode: {resolved_mode}")

        return exported_paths

    @classmethod
    def from_combined_model(
        cls,
        model,
        params: Any,
        box: jax.Array,
        species: jax.Array,
        apply_fn: Optional[Any] = None,
    ):
        """Create exporter from CombinedModel instance."""
        ml_model = model.ml_model
        topology = model.topology

        bonds, angles = topology.get_bonds_and_angles()
        dihedrals = topology.get_dihedrals()
        rep_pairs = topology.get_repulsive_pairs()

        if model.use_priors:
            prior_params = params.get("prior", model.prior.params)
            if getattr(model.prior, "uses_splines", False):
                export_logger.info(
                    "Spline priors detected: spline arrays will be captured as constants through "
                    "model.compute_total_energy during MLIR tracing."
                )
        else:
            prior_params = {}

        species_host = jax.device_get(jnp.asarray(species))
        species_unique = int(jnp.unique(jnp.asarray(species_host)).shape[0])
        species_min = int(jnp.min(jnp.asarray(species_host))) if species_host.size else -1
        species_max = int(jnp.max(jnp.asarray(species_host))) if species_host.size else -1
        export_logger.info(
            "Exporter setup: ml_model=%s cutoff=%.3f species_shape=%s unique_species=%d species_min=%d species_max=%d box_shape=%s bonds=%d angles=%d rep_pairs=%d dihedrals=%d",
            getattr(model, "ml_model_type", type(ml_model).__name__),
            float(ml_model.cutoff),
            tuple(jnp.asarray(species_host).shape),
            species_unique,
            species_min,
            species_max,
            tuple(jnp.asarray(box).shape),
            int(bonds.shape[0]) if getattr(bonds, "ndim", 0) > 0 else 0,
            int(angles.shape[0]) if getattr(angles, "ndim", 0) > 0 else 0,
            int(rep_pairs.shape[0]) if getattr(rep_pairs, "ndim", 0) > 0 else 0,
            int(dihedrals.shape[0]) if getattr(dihedrals, "ndim", 0) > 0 else 0,
        )

        if apply_fn is None:
            compute_with_apply = getattr(ml_model, "_compute_energy_with_apply", None)
            compute_per_atom_with_apply = getattr(ml_model, "_compute_per_atom_energy_with_apply", None)

            def default_apply_fn(
                params_,
                apply_model_,
                nneigh_fn_,
                displacement_,
                R_,
                mask_,
                box_,
                species_,
                bonds_,
                angles_,
                rep_pairs_,
                dihedrals_,
                prior_params_,
                neighbor=None,
            ):
                del nneigh_fn_, displacement_, box_, bonds_, angles_, rep_pairs_, dihedrals_, prior_params_
                if model.prior_only:
                    e_ml = jnp.asarray(0.0, dtype=jnp.float32)
                else:
                    ml_params = params_.get("ml", params_) if isinstance(params_, dict) else params_
                    if not model.use_priors and compute_per_atom_with_apply is not None:
                        e_ml = compute_per_atom_with_apply(
                            apply_model_,
                            ml_params,
                            R_,
                            mask_,
                            species_,
                            neighbor,
                        )
                    elif compute_with_apply is not None:
                        e_ml = compute_with_apply(
                            apply_model_,
                            ml_params,
                            R_,
                            mask_,
                            species_,
                            neighbor,
                        )
                    elif apply_model_ is ml_model.model_apply_fn:
                        e_ml = ml_model.compute_energy(
                            ml_params,
                            R_,
                            mask_,
                            species_,
                            neighbor,
                        )
                    else:
                        raise ValueError(
                            "This ML model does not support export-time apply_fn overrides."
                        )

                if model.use_priors:
                    r_detached = jax.lax.stop_gradient(R_)
                    r_masked = jnp.where(mask_[:, None] > 0, R_, r_detached)
                    if model.train_priors and isinstance(params_, dict) and "prior" in params_:
                        e_prior = model.prior.compute_total_energy(
                            r_masked,
                            mask_,
                            species=species_,
                            params=params_["prior"],
                        )
                    else:
                        e_prior = model.prior.compute_total_energy(
                            r_masked,
                            mask_,
                            species=species_,
                        )
                    if model.prior_only:
                        return e_prior
                    # e_ml may be a per-atom array (ndim==1) when using the
                    # per-atom export apply path.  Adding a scalar e_prior to a
                    # per-atom array would broadcast it across all n_atoms slots,
                    # overcounting by a factor of n_atoms.  Instead, distribute
                    # e_prior evenly over valid atoms so padded slots stay zero
                    # and the total remains correct.
                    e_ml_arr = jnp.asarray(e_ml)
                    if e_ml_arr.ndim == 1:
                        valid_float = (mask_ > 0).astype(e_ml_arr.dtype)
                        n_valid = jnp.maximum(jnp.sum(valid_float), jnp.ones((), dtype=e_ml_arr.dtype))
                        e_prior_per_atom = jnp.asarray(e_prior, dtype=e_ml_arr.dtype) / n_valid * valid_float
                        return e_ml_arr + e_prior_per_atom
                    return e_ml + e_prior

                return e_ml

            apply_fn = default_apply_fn

        export_mode = "auto"
        species_input_convention = "model"
        also_export_naive = False
        config = getattr(model, "config", None)
        if config is not None:
            export_mode = _normalize_export_mode(config.get("export", "mode", default="auto"))
            species_input_convention = _normalize_species_input_convention(
                config.get("export", "species_input_convention", default="model")
            )
            also_export_naive = bool(config.get("export", "also_export_naive", default=False))
            naive_equivalence_atol = config.get("export", "naive_equivalence_atol", default=None)
            naive_equivalence_per_atom_atol = config.get(
                "export",
                "naive_equivalence_per_atom_atol",
                default=1.0e-6,
            )
        else:
            naive_equivalence_atol = None
            naive_equivalence_per_atom_atol = 1.0e-6

        # Use per-atom apply function for export if the model exposes one.
        # The per-atom version returns energies of shape (n_atoms,) with zeros
        # for padded/masked atoms, so the export wrapper sums only real atoms.
        export_apply_model = getattr(ml_model, "model_export_apply_fn", None) or ml_model.model_apply_fn

        return cls(
            apply_fn=apply_fn,
            apply_model=export_apply_model,
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
            r_cutoff=ml_model.cutoff,
            combined_model=model,
            ml_model=ml_model,
            sample_positions=getattr(ml_model, "_R0", None),
            sample_neighbors=getattr(ml_model, "nbrs_init", None),
            sample_species_model=getattr(ml_model, "_species0", None),
            export_mode=export_mode,
            species_input_convention=species_input_convention,
            also_export_naive=also_export_naive,
            naive_equivalence_atol=naive_equivalence_atol,
            naive_equivalence_per_atom_atol=naive_equivalence_per_atom_atol,
        )

    def __repr__(self) -> str:
        return f"ModelExporter(r_cutoff={self.r_cutoff}, n_atoms={self.species.shape[0]})"
