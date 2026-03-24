# Future Work — Architectural Improvements

Items identified during the cleanup but deferred until the current version is validated.

---

## 1. Pluggable Prior Terms Registry

Currently, adding a new prior energy term requires editing `PriorEnergy` internals
(the `__init__`, `compute_energy`, parameter builders) and the VJP tuple in
`CombinedModel.compute_force_components`. A registry pattern — similar to what
exists for ML models (`@register_ml_model`) and optimizers (`@register_optimizer`) —
would allow new prior terms to be self-contained modules that register themselves.

**Scope:** Medium-large. Requires refactoring `PriorEnergy` to iterate over
registered terms dynamically, and reworking the VJP force decomposition to handle
a variable number of energy components.

---

## 2. AA Typing Centralization

Amino acid group classification (positive/negative/polar/nonpolar), charge
tables, and residue name mappings are defined in multiple places:
- `models/prior_energy.py` (group indices, charge tables)
- `data_prep/prior_fitting_script.py` (residue-specific angle handling)
- `data_prep/cg_1bead.py` (species mapping)

A single `aa_typing.py` module (or a shared data file) would eliminate drift
risk and make it easier to support non-standard residues or modified amino acids.

---

## 3. `data_prep/` Package Layout

The `data_prep/` scripts rely on `sys.path` manipulation and flat imports
(e.g., `from h5_dataset_npz_transform import ...`). This works when running
scripts directly but breaks under `python -m data_prep.run_pipeline`.

Making `data_prep/` a proper Python package with relative imports would improve
robustness and IDE support. The `__init__.py` is now in place; the next step
is converting flat imports to relative ones.

---

## 4. Force Components VJP Tuple Expansion

`CombinedModel.compute_force_components()` uses a fixed-length tuple for the
VJP decomposition. The tuple length (`n=7` for priors-enabled) is hardcoded
and must be manually updated when adding prior terms. This is fragile and
tightly coupled to `PriorEnergy.compute_energy()`.

A dynamic approach — e.g., returning a dict from `compute_energy` and using
`jax.tree_util` for the VJP — would decouple force decomposition from the
number of energy components.

---

## 5. Exporter Per-Backend Specialization

`ModelExporter` currently works for any `CombinedModel` backend, but uses
Allegro-specific assumptions about the graph structure (e.g.,
`SimpleSparseNeighborList`, `nbr_order = [1, 1]`). If MACE or PaiNN export
requires different graph types or neighbor orderings, the exporter would need
a backend dispatch mechanism.

---

## 6. Checkpoint Format Migration

The current checkpoint format is Python pickle (`.pkl`), which creates tight
coupling to the exact class hierarchy and JAX version. For long-term
reproducibility, consider migrating to a format that stores arrays and metadata
separately (e.g., safetensors, structured NPZ, or Orbax).

The `params['ml']` key rename (from `'allegro'`) is the first step toward a
stable, backend-agnostic checkpoint schema.

---

## 7. `data_prep/` Topology Duplication

`prior_fitting_script.py` has its own `build_bonds_angles_dihedrals` that uses
residue-ID-sorted ordering, while `models/topology.py` uses sequential
0..N-1 ordering. Both are correct for their contexts (fitting on raw data vs
training on padded systems), but sharing a common geometry kernel
(bond distances, angle computations, dihedral computations) would reduce
maintenance surface.
