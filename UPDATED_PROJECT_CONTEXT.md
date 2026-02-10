# cameo_cg — Codebase Reference

> Comprehensive technical reference for the CAMEO coarse-grained protein ML force field framework.
> Intended audience: Claude instances and developers working on this codebase.
> Last generated: 2026-02-10

---

## 1. Purpose

Train a **transferable, hybrid ML + physics coarse-grained force field** for protein simulations. Each protein residue is represented by a single bead at the Cα position. The model combines:

- **Allegro** (or MACE/PaiNN): an equivariant graph neural network that learns many-body interactions from data.
- **Physics-based priors**: classical bonded (bonds, angles, dihedrals) and non-bonded (soft-sphere repulsion) energy terms with parameters fitted via Boltzmann inversion of all-atom MD distributions.

Training uses **force matching** — minimizing `MSE(F_predicted, F_reference)` — via the `chemtrain` framework, which provides JAX-based multi-GPU/multi-node parallelism through `shard_map` and `lax.pmean`.

Training targets come from the **mdCATH** dataset (all-atom MD trajectories of diverse proteins). The trained model is exported to **MLIR** (StableHLO) for deployment in **LAMMPS** via `chemtrain-deploy`.

---

## 2. End-to-End Pipeline

```
Raw MD trajectories (HDF5 / mdCATH)
  │  data_prep/h5_dataset_npz_transform.py
  ▼
Per-protein all-atom NPZ
  │  data_prep/cg_1bead.py  (Cα extraction + aggforce projection)
  ▼
Per-protein CG NPZ  (positions, projected forces, species, mask)
  │  data_prep/pad_and_combine_datasets.py
  ▼
Combined padded NPZ  (all proteins padded to N_max, global species map)
  │  data_prep/prior_fitting_script.py  (Boltzmann inversion → YAML)
  ▼
Fitted prior parameters (YAML)  +  Combined dataset (NPZ)
  │  scripts/train.py  or  scripts/train_ensemble.py
  ▼
Trained model  (params.pkl + checkpoints)
  │  scripts/evaluate.py  /  scripts/evaluate_forces.py
  ▼
Evaluation metrics + force plots
  │  export/exporter.py  (AllegroExporter)
  ▼
MLIR file  →  LAMMPS MD simulation via chemtrain-deploy
```

The data preparation pipeline is orchestrated by `data_prep/run_pipeline.py`, which runs steps 1–4 in sequence. Training, evaluation, and export are handled by the scripts in `scripts/`.

---

## 3. Repository Structure

```
cameo_cg/
├── config/
│   ├── manager.py              # ConfigManager: typed accessor for nested YAML
│   └── types.py                # TypedDicts (EnergyComponents, ForceComponents, etc.)
│
├── data/
│   ├── loader.py               # DatasetLoader: NPZ loading, shuffling, train/val split
│   └── preprocessor.py         # CoordinatePreprocessor: box computation, centering, parking
│
├── data_prep/                  # Raw data → training-ready dataset (offline pipeline)
│   ├── run_pipeline.py         # 4-step orchestrator (H5→NPZ→CG→pad→prior_fit)
│   ├── h5_dataset_npz_transform.py   # Step 1: HDF5 → raw NPZ
│   ├── cg_1bead.py             # Step 2: all-atom → Cα CG (+ aggforce)
│   ├── pad_and_combine_datasets.py   # Step 3: pad to N_max, merge, global species map
│   ├── prior_fitting_script.py # Step 4: Boltzmann inversion → fitted priors YAML
│   └── analyze_dataset.py      # Diagnostic statistics for NPZ datasets
│
├── models/
│   ├── combined_model.py       # CombinedModel: top-level Prior + ML orchestrator
│   ├── allegro_model.py        # AllegroModel: wraps chemutils Allegro GNN
│   ├── mace_model.py           # MACEModel: wraps chemutils MACE GNN
│   ├── painn_model.py          # PaiNNModel: wraps chemutils PaiNN
│   ├── prior_energy.py         # PriorEnergy: bonds, angles, dihedrals, repulsion
│   └── topology.py             # TopologyBuilder: index arrays for chain connectivity
│
├── training/
│   ├── trainer.py              # Trainer: multi-stage pipeline + LBFGS prior pretrain
│   └── optimizers.py           # Optimizer factory (AdaBelief, Yogi, Adam, Lion, etc.)
│
├── evaluation/
│   ├── evaluator.py            # Per-frame and batch evaluation metrics
│   ├── visualizer.py           # LossPlotter + ForceAnalyzer
│   ├── combined_plot.py        # Multi-panel evaluation figures
│   └── analyze_scaling_behavior.py  # Amdahl/Gustafson scaling fits
│
├── export/
│   └── exporter.py             # AllegroExporter: MLIR export for LAMMPS
│
├── utils/
│   └── logging.py              # 5 named loggers: Data, Model, Training, Export, Eval
│
├── scripts/
│   ├── train.py                # Main training entry point (single/multi-node)
│   ├── train_ensemble.py       # Ensemble training (N seeds, best-model selection)
│   ├── evaluate.py             # Full dataset evaluation
│   ├── evaluate_forces.py      # Quick force diagnostics on random frames
│   ├── run_training.sh         # SLURM submission script (1 or N nodes)
│   └── run_scaling_sweep.sh    # Multi-device scaling benchmark
│
├── env_setup/                  # HPC environment (JUWELS Booster)
│   ├── load_modules.sh         # Module stack (JAX 0.4.34, CUDA 12, etc.)
│   ├── config.sh               # Virtual environment configuration
│   └── set_lammps_paths.sh     # LAMMPS + chemtrain-deploy library paths
│
├── md_setup/                   # LAMMPS simulation preparation
│   ├── lmp_input_gen.py        # Generate LAMMPS data file from NPZ
│   ├── inp_lammps_trained.in   # Template LAMMPS input script
│   └── submit_lammps_chemtrain.sh  # SLURM job for LAMMPS
│
├── config_template.yaml        # Full configuration reference with comments
├── CONTEXT_MD_FILES/           # Documentation and context files
└── EXTERNAL_CODE_CONTEXT.md    # Notes on external dependencies
```

---

## 4. Model Architecture

### 4.1 CombinedModel (`models/combined_model.py`)

The top-level model that composes ML and physics energy:

```
E_total = E_ml(params_ml, R, species, neighbors) + E_prior(R, mask)
```

- Two modes controlled by `model.use_priors` in the YAML config: hybrid (ML + prior) or pure ML.
- ML backbone selection via `model.ml_model` config key: `"allegro"` (default), `"mace"`, or `"painn"`. All three use the same `(init_fn, apply_fn)` interface from `chemutils`.
- Key methods:
  - `compute_energy(params, R, mask, species)` → scalar energy
  - `compute_components(params, R, mask, species)` → dict of per-term energies
  - `compute_force_components(params, R, mask, species)` → per-term force arrays via autodiff
  - `energy_fn_template(params)` → closure for chemtrain's `ForceMatching` trainer
- Applies `jax.lax.stop_gradient` to padded atom coordinates before prior computation to prevent NaN gradients.

### 4.2 ML Backends

All three backends come from `chemutils` (inside `chemtrain-deploy`). They share the same `hk.transform` / `(init_fn, apply_fn)` interface:

- **AllegroModel** (`allegro_model.py`): Equivariant message-passing GNN. Configurable sizes: `"default"` (3 layers, 24 radial basis), `"large"` (4 layers, 36), `"med"` (3 layers, 18). Uses `jax_md` Dense neighbor lists (free-space, no PBC).
- **MACEModel** (`mace_model.py`): Multi Atomic Cluster Expansion. Drop-in replacement for Allegro (same interface). Computes per-node energies instead of per-edge.
- **PaiNNModel** (`painn_model.py`): Polarizable Atom Interaction Neural Network. Third option.

### 4.3 PriorEnergy (`models/prior_energy.py`)

Four physics-based energy terms for the linear CG chain:

| Term | Formula | Topology source |
|------|---------|-----------------|
| Bond | `0.5 * kr * Σ(r - r0)²` | Consecutive pairs (i, i+1) |
| Angle | `Σ_n [a_n cos(nθ) + b_n sin(nθ)]` (Fourier series) | Triplets (i, i+1, i+2) |
| Dihedral | `Σ_n [k_n (1 + cos(nφ - γ_n))]` | Quadruplets (i, i+1, i+2, i+3) |
| Repulsive | `ε * Σ(σ/r)⁴` (soft-sphere) | Non-bonded pairs with sequence separation ≥ 6 |

Each term is scaled by a configurable weight. Default: bond=0.5, angle=0.25, dihedral=0.25, repulsive=1.0. The rationale: fitted terms (bond/angle/dihedral) are derived from Boltzmann inversion of correlated distributions so they should be weighted to avoid double-counting, while the repulsive term is manually chosen and should run at full strength.

**Numerical safety** (critical for multi-protein padded datasets):
- `_safe_norm()`: Custom VJP that returns zero gradient for zero-length vectors (padded atoms at same location).
- `_safe_atan2()`: Custom VJP adding epsilon to denominator to avoid 0/0 at (y=0, x=0).
- `jax.lax.stop_gradient` applied to intermediate geometry values (r, θ, φ) for invalid entries before they enter the energy computation.

### 4.4 TopologyBuilder (`models/topology.py`)

Generates static index arrays for the linear chain:
- `precompute_chain_topology(N_max)` → bonds (N-1, 2) and angles (N-2, 3)
- `precompute_dihedrals(N_max)` → dihedrals (N-3, 4)
- `precompute_repulsive_pairs(N_max, min_sep=6)` → non-bonded pairs with sequence separation ≥ 6

Note: `get_excluded_volume_pairs(min_sep=2, max_sep=5)` exists but is **not currently used**. This is a known gap — residues at separation 2–5 have no repulsive interaction.

---

## 5. Training Pipeline

### 5.1 Entry Point: `scripts/train.py`

```
config.yaml → JAX distributed init → DatasetLoader → CoordinatePreprocessor
  → CombinedModel → Trainer.train_full_pipeline() → AllegroExporter → LossPlotter
```

CLI: `python train.py <config.yaml> [job_id] [--resume checkpoint.pkl|auto]`

JAX distributed initialization is manual: `jax.distributed.initialize()` with explicit `local_device_ids=[0,1,2,3]`. This is necessary because SLURM uses 1 task per node (not 1 per GPU), and JAX's auto-detection would only see GPU 0.

### 5.2 Training Stages (Trainer: `training/trainer.py`)

`train_full_pipeline(resume_from=None)` runs up to three stages:

1. **Prior Pretrain** (optional, `pretrain_prior: true`):
   - LBFGS optimization of prior parameters only (ML params frozen).
   - Loss: L2 force matching `Σ((F_pred - F_ref)² * mask) / n_real_atoms`.
   - Multi-node: rank 0 runs LBFGS, broadcasts fitted params to all ranks.
   - Convergence: `grad_norm < tol_grad` after `min_steps`.
   - Updates `model.prior.params` in place.

2. **Stage 1** (e.g., AdaBelief, 100 epochs):
   - Full model training via chemtrain's `ForceMatching` trainer.
   - Learning rate: warmup + cosine decay schedule.

3. **Stage 2** (e.g., Yogi, optional fine-tuning):
   - Continues from Stage 1 params.
   - Typically lower learning rate.

Checkpoint resume detects completed stages from metadata and skips them.

### 5.3 Optimizer Factory (`training/optimizers.py`)

`create_optimizer(name, config)` builds optax chains: `gradient_clip → weight_decay → optimizer`.

Supported: AdaBelief, Yogi, Adam, Lion, Polyak SGD, Fromage. All use `optax.warmup_cosine_decay_schedule` for LR scheduling.

### 5.4 Loss Function

```
Loss = γ_F × MSE(F_pred, F_ref) + γ_U × MSE(E_pred, E_ref)
```

Typically `γ_F=1.0, γ_U=0.0` (force-matching only). chemtrain reports **MSE, not RMSE**.

### 5.5 Ensemble Training (`scripts/train_ensemble.py`)

Trains N models with different random seeds (same data split), computes variance, selects best model by validation loss. Optionally saves all models or just the best.

---

## 6. Data Pipeline

### 6.1 Offline Preparation (`data_prep/`)

Orchestrated by `run_pipeline.py` with four steps:

1. **HDF5 → Raw NPZ** (`h5_dataset_npz_transform.py`): Extracts frames from mdCATH HDF5 files. Supports temperature-group filtering and kcal/mol → eV conversion.

2. **All-atom → CG** (`cg_1bead.py`): Extracts Cα atoms, optionally applies `aggforce` constraint-aware optimal force projection. Supports per-AA species mapping (20 types) or 4-way charge-based grouping.

3. **Pad & Combine** (`pad_and_combine_datasets.py`): Pads all proteins to global N_max with zeros. Creates a unified species ID mapping across proteins. Generates validity masks (1=real, 0=padded). Can output one combined NPZ or separate padded NPZs (`--no_combine`).

4. **Prior Fitting** (`prior_fitting_script.py`): Boltzmann inversion on bond length, angle, and dihedral distributions. Fits harmonic bond params (r0, kr), Fourier angle coefficients (a, b), periodic dihedral params (k, γ). Outputs YAML + diagnostic plots.

### 6.2 NPZ Data Format

```python
{
    "R":       float32[n_frames, n_atoms, 3],   # Positions (Å)
    "F":       float32[n_frames, n_atoms, 3],   # Forces (kcal/mol/Å)
    "species": int32[n_frames, n_atoms],         # AA type IDs (0-indexed)
    "mask":    float32[n_frames, n_atoms],        # 1=real, 0=padded
    "Z":       int32[n_atoms],                    # Atomic numbers (unused for CG)
    "N_max":   int,                               # Max atoms across all proteins
    "aa_to_id": dict,                             # AA name → species ID mapping
}
```

### 6.3 Runtime Data Loading (`data/`)

- **DatasetLoader** (`loader.py`): Loads NPZ, shuffles with seed, applies `max_frames` limit, provides train/val split.
- **CoordinatePreprocessor** (`preprocessor.py`):
  - `compute_box_extent(R, mask, cutoff)` → `box = max_range + 2 * buffer_multiplier * cutoff`
  - `center_and_park(R, mask, box)` → centers real atoms at origin, parks padded atoms at `0.95 * box_extent` to keep them outside neighbor list range.

---

## 7. Multi-Node Distributed Training

Architecture: 1 process per node, each process manages 4 local GPUs.

| Setting | Value |
|---------|-------|
| Processes per node | 1 (NOT 1 per GPU) |
| GPUs per process | 4 (`CUDA_VISIBLE_DEVICES=0,1,2,3`) |
| Gradient sync | chemtrain's `shard_map` + `lax.pmean` |
| Coordinator | FQDN of first SLURM node, port `29400 + (JOB_ID % 1000)` |

After `jax.distributed.initialize()`, `jax.devices()` returns all GPUs across all nodes. chemtrain's `Mesh(jax.devices(), ...)` spans them all.

```bash
# Single-node (4 GPUs):
sbatch scripts/run_training.sh config.yaml

# Multi-node (8 GPUs):
sbatch --nodes=2 scripts/run_training.sh config.yaml
```

---

## 8. Export & Deployment

### 8.1 MLIR Export (`export/exporter.py`)

`AllegroExporter` extends chemtrain's `Exporter`:
- Converts the trained model to MLIR (StableHLO) via `chemtrain-deploy`.
- Handles LAMMPS 1-based species indexing (converts to 0-based internally).
- Includes prior topology (bonds, angles, dihedrals, repulsive pairs) in the exported function.
- Factory: `AllegroExporter.from_combined_model(model, params, box, species)`.

### 8.2 LAMMPS Integration

- `md_setup/lmp_input_gen.py`: Generates a LAMMPS `.data` file from a NPZ frame.
- `md_setup/inp_lammps_trained.in`: Template LAMMPS input script using `pair_style chemtrain_deploy cuda12`.
- Requires LAMMPS built with the chemtrain-deploy plugin (see `env_setup/LAMMPS_build.md`).
- CUDA compatibility workaround: copy `xla_cuda_plugin.so` from JAX site-packages, rename to `pjrt_plugin.xla_cuda12.so`, and modify `connector/compiler.cpp` to use platform `"cuda12"`.

---

## 9. Configuration

All parameters are in a single YAML file (see `config_template.yaml`). Key sections:

```yaml
seed: 193749
model_context: "allegro_cg_protein_4zohB01"
protein_name: "4zohB01"

data:
  path: "data_prep/datasets/..."
  max_frames: 2500

preprocessing:
  buffer_multiplier: 2.0     # Box extent buffer
  park_multiplier: 0.95      # Parking location for padded atoms

model:
  use_priors: true            # true = hybrid, false = pure ML
  ml_model: "allegro"         # "allegro", "mace", or "painn"
  allegro_size: "default"     # "default", "large", "med"
  cutoff: 10.0                # Neighbor list cutoff (Å)
  dr_threshold: 1.0           # Neighbor list rebuild threshold
  priors:
    weights: {bond: 0.5, angle: 0.25, dihedral: 0.25, repulsive: 1.0}
    r0: 3.84                  # Bond equilibrium length
    kr: 154.5                 # Bond force constant
    a: [...]                  # Angle Fourier cosine coefficients
    b: [...]                  # Angle Fourier sine coefficients
    epsilon: 1.0              # Repulsive energy scale
    sigma: 3.0                # Repulsive length scale
    k_dih: [...]              # Dihedral force constants
    gamma_dih: [...]          # Dihedral phase angles

training:
  pretrain_prior: true
  epochs_adabelief: 30
  epochs_yogi: 50
  batch_per_device: 2

optimizer:
  adabelief: {lr: 0.05, peak_lr: 0.01, ...}
  yogi: {lr: 0.001, ...}

ensemble:
  enabled: false
  n_models: 5
  base_seed: 42
```

Accessed via `ConfigManager("config.yaml")` which provides typed getter methods (e.g., `config.get_cutoff()`, `config.use_priors()`, `config.get_ml_model_type()`).

---

## 10. External Dependencies

| Package | Role | Source |
|---------|------|--------|
| `chemtrain` | ForceMatching trainer, shard_map, DataLoaders | `github.com/tummfm/chemtrain` (pinned commit) |
| `chemtrain-deploy` / `chemutils` | Allegro/MACE/PaiNN model implementations, MLIR export | `github.com/tummfm/chemtrain-deploy` |
| JAX / jax_md | Autodiff, neighbor lists, distributed training | Module on JUWELS (v0.4.34) |
| optax | Optimizers and LR schedules | pip |
| jax_sgmc | NumpyDataLoader | pip |
| aggforce | Constraint-aware force projection for CG | pip / separate install |
| LAMMPS | MD simulation engine | Built from source with chemtrain-deploy plugin |

**Installation layout**: `chemtrain-deploy` is cloned first, then `chemtrain` is cloned inside `chemtrain-deploy/external/chemtrain/`. The `cameo_cg` code lives inside this `chemtrain` directory (or is symlinked as `clean_code_base`).

---

## 11. Known Issues & Design Decisions

1. **Excluded volume gap**: Residues at sequence separation 2–5 have no repulsive interaction. The code for generating these pairs exists (`TopologyBuilder.get_excluded_volume_pairs()`) and was partially implemented in `PriorEnergy`, but is commented out.

2. **Prior weight scheme**: Weights are applied at the energy level, which also scales forces. This is by design for fitted terms but is debatable for the manually chosen repulsive term (currently addressed by setting repulsive weight to 1.0).

3. **NumpyDataLoader patch**: `Trainer` monkey-patches `jax_sgmc.NumpyDataLoader` to ensure `cache_size >= 1` (workaround for a bug in the library).

4. **CG forces don't sum to zero**: Expected behavior — aggforce projection does not enforce net-zero forces on the CG subsystem.

5. **SO3LR integration**: Planned but blocked by a JAX version conflict (SO3LR requires 0.5.3, environment has 0.4.34). See `CONTEXT_MD_FILES/PLAN_SO3LR.md`.

---

## 12. HPC Environment (JUWELS Booster)

- Partition: `booster` (A100 GPUs, 4 per node)
- SLURM accounts: `cameo` or `atmlaml`
- Module stack: GCC 13.3, Python 3.12, CUDA 12, JAX 0.4.34, cuDNN 9.5, NCCL
- Virtual environment: `clean_booster_env` (activated via `source`)
- Key environment variables set at runtime: `CUDA_VISIBLE_DEVICES`, `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_FLAGS` (CUDA data dir, autotune off)

---

## 13. Glossary

| Term | Meaning |
|------|---------|
| CG | Coarse-grained — reduced-resolution molecular model |
| Force matching | Training by minimizing MSE between predicted and reference forces |
| Boltzmann inversion | Fitting potential parameters from equilibrium distribution histograms: `U(x) = -kT ln P(x)` |
| N_max | Maximum number of CG beads across all proteins in the dataset (smaller proteins are zero-padded to this size) |
| Parking | Moving padded (fake) atoms to a distant corner of the simulation box so they don't appear in neighbor lists |
| MLIR / StableHLO | Intermediate representation used by XLA; the format for deploying JAX models to LAMMPS |
| aggforce | Library for constraint-aware optimal force projection from all-atom to CG resolution |
| mdCATH | Large-scale dataset of all-atom MD simulations for diverse proteins from the CATH classification |
