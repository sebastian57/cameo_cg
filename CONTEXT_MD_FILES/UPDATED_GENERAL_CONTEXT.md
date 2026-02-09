# Updated General Project Context

> **Purpose**: Comprehensive reference for the CAMEO coarse-grained protein ML force field codebase.
> **Last Updated**: 2026-02-09

---

## 1. Project Goal

Train a **hybrid ML + physics** force field for **1-bead-per-residue coarse-grained proteins** using the Allegro equivariant GNN combined with classical physics-based priors. The model learns to predict CG forces from CG positions via force matching, enabling fast MD simulations of protein dynamics through LAMMPS.

---

## 2. Full Pipeline Overview

```
Raw MD (HDF5/mdcath)
    | [data_prep/h5_dataset_npz_transform.py]
Raw NPZ (per-protein, all-atom)
    | [data_prep/cg_1bead.py]
CG NPZ (CA beads, aggforce projection)
    | [data_prep/pad_and_combine_datasets.py]
Combined NPZ (padded to N_max, global species mapping)
    | [data_prep/prior_fitting_script.py]
Fitted Priors YAML (bond/angle/dihedral/repulsive params)
    | [scripts/train.py or train_ensemble.py]
Trained Model (params.pkl + checkpoints)
    | [scripts/evaluate.py / evaluate_forces.py]
Evaluation Metrics + Force Plots
    | [export/exporter.py]
MLIR for LAMMPS deployment
```

---

## 3. Directory Structure

```
cameo_cg/
├── config/                  # Configuration system
│   ├── manager.py           # ConfigManager: typed YAML accessor
│   └── types.py             # TypedDicts for all data structures
├── data/                    # Data loading & preprocessing
│   ├── loader.py            # DatasetLoader: NPZ loading, species mapping
│   └── preprocessor.py      # CoordinatePreprocessor: centering, boxing, parking
├── data_prep/               # Raw data → training-ready pipeline
│   ├── run_pipeline.py      # 4-step orchestrator
│   ├── h5_dataset_npz_transform.py  # HDF5 → raw NPZ
│   ├── cg_1bead.py          # All-atom → CA coarse-graining
│   ├── pad_and_combine_datasets.py  # Padding + merging
│   ├── prior_fitting_script.py      # Boltzmann inversion for priors
│   └── analyze_dataset.py   # Dataset statistics & diagnostics
├── models/                  # Model architecture
│   ├── combined_model.py    # CombinedModel: Prior + Allegro orchestrator
│   ├── allegro_model.py     # AllegroModel: GNN wrapper + neighbor lists
│   ├── prior_energy.py      # PriorEnergy: bonds, angles, dihedrals, repulsion
│   └── topology.py          # TopologyBuilder: chain connectivity indices
├── training/                # Training orchestration
│   ├── trainer.py           # Trainer: multi-stage pipeline + LBFGS pretrain
│   └── optimizers.py        # Optimizer factory (optax-based)
├── scripts/                 # Entry points
│   ├── train.py             # Main training script (single/multi-node)
│   ├── train_ensemble.py    # Ensemble training (N seeds, best selection)
│   ├── evaluate.py          # Full dataset evaluation
│   ├── evaluate_forces.py   # Quick force diagnostics on random frames
│   ├── analyze_scaling.py   # Parse scaling test results
│   ├── run_training.sh      # SLURM submission script
│   └── run_scaling_sweep.sh # Multi-device scaling benchmark
├── evaluation/              # Evaluation & visualization
│   ├── evaluator.py         # Evaluator: energy/force metrics
│   ├── visualizer.py        # LossPlotter + ForceAnalyzer
│   ├── combined_plot.py     # Multi-panel evaluation figures
│   └── analyze_scaling_behavior.py  # Amdahl/Gustafson scaling fits
├── export/                  # Model export
│   └── exporter.py          # AllegroExporter: MLIR for LAMMPS
├── utils/                   # Shared utilities
│   └── logging.py           # 5 named loggers (Data/Model/Training/Export/Eval)
├── env_setup/               # HPC environment
│   ├── load_modules.sh      # JUWELS module stack
│   ├── config.sh            # Virtual environment setup
│   └── set_lammps_paths.sh  # LAMMPS + chemtrain-deploy paths
├── config_*.yaml            # Training configuration templates
└── md_setup/                # LAMMPS input generation
    └── lmp_input_gen.py
```

---

## 4. Model Architecture

### 4.1 CombinedModel (models/combined_model.py)

The top-level model that composes ML and physics energy terms:

```
E_total = E_allegro(params_allegro, R, species, neighbors) + E_prior(R, mask)
```

- **Modes**: `use_priors: true` (hybrid) or `use_priors: false` (pure Allegro)
- **Key methods**:
  - `compute_energy()` → scalar total energy
  - `compute_components()` → dict of individual energy terms
  - `compute_force_components()` → per-term force decomposition via autodiff
  - `energy_fn_template(params)` → closure for chemtrain's ForceMatching trainer
- **Critical**: Applies `stop_gradient` to padded atom coordinates before prior computation to prevent NaN gradients from undefined geometry (v/||v|| at v=0)

### 4.2 AllegroModel (models/allegro_model.py)

Wrapper around the Allegro equivariant GNN from `chemutils`:

- Uses `jax_md` neighbor lists (free-space, no PBC)
- Configurable sizes: `"default"` (3 layers, 24 radial basis), `"large"` (4 layers, 36), `"med"` (3 layers, 18)
- Handles coordinate masking and species validation for padded atoms
- Parameters initialized via `jax.random.PRNGKey`

### 4.3 PriorEnergy (models/prior_energy.py)

Physics-based energy with four terms:

| Term | Formula | Topology |
|------|---------|----------|
| Bond | `0.5 * kr * sum((r - r0)^2)` | Consecutive pairs (i, i+1) |
| Angle | `sum_n [a_n cos(n*theta) + b_n sin(n*theta)]` | Triplets (i, i+1, i+2) |
| Dihedral | `sum_n [k_n (1 + cos(n*phi - gamma_n))]` | Quadruplets (i, i+1, i+2, i+3) |
| Repulsive | `epsilon * sum((sigma/r)^4)` | Pairs with seq. sep. >= 6 |

**Default weights**: bond=0.5, angle=0.25, dihedral=0.25, repulsive=1.0

**Numerical safety** (critical for multi-protein padded datasets):
- `_safe_norm()`: Custom VJP returning 0 gradient for zero vectors
- `_safe_atan2()`: Custom VJP with epsilon in denominator at (0,0)
- `stop_gradient` applied to intermediate values (r, theta, phi) for invalid entries
- `jnp.where` masks energy contributions from padded atoms

### 4.4 TopologyBuilder (models/topology.py)

Generates index arrays for the linear chain topology:
- `precompute_chain_topology(N_max)` → bonds (N-1, 2) and angles (N-2, 3)
- `precompute_dihedrals(N_max)` → dihedrals (N-3, 4)
- `precompute_repulsive_pairs(N_max, min_sep=6)` → non-bonded pairs
- `get_excluded_volume_pairs(min_sep=2, max_sep=5)` → available but **not currently used**

---

## 5. Training Pipeline

### 5.1 Entry Point: scripts/train.py

```
config.yaml → _initialize_jax_distributed() → DatasetLoader → CoordinatePreprocessor
    → CombinedModel → Trainer.train_full_pipeline() → AllegroExporter → LossPlotter
```

**Key responsibilities**:
1. Initialize JAX distributed (manual init with `local_device_ids=[0,1,2,3]`)
2. Load and preprocess data (center, park padded atoms)
3. Split train/validation
4. Create CombinedModel and Trainer
5. Run full training pipeline (pretrain + stage1 + stage2)
6. Export to MLIR + pickle params
7. Plot loss curves from log file

**CLI**: `python train.py <config.yaml> [job_id] [--resume checkpoint.pkl|auto]`

### 5.2 Trainer (training/trainer.py)

Orchestrates the multi-stage training pipeline:

**`train_full_pipeline(resume_from=None)`**:
1. **Prior Pretrain** (optional, `pretrain_prior: true`):
   - LBFGS optimization of prior params only (Allegro frozen)
   - Loss: L2 force matching `sum((F_pred - F_ref)^2 * mask) / n_real`
   - Multi-node: rank 0 optimizes, broadcasts to all
   - Convergence: `grad_norm < tol_grad` after `min_steps`
2. **Stage 1** (e.g., AdaBelief, 100 epochs):
   - Full model training via chemtrain's `ForceMatching` trainer
   - Learning rate: warmup + cosine decay schedule
3. **Stage 2** (e.g., Yogi, optional fine-tuning):
   - Continues from Stage 1 params
   - Typically lower learning rate

**Checkpoint resume**: Detects stage and epoch from metadata, skips completed stages.

**`train_stage(optimizer_name, epochs, start_epoch, checkpoint_freq)`**:
- Creates optimizer via factory
- Wraps data in chemtrain `DataLoaders`
- Instantiates `ForceMatching` trainer with energy function template
- Saves checkpoints with `.meta.pkl` metadata

### 5.3 Optimizer Factory (training/optimizers.py)

`create_optimizer(name, config)` builds optax chains:

| Optimizer | Default Params | Notes |
|-----------|---------------|-------|
| AdaBelief | beta1=0.9, beta2=0.999, eps=1e-8 | Good default for Stage 1 |
| Yogi | beta1=0.9, beta2=0.999, eps=1e-6 | Fine-tuning Stage 2 |
| Adam | beta1=0.9, beta2=0.999, eps=1e-8 | Standard alternative |
| Lion | beta1=0.9, beta2=0.99 | Memory-efficient |
| Polyak SGD | f_star parameter | Adaptive step size |
| Fromage | Fixed learning rate | Simple SGD |

**Schedule**: `optax.warmup_cosine_decay_schedule(init_value, peak_value, warmup_steps, decay_steps, end_value)`

**Chain**: gradient_clip → weight_decay → optimizer

### 5.4 Loss Function

```
Loss = gamma_F * MSE(F_pred, F_ref) + gamma_U * MSE(E_pred, E_ref)
```
- Typically `gamma_F=1.0, gamma_U=0.0` (force-matching only)
- chemtrain reports **MSE**, not RMSE

---

## 6. Data Pipeline

### 6.1 Data Preparation (data_prep/)

**4-step pipeline** orchestrated by `run_pipeline.py`:

1. **HDF5 → Raw NPZ**: Extract frames from mdcath HDF5 datasets
2. **All-atom → CG**: Coarse-grain to CA beads using aggforce projection
3. **Pad & Combine**: Pad all proteins to N_max, create global species mapping
4. **Prior Fitting**: Boltzmann inversion to fit bond/angle/dihedral parameters

### 6.2 Data Loading (data/loader.py)

`DatasetLoader(npz_path, max_frames=None, seed=42)`:
- Loads single NPZ file or directory of NPZ files (auto-concatenation)
- Shuffles frames with seed before applying max_frames limit
- Provides `get_batch()`, `get_all()`, and properties (n_frames, N_max)

### 6.3 Preprocessing (data/preprocessor.py)

`CoordinatePreprocessor(config)`:
- `compute_box_extent(R, mask, cutoff)` → box = max_range + 2 * buffer * cutoff
- `center_and_park(R, mask, box)` → centers real atoms, parks padded at 0.95 * box

---

## 7. NPZ Data Format

```python
{
    "R": float32[n_frames, n_atoms, 3],      # Positions (Angstrom)
    "F": float32[n_frames, n_atoms, 3],      # Forces (kcal/mol/Angstrom)
    "species": int32[n_frames, n_atoms],     # AA type IDs (0-indexed)
    "mask": float32[n_frames, n_atoms],      # 1=real, 0=padded
    "Z": int32[n_atoms],                     # Atomic numbers (unused for CG)
    "N_max": int,                            # Max atoms across all proteins
    "aa_to_id": dict,                        # AA name → species ID mapping
}
```

**Padding**: Smaller proteins padded to N_max with zeros; padded atoms "parked" at 0.95 * box_extent to avoid neighbor list interactions.

---

## 8. Multi-Node Distributed Training

### Architecture
- **1 process per node**, each process manages **4 local GPUs**
- JAX distributed: `jax.distributed.initialize()` with manual `local_device_ids=[0,1,2,3]`
- chemtrain's `shard_map` + `lax.pmean` for gradient synchronization across all GPUs

### Why Manual Init (not SLURM auto-detection)
JAX auto-detection uses `SLURM_LOCALID` → assumes 1 GPU per task. With `--ntasks-per-node=1`, each task gets `SLURM_LOCALID=0` → only GPU 0 per node.

### Coordinator Setup
- Address: FQDN of first node (e.g., `jwb0262.juwels`)
- Port: `29400 + (SLURM_JOB_ID % 1000)` (job-specific to avoid conflicts)

### SLURM Configuration (run_training.sh)
```bash
#SBATCH --account=atmlaml
#SBATCH --nodes=1              # Override with sbatch --nodes=N
#SBATCH --ntasks-per-node=1    # 1 process per node
#SBATCH --gpus-per-task=4      # 4 GPUs per process
#SBATCH --partition=booster
```

---

## 9. Configuration System

### ConfigManager (config/manager.py)

Typed accessor for YAML configuration with defaults:

```python
config = ConfigManager("config.yaml")
config.get_seed()                    # int
config.get_data_path()               # str
config.get_cutoff()                  # float (default: 10.0)
config.get_allegro_config("default") # dict of Allegro hyperparams
config.get_prior_params()            # dict of prior params
config.get_epochs("adabelief")       # int
config.use_priors()                  # bool
config.pretrain_prior_enabled()      # bool
config.is_ensemble_enabled()         # bool
```

### Available Config Templates

| File | Description |
|------|-------------|
| `config_template.yaml` | Standard: Prior+Allegro, 4zohB01, 2500 frames |
| `config_preprior.yaml` | With LBFGS prior pretrain, 2g4q4z5k, 10000 frames |
| `config_allegro_only.yaml` | Pure Allegro (use_priors: false) |
| `config_excluded_vol.yaml` | Excluded volume prior testing |
| `config_4pro.yaml` | 4-protein combined dataset |
| `config_timing_test.yaml` | Scaling benchmarks |

### Key Config Sections

```yaml
seed: 193749
data:
  path: "data_prep/datasets/dataset.npz"
  max_frames: 2500
preprocessing:
  buffer_multiplier: 2.0
  park_multiplier: 0.95
model:
  use_priors: true
  cutoff: 12.0
  allegro_size: "default"    # "default" | "large" | "med"
  priors:                    # Bond/angle/dihedral/repulsive parameters
    r0: 3.81
    kr: 19.8
    ...
training:
  pretrain_prior: false
  stage1_optimizer: "adabelief"
  stage2_optimizer: "yogi"
  epochs_adabelief: 100
  epochs_yogi: 0             # 0 = skip stage 2
  batch_per_device: 8
  val_fraction: 0.1
  gammas: {F: 1.0, U: 0.0}
  checkpoint_freq: 0
ensemble:
  enabled: false
  n_models: 5
  base_seed: 42
  save_all_models: false
```

---

## 10. Evaluation & Visualization

### Evaluator (evaluation/evaluator.py)
- `evaluate_frame(R, F_ref, mask, species)` → energy, RMSE, MAE, max error, components
- `evaluate_batch(...)` → batch statistics (mean/std over frames)

### ForceAnalyzer (evaluation/visualizer.py)
- Force component scatter plots (X/Y/Z predicted vs reference)
- Force magnitude comparison
- Force error distribution histograms

### LossPlotter (evaluation/visualizer.py)
- Parses training log files (multi-node aware, deduplicates epochs)
- Loss trajectory with stage boundary annotations

### Scaling Analysis (evaluation/analyze_scaling_behavior.py)
- Fits Amdahl's law, Gustafson's law, power-law models
- Predicts GPU-hours for target device counts
- Generates speedup/efficiency/time plots

---

## 11. Export (export/exporter.py)

`AllegroExporter` extends chemtrain's `Exporter`:
- Converts trained model to MLIR (StableHLO) for LAMMPS ML-IAP
- Handles LAMMPS 1-based species indexing (converts to 0-based internally)
- Includes prior topology (bonds, angles, dihedrals, repulsive pairs) in export
- Factory: `AllegroExporter.from_combined_model(model, params, box, species)`

**Output formats**: `.mlir` (LAMMPS), `.pkl` (Python parameters), `.json` (ensemble metadata)

---

## 12. Logging (utils/logging.py)

Five named loggers with `[Name] message` format:

| Logger | Usage |
|--------|-------|
| `data_logger` | Dataset loading, preprocessing, splits |
| `model_logger` | Architecture init, species detection |
| `training_logger` | Optimization progress, LBFGS, checkpoints |
| `export_logger` | MLIR/pickle output |
| `eval_logger` | Metrics, evaluation results |

`set_log_level(logging.DEBUG)` adjusts all loggers globally.

---

## 13. External Dependencies

| Package | Purpose |
|---------|---------|
| JAX / jax_md | Autodiff, neighbor lists, distributed training |
| chemtrain | ForceMatching trainer, shard_map, DataLoaders |
| chemutils | Allegro model implementation |
| optax | Optimizers and learning rate schedules |
| jax_sgmc | NumpyDataLoader |
| chemtrain-deploy | MLIR export for LAMMPS |
| aggforce | Constraint-aware force projection for CG |
| numpy / matplotlib | Data handling, plotting |

---

## 14. Known Issues & Design Decisions

1. **Excluded volume gap**: Residues at separation 2-5 have no repulsion. Code exists in `TopologyBuilder.get_excluded_volume_pairs()` and was partially implemented in `PriorEnergy` but is currently commented out.

2. **NaN prevention for padded atoms**: Two-layer fix:
   - `CombinedModel`: `stop_gradient` on padded R before prior computation
   - `PriorEnergy`: `stop_gradient` on intermediate geometry (r, theta, phi) for invalid entries, plus custom VJPs (`_safe_norm`, `_safe_atan2`)

3. **Force imbalance**: CG forces from aggforce projection don't sum to zero (expected behavior).

4. **NumpyDataLoader patch**: `Trainer._apply_dataloader_patch()` monkey-patches jax_sgmc's NumpyDataLoader to ensure `cache_size >= 1`.

5. **JAX KeyArray compatibility**: `evaluate_forces.py` patches `jax.random.KeyArray` (removed in newer JAX) to use `jax.Array`.

---

## 15. Force Magnitude Context

| Metric | Typical Range | Notes |
|--------|--------------|-------|
| Mean \|F\| | 15-30 kcal/mol/A | 1-bead CG proteins |
| Max \|F\| | 50-100 kcal/mol/A | |
| RMS per component | 10-20 kcal/mol/A | |
| Good MSE | 3-6 | ~15-20% relative error |
| Fair MSE | 10-15 | ~25-30% relative error |
| CG RMSE (literature) | 5-15 | Irreducible CG error |
