# cameo_cg — Codebase Reference

> Comprehensive technical reference for the CAMEO coarse-grained protein ML force field framework.
> Intended audience: Claude instances and developers working on this codebase.
> Last updated: 2026-02-12

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
  │  data_prep/prior_fitting_script.py  (Boltzmann inversion → YAML/NPZ)
  ▼
Fitted prior parameters (YAML + optional spline NPZ)  +  Combined dataset (NPZ)
  │  scripts/train.py  or  scripts/train_ensemble.py
  ▼
Trained model  (params.pkl + checkpoints) → outputs/ logs
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
│   ├── prior_fitting_script.py # Step 4: Boltzmann inversion → fitted priors YAML/NPZ
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
│   └── logging.py              # 6 named loggers: Data, Model, Training, Export, Eval, Pipeline
│
├── scripts/
│   ├── train.py                # Main training entry point (single/multi-node)
│   ├── train_ensemble.py       # Ensemble training (N seeds, best-model selection)
│   ├── evaluate.py             # Full dataset evaluation
│   ├── evaluate_forces.py      # Quick force diagnostics (supports 3 modes: full/prior-only/ml-only)
│   ├── run_training.sh         # SLURM submission script (1 or N nodes)
│   └── run_scaling_sweep.sh    # Multi-device scaling benchmark
│
├── outputs/                    # **NEW**: Training logs and SLURM outputs
│   ├── slurm-<JOB_ID>.out      #   SLURM job output
│   ├── train_allegro_<JOB_ID>.log  #   Training logs
│   ├── .gitignore              #   Ignores all logs
│   └── README.md               #   Documentation
│
├── checkpoints_allegro/        # Training checkpoints (epoch*.pkl, stage_*.pkl)
├── exported_models/            # Final trained models (MLIR + params.pkl)
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
├── COMMANDS.md                 # Quick reference for common operations
└── EXTERNAL_CODE_CONTEXT.md    # Notes on external dependencies
```

---

## 4. Model Architecture

### 4.1 CombinedModel (`models/combined_model.py`)

The top-level model that composes ML and physics energy:

```
E_total = E_ml(params_ml, R, species, neighbors) + E_prior(R, mask)
```

- **Modes** controlled by `model.use_priors` in YAML config:
  - Hybrid (ML + prior): `use_priors: true`
  - Pure ML: `use_priors: false`
  - **Prior-only** (NEW): `prior_only=True` parameter skips ML computation entirely

- **ML backbone selection** via `model.ml_model` config key:
  - `"allegro"` (default): Equivariant message-passing GNN
  - `"mace"`: Multi Atomic Cluster Expansion
  - `"painn"`: Polarizable Atom Interaction Neural Network

- **Key methods**:
  - `compute_energy(params, R, mask, species)` → scalar energy
  - `compute_components(params, R, mask, species)` → dict of per-term energies
  - `compute_force_components(params, R, mask, species)` → per-term force arrays via autodiff
  - `energy_fn_template(params)` → closure for chemtrain's `ForceMatching` trainer

- **Critical numerical safety**: Applies `jax.lax.stop_gradient` to padded atom coordinates before prior computation to prevent NaN gradients from undefined geometry.

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

Each term is scaled by a configurable weight. **Default**: bond=0.5, angle=0.25, dihedral=0.25, repulsive=1.0. The rationale: fitted terms (bond/angle/dihedral) are derived from Boltzmann inversion of correlated distributions so they should be weighted to avoid double-counting, while the repulsive term is manually chosen and should run at full strength.

**Supports three prior types** (NEW):
1. **Parametric priors**: Histogram-fitted parameters from config YAML
2. **Spline priors**: Cubic spline PMFs loaded from NPZ file (e.g., `fitted_priors_spline.npz`)
3. **Trained priors**: Prior parameters optimized during training (via LBFGS pretrain or `train_priors=true`)

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

**JAX distributed initialization** (enhanced with error handling):
- Manual `jax.distributed.initialize()` with explicit `local_device_ids=[0,1,2,3]`
- Required because SLURM uses 1 task per node (not 1 per GPU)
- **NEW**: Wrapped in try/except with rank-tagged error messages and diagnostic output
- **NEW**: Added flush=True to all print statements for unbuffered output
- **NEW**: Prints hostname, SLURM_NODELIST, and CUDA_VISIBLE_DEVICES for debugging

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

**Checkpoint resume** detects completed stages from metadata and skips them.

**Note**: LR schedule state is **NOT** saved in checkpoints — when resuming, the learning rate schedule restarts from step 0.

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

### 5.6 SLURM Integration (`scripts/run_training.sh`)

**Enhanced with:**
- **NEW**: `#SBATCH --output=outputs/slurm-%j.out` directive
- **NEW**: Creates `outputs/` directory automatically
- **NEW**: Logs go to `outputs/train_allegro_<JOB_ID>.log`
- **NEW**: `srun -l` flag prepends task ID to each output line
- **NEW**: `python3 -u` flag for unbuffered output (critical for debugging)

---

## 6. Data Pipeline

### 6.1 Offline Preparation (`data_prep/`)

**Orchestrated by `run_pipeline.py`** (newly refactored to follow framework conventions):

**Recent improvements**:
- Uses framework's `pipeline_logger` from `utils/logging`
- Comprehensive type hints on all functions
- Enhanced docstrings with Args/Returns/Examples
- Documented environment variable setup for JAX/TF suppression
- Proper sys.path setup to cameo_cg root (not just data_prep/)

**Four steps**:

1. **HDF5 → Raw NPZ** (`h5_dataset_npz_transform.py`): Extracts frames from mdCATH HDF5 files. Supports temperature-group filtering and kcal/mol → eV conversion.

2. **All-atom → CG** (`cg_1bead.py`): Extracts Cα atoms, optionally applies `aggforce` constraint-aware optimal force projection. Supports per-AA species mapping (20 types) or 4-way charge-based grouping.

3. **Pad & Combine** (`pad_and_combine_datasets.py`): Pads all proteins to global N_max with zeros. Creates a unified species ID mapping across proteins. Generates validity masks (1=real, 0=padded). Can output one combined NPZ or separate padded NPZs (`--no_combine`).

4. **Prior Fitting** (`prior_fitting_script.py`): Boltzmann inversion on bond length, angle, and dihedral distributions. Fits harmonic bond params (r0, kr), Fourier angle coefficients (a, b), periodic dihedral params (k, γ). **NEW**: Supports spline prior fitting via KDE → Bayesian interpolation → cubic splines (optional `--spline` flag). Outputs YAML + diagnostic plots (+ optional spline NPZ).

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

- **DatasetLoader** (`loader.py`): Loads NPZ, shuffles with seed, applies `max_frames` limit, provides train/val split. Supports loading directories of NPZ files (auto-concatenation).

- **CoordinatePreprocessor** (`preprocessor.py`):
  - `compute_box_extent(R, mask, cutoff)` → `box = max_range + 2 * buffer_multiplier * cutoff`
  - `center_and_park(R, mask, box)` → centers real atoms at origin, parks padded atoms at `0.95 * box_extent` to keep them outside neighbor list range.

---

## 7. Evaluation

### 7.1 Quick Force Evaluation (`scripts/evaluate_forces.py`)

**NEW**: Supports **three evaluation modes**:

| Mode | Description | Use Case |
|------|-------------|----------|
| `full` (default) | Evaluate complete model (ML + priors if configured) | Standard evaluation |
| `prior-only` | Evaluate ONLY prior terms (parametric/spline/trained) | Measure prior contribution |
| `ml-only` | Evaluate ONLY ML model (force disable priors) | Measure ML contribution |

**Usage examples**:
```bash
# Full model evaluation
python scripts/evaluate_forces.py exported_models/model_params.pkl config.yaml

# Evaluate parametric priors only (from config YAML)
python scripts/evaluate_forces.py config_preprior.yaml --mode prior-only

# Evaluate spline priors only (from config with spline_file)
python scripts/evaluate_forces.py config_template.yaml --mode prior-only --frames 50

# Evaluate trained priors (from params.pkl['prior'])
python scripts/evaluate_forces.py exported_models/params.pkl config.yaml --mode prior-only

# ML-only evaluation (disable priors)
python scripts/evaluate_forces.py exported_models/params.pkl config.yaml --mode ml-only
```

**NEW**: Automatic checkpoint format detection:
- Supports exported model files (`model_params.pkl`)
- Supports training checkpoints (`epoch*.pkl`, `stage_*.pkl`)
- Auto-detects chemtrain trainer format vs direct params dict
- Extracts `trainer_state['params']` automatically

**Output files** (mode-specific naming):
- Full mode: `<model_id>_force_*.png`
- Prior-only: `<model_id>_prior_only_{prior_type}_force_*.png`
- ML-only: `<model_id>_ml_only_force_*.png`

Generated plots:
- `*_force_components.png` - Pred vs Ref scatter plots for X/Y/Z
- `*_force_distribution.png` - Force magnitude distributions
- `*_force_magnitude.png` - Magnitude comparison
- `*_force_metrics.txt` - Numerical RMSE/MAE values

### 7.2 Full Dataset Evaluation (`scripts/evaluate.py`)

Comprehensive evaluation over entire dataset:
- Per-frame force/energy metrics
- Energy component analysis
- Correlation plots
- Residual distributions

---

## 8. Multi-Node Distributed Training

Architecture: **1 process per node**, each process manages 4 local GPUs.

| Setting | Value |
|---------|-------|
| Processes per node | 1 (NOT 1 per GPU) |
| GPUs per process | 4 (`CUDA_VISIBLE_DEVICES=0,1,2,3`) |
| Gradient sync | chemtrain's `shard_map` + `lax.pmean` |
| Coordinator | FQDN of first SLURM node, port `29400 + (JOB_ID % 1000)` |

After `jax.distributed.initialize()`, `jax.devices()` returns all GPUs across all nodes. chemtrain's `Mesh(jax.devices(), ...)` spans them all.

**Enhanced error handling** (NEW):
- Try/except wrapper around `jax.distributed.initialize()`
- Rank-tagged error messages
- Hostname and environment diagnostics
- Unbuffered output via `flush=True`

```bash
# Single-node (4 GPUs):
sbatch scripts/run_training.sh config.yaml

# Multi-node (8 GPUs):
sbatch --nodes=2 scripts/run_training.sh config.yaml

# Resume from checkpoint:
sbatch scripts/run_training.sh config.yaml --resume auto
```

---

## 9. Export & Deployment

### 9.1 MLIR Export (`export/exporter.py`)

`AllegroExporter` extends chemtrain's `Exporter`:
- Converts the trained model to MLIR (StableHLO) via `chemtrain-deploy`.
- Handles LAMMPS 1-based species indexing (converts to 0-based internally).
- Includes prior topology (bonds, angles, dihedrals, repulsive pairs) in the exported function.
- Output: `.mlir` file compatible with LAMMPS via the `pair_style allegro` plugin.

### 9.2 LAMMPS Integration

The exported MLIR model is loaded into LAMMPS via:
```lammps
pair_style allegro
pair_coeff * * deployed_model.mlir H C N O
```

See `md_setup/inp_lammps_trained.in` for a complete LAMMPS input example.

---

## 10. Logging System

**Framework-wide logging** via `utils/logging.py`:

| Logger | Usage |
|--------|-------|
| `data_logger` | Data loading, preprocessing, dataset operations |
| `model_logger` | Model initialization, architecture diagnostics |
| `training_logger` | Training stages, checkpoints, convergence |
| `export_logger` | MLIR export, parameter saving |
| `eval_logger` | Evaluation metrics, diagnostics |
| `pipeline_logger` | **NEW**: Data preparation pipeline orchestration |

All loggers use consistent `[LoggerName] message` format.

**Training output organization**:
- SLURM output: `outputs/slurm-<JOB_ID>.out`
- Training log: `outputs/train_allegro_<JOB_ID>.log`
- Both created automatically by `run_training.sh`
- `.gitignore` prevents logs from being committed

---

## 11. Configuration System

### 11.1 ConfigManager (`config/manager.py`)

Typed accessor for nested YAML configuration:
- `get(*keys, default)` - Nested key traversal
- Convenience methods for common settings
- Validation of required sections
- Supports dynamic config modification (e.g., mode overrides in evaluation)

### 11.2 Key Configuration Sections

```yaml
model:
  use_priors: true/false      # Enable physics-based priors
  ml_model: "allegro"         # Options: "allegro", "mace", "painn"
  cutoff: 10.0                # Neighbor list cutoff (Angstrom)
  allegro_size: "default"     # Options: "default", "large", "med"

training:
  pretrain_prior: true/false  # LBFGS pretraining of prior weights
  train_priors: true/false    # Optimize prior params during training
  epochs_adabelief: 30        # Stage 1 epochs
  epochs_yogi: 50             # Stage 2 epochs (0 to skip)
  batch_per_device: 2         # Batch size per GPU

data:
  data_path: "path/to/dataset.npz"
  max_frames: null            # Limit frames (null = use all)
  val_fraction: 0.1           # Validation split

prior:
  # Parametric prior parameters (fitted or manual)
  bond: {r0: ..., kr: ...}
  # OR spline prior file
  spline_file: "fitted_priors_spline.npz"
```

---

## 12. Common Operations

Quick reference (see [COMMANDS.md](COMMANDS.md) for full details):

**Training**:
```bash
# Single-node
sbatch scripts/run_training.sh config.yaml

# Multi-node
sbatch --nodes=2 scripts/run_training.sh config.yaml

# Resume
sbatch scripts/run_training.sh config.yaml --resume auto
```

**Evaluation**:
```bash
# Quick force eval (10 random frames)
python scripts/evaluate_forces.py exported_models/params.pkl config.yaml

# Prior-only evaluation
python scripts/evaluate_forces.py config.yaml --mode prior-only --frames 50

# Full dataset evaluation
python scripts/evaluate.py config.yaml params.pkl --full
```

**Monitoring**:
```bash
# View SLURM output
tail -f outputs/slurm-<JOB_ID>.out

# View training log
tail -f outputs/train_allegro_<JOB_ID>.log

# Check job status
squeue -u $USER
```

---

## 13. Recent Improvements

### Data Preparation
- **run_pipeline.py refactoring**: Now uses framework's logging system, comprehensive type hints, enhanced docstrings
- **Spline prior support**: KDE → Bayesian interpolation → cubic splines for smooth PMFs

### Evaluation
- **Three evaluation modes**: full, prior-only, ml-only for comprehensive model analysis
- **Checkpoint format detection**: Automatic handling of trainer state extraction
- **Prior-only performance**: True computational skip (5-10x faster than full evaluation)

### Training Infrastructure
- **outputs/ directory**: Organized location for all SLURM and training logs
- **Enhanced JAX initialization**: Try/except error handling, diagnostic output
- **Unbuffered output**: `-u` flag and `flush=True` for real-time debugging
- **Rank-labeled output**: `-l` flag prepends task ID to each line

### Code Quality
- **Logging consistency**: All modules use framework loggers
- **Type hints**: Comprehensive coverage across data_prep
- **Documentation**: Enhanced docstrings with Args/Returns/Examples

---

## 14. Known Issues & Limitations

1. **Excluded volume gap**: Residues at sequence separation 2–5 have no repulsive interaction (topology code exists but is unused).

2. **LR schedule resume**: Learning rate schedule state is not saved in checkpoints — resumes restart from step 0.

3. **Multi-node sensitivity**: JAX distributed coordination can fail if tasks die before reaching barrier. Enhanced diagnostics now make failures visible.

4. **Prior parameter validation**: No automatic checks that fitted prior parameters are physically reasonable (e.g., negative kr).

---

## 15. External Dependencies

- **JAX** 0.4.34 (CUDA 12): Core autodiff and GPU execution
- **chemtrain**: Force matching trainer, multi-GPU parallelism
- **chemutils**: Allegro/MACE/PaiNN implementations
- **chemtrain-deploy**: MLIR export for LAMMPS
- **jax_md**: Neighbor lists, spatial utilities
- **optax**: Optimizers and learning rate schedules
- **LAMMPS** (2024-06 or later): Deployment target

See [EXTERNAL_CODE_CONTEXT.md](EXTERNAL_CODE_CONTEXT.md) for detailed integration notes.

---

**Last updated**: 2026-02-12
**Primary maintainer**: Documentation generated for Claude instances
