# Chemtrain Coarse-Grained Protein Force Field Pipeline - Cleanup Context

**Last Updated:** 2026-01-23
**Status:** Initial documentation - awaiting OOP architecture design

---

## Project Overview

This pipeline implements a coarse-grained (CG) machine learning force field for protein simulations. It combines:
- **Prior energy terms** (bonds, angles, dihedrals, repulsive interactions)
- **Allegro ML model** (equivariant graph neural network)

The workflow spans: data preparation → force matching training → model export → evaluation.

---

## Primary Goal

**Clean up and refactor the following files into a user-friendly, object-oriented codebase:**

1. Make code more readable and maintainable
2. Remove unused/dead code
3. Improve variable/function naming
4. Organize into clear OOP class structure
5. Simplify execution workflow
6. Separate concerns (data, models, training, evaluation)

---

## Files to Clean Up

### 1. **Core Energy Functions**

#### `allegro_energyfn_multiple_proteins.py`
**Location:** `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/`
**Lines:** 409 lines
**Purpose:** Defines the combined Allegro + Prior energy function for multiple protein systems

**Key Components:**
- `make_allegro_energy_fn(config)` - Main factory function
- `init_fn()` - Initializes Allegro model, neighbor lists, and prior topology
- `apply_fn()` - Computes total energy (Allegro + prior terms)
- `prior_fn_padded()` - Computes prior energy components:
  - Bond stretching (harmonic)
  - Angle bending (Fourier series)
  - Repulsive pairs (soft sphere, min_sep=6)
  - Dihedrals (periodic)
- `components_fn()` - Returns energy breakdown for analysis
- `compute_force_components()` - Returns force breakdown via autodiff

**Key Functions (Helper):**
- `filter_neighbors_by_mask()` - Masks neighbor list for padded systems
- `precompute_chain_topology()` - Generates bonds/angles for linear chain
- `precompute_repulsive_pairs()` - Generates non-bonded pairs
- `precompute_dihedrals()` - Generates dihedral indices
- `angular_fourier_energy()` - Fourier series for angle potential
- `dihedral_periodic_energy()` - Cosine series for dihedral potential

**Issues to Address:**
- Duplicate helper functions (also in `prior_energyfn.py`)
- Hardcoded energy scaling factors (0.5, 0.1, 0.25, 0.15) on line 304
- Mixed responsibilities (model setup, energy computation, topology)
- Inconsistent coordinate masking/parking logic
- Commented-out code (line 275)

---

#### `prior_energyfn.py`
**Location:** `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/`
**Lines:** 273 lines
**Purpose:** Prior energy function for pre-training (without Allegro)

**Key Components:**
- `make_prior_energy_fn(config)` - Factory for prior-only energy
- `init_fn()` - Topology setup (bonds, angles, repulsive, dihedrals)
- `prior_fn_padded_pretrain()` - Computes prior energy for pre-training
- `apply_fn_priors()` - Wrapper for force matching

**Issues to Address:**
- 90% overlap with `allegro_energyfn_multiple_proteins.py`
- Should inherit from shared base or be merged
- Sparse comments
- Pre-training code is commented out in main training script

---

### 2. **Training Scripts**

#### `train_fm_multiple_proteins.py`
**Location:** `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/`
**Lines:** 532 lines
**Purpose:** Main training script with multi-node JAX distributed support

**Key Components:**
- JAX distributed initialization (lines 22-49)
- Data loading and preprocessing
- Two-stage training:
  - **Stage 1:** AdaBelief optimizer (warmup + cosine decay)
  - **Stage 2:** Yogi optimizer (fine-tuning)
- Model export to MLIR for LAMMPS
- Loss plotting and checkpointing

**Key Functions:**
- `load_npz()` - Load dataset from NPZ
- `compute_extent_and_shift_masked()` - Compute simulation box size
- `shift_and_park_dataset()` - Center and park padded atoms
- `group_amino_acids_4way()` - Group amino acids into 4 species types
- `energy_fn_template()` - Template for chemtrain ForceMatching

**Issues to Address:**
- Massive monolithic `main()` function (~360 lines)
- Commented-out code blocks (lines 200-208, 252-263, 513-516)
- Hardcoded paths and magic numbers
- Data preprocessing mixed with training logic
- Patching external libraries at runtime (lines 189-208)
- Debug print statements scattered throughout
- Inconsistent commenting style

---

#### `compute_single_multi.py`
**Location:** `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/`
**Lines:** 303 lines
**Purpose:** Single-point energy/force evaluation and analysis

**Key Components:**
- Load trained model parameters
- Compute energy and forces for a single frame
- Force component analysis and plotting
- Error metrics (RMSE, MAE)

**Issues to Address:**
- Duplicate preprocessing functions from training script
- Analysis and plotting mixed with evaluation logic
- Should be separated into: evaluation + visualization modules

---

### 3. **Model Export**

#### `model_exporters.py`
**Location:** `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/`
**Lines:** 199 lines
**Purpose:** Export trained models to MLIR format for chemtrain-deploy/LAMMPS

**Key Components:**
- `AllegroExport` - Exporter for Allegro models
- `MACEExport` - Exporter for MACE models (unused in current pipeline)
- `PaiNNExport` - Exporter for PaiNN models (unused in current pipeline)
- `export_model()` - Main export function

**Issues to Address:**
- Clean and well-structured already
- MACE and PaiNN exporters unused - document or remove
- Could add more docstrings

---

### 4. **Analysis and Visualization**

#### `extract_and_plot_loss.py`
**Location:** `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/`
**Lines:** 96 lines
**Purpose:** Parse training logs and plot loss curves

**Key Components:**
- `plot_losses()` - Parse log file, extract train/val losses
- Creates annotated plots with hyperparameters
- Saves loss data to text file

**Issues to Address:**
- Hardcoded string slicing (line 78: `dataset[19:]`)
- Config access should be more robust
- Could be integrated into a training callbacks system

---

### 5. **Configuration**

#### `config_allegro_exp2.yaml`
**Location:** `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/`
**Lines:** 139 lines
**Purpose:** Main training configuration file (will be renamed in clean version)

**Key Sections:**
- **Metadata:** seed, model context, protein name, model ID
- **Data:** path to NPZ dataset, max frames
- **System:** simulation box dimensions
- **Model:**
  - Multiple Allegro configurations (default, large, med) - allows switching
  - Cutoff and neighbor list parameters
  - **Priors:** Pre-fitted parameters for bonds (r0, kr), angles (a, b), repulsive (epsilon, sigma), dihedrals (k_dih, gamma_dih)
- **Optimizer:** Multiple optimizer configs (adam, yogi, adabelief, lion, polyak_sgd, fromage)
- **Training:** epochs per stage, validation split, batch sizes, force matching weights (gammas), paths

**Issues to Address:**
- Very clean structure already
- Multiple Allegro configs provided but only one used (good for flexibility)
- Commented-out data path (line 9)
- This structure should be preserved and potentially enhanced with validation

---

### 6. **Execution Scripts**

#### `run_training.sh`
**Location:** `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/`
**Lines:** 41 lines
**Purpose:** SLURM submission script for single-node training (4 GPUs)

**Key Components:**
- Environment setup (modules, conda, CUDA paths)
- Single node, 4 GPU configuration
- Launches `train_fm_multi_nopriors.py` (note: different script!)

**Issues to Address:**
- Points to wrong training script (`train_fm_multi_nopriors.py` instead of `train_fm_multiple_proteins.py`)
- Hardcoded absolute paths
- Could be templated or made more portable

---

#### `run_training_multi_node.sh`
**Location:** `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/`
**Lines:** 40 lines
**Purpose:** SLURM submission script for multi-node training (2 nodes, 8 GPUs)

**Key Components:**
- JAX distributed coordinator setup
- Multi-node configuration
- Launches `train_fm_multiple_proteins.py`

**Issues to Address:**
- Similar to single-node script - should share common setup
- Hardcoded paths
- Could extract environment setup to shared script

---

### 6. **Data Preparation** (Secondary Focus)

#### `data_prep/cg_1bead.py`
**Location:** `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/data_prep/`
**Lines:** 302 lines
**Purpose:** Coarse-grain atomistic simulations to 1-bead-per-residue (CA atoms)

**Key Components:**
- `generate_optim_forcematch()` - Optimal force mapping using aggforce library
- `extract_ca_indices()` - Extract CA atom indices from PDB
- `build_cg_dataset()` - Main CG workflow
- `per_type_force_normalization()` - Normalize forces by species type
- `group_amino_acids_4way()` - Group amino acids into 4 categories

**Issues to Address:**
- Commented-out normalization code
- Some debug prints
- Could separate force mapping, extraction, and normalization

---

#### `data_prep/pad_and_combine_datasets.py`
**Location:** `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/data_prep/`
**Lines:** 163 lines
**Purpose:** Combine multiple protein datasets with padding

**Key Components:**
- `combine_and_pad_npz()` - Merge datasets with different N_atoms
- Creates global AA→ID mapping across proteins
- Generates masks for valid atoms

**Issues to Address:**
- Buggy parking code (lines 121-125) - references undefined variables
- Hardcoded paths in `main()`
- Should be importable, not just a script

---

#### `data_prep/dump_to_npz.py`
**Location:** `/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/data_prep/`
**Lines:** 47 lines
**Purpose:** Convert LAMMPS dump files to NPZ format

**Key Components:**
- `dump_to_npz()` - Parse LAMMPS dump and save as NPZ

**Issues to Address:**
- Minimal - mostly clean
- Could add error handling

---

## Identified Code Duplication

### Functions Appearing in Multiple Files:

1. **Topology Generation:**
   - `precompute_chain_topology()` - in both energy files
   - `precompute_repulsive_pairs()` - in both energy files
   - `precompute_dihedrals()` - in both energy files

2. **Prior Energy Terms:**
   - `angular_fourier_energy()` - in both energy files
   - `dihedral_periodic_energy()` - in both energy files
   - `dihedral_angles_single_frame()` - in both energy files

3. **Data Processing:**
   - `group_amino_acids_4way()` - in training script and cg_1bead.py
   - `compute_extent_and_shift_masked()` - in training and evaluation scripts
   - `shift_and_park_dataset()` - similar logic in multiple places
   - `load_npz()` - appears in 3+ files with variations

---

## Proposed Object-Oriented Architecture

### Module Structure (to be refined with user)

```
clean_code_base/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── config_loader.py          # YAML config handling
├── data/
│   ├── __init__.py
│   ├── loader.py                 # Dataset loading utilities
│   ├── preprocessor.py           # Coordinate preprocessing, padding
│   └── coarse_graining.py        # CG methods (1-bead, force mapping)
├── models/
│   ├── __init__.py
│   ├── base_energy.py            # Abstract base class for energy functions
│   ├── prior_energy.py           # Prior energy terms (bonds, angles, etc.)
│   ├── allegro_energy.py         # Allegro ML model wrapper
│   ├── combined_energy.py        # Prior + ML combined model
│   └── topology.py               # Topology generation utilities
├── training/
│   ├── __init__.py
│   ├── trainer.py                # Training loop orchestration
│   ├── optimizers.py             # Optimizer configurations
│   └── callbacks.py              # Checkpointing, logging, plotting
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py              # Single-point evaluation
│   └── visualizer.py             # Plotting and analysis
├── export/
│   ├── __init__.py
│   └── exporter.py               # Model export to MLIR (refactored)
└── scripts/
    ├── train.py                  # Simplified training script
    ├── evaluate.py               # Evaluation script
    └── prepare_data.py           # Data preparation pipeline
```

### Key Classes (Initial Ideas):

1. **DataLoader** - Handle NPZ loading, species mapping, masking
2. **CoordinatePreprocessor** - Centering, parking, box computation
3. **TopologyBuilder** - Generate bonds, angles, dihedrals, repulsive pairs
4. **PriorEnergy** - Compute bond/angle/dihedral/repulsive energies
5. **AllegroModel** - Wrap Allegro initialization and inference
6. **CombinedModel** - Compose Prior + Allegro with weights
7. **Trainer** - Orchestrate training loop, checkpointing, multi-GPU
8. **Evaluator** - Single-point evaluation and force analysis
9. **ModelExporter** - Export to MLIR for LAMMPS
10. **ConfigManager** - Handle YAML configs with validation

---

## Cleanup Priorities

### Phase 1: Core Energy Functions (HIGH)
- [ ] Extract shared topology functions to `topology.py`
- [ ] Extract shared prior energy terms to `prior_energy.py`
- [ ] Refactor `allegro_energyfn_multiple_proteins.py` into classes
- [ ] Remove duplicate code between prior and allegro energy files

### Phase 2: Data Pipeline (MEDIUM)
- [ ] Create unified `DataLoader` class
- [ ] Fix bugs in `pad_and_combine_datasets.py`
- [ ] Consolidate preprocessing functions
- [ ] Create clean data preparation workflow

### Phase 3: Training (HIGH)
- [ ] Break up monolithic `train_fm_multiple_proteins.py`
- [ ] Create `Trainer` class with methods for each stage
- [ ] Extract optimizer configs to separate module
- [ ] Remove commented-out code and patches
- [ ] Add proper logging instead of print statements

### Phase 4: Evaluation & Export (MEDIUM)
- [ ] Separate evaluation from plotting in `compute_single_multi.py`
- [ ] Integrate loss plotting into training callbacks
- [ ] Clean up model exporters (document/remove unused ones)

### Phase 5: Scripts & Documentation (LOW)
- [ ] Create unified execution scripts
- [ ] Add comprehensive docstrings
- [ ] Create user guide and examples

---

## Architecture Decisions (User Confirmed)

1. **OOP Architecture:** ✓ Full OOP with classes - Trainer, Model, DataLoader, Evaluator classes with clean separation
2. **Prior Pre-training:** ✓ Keep and restore as an optional first stage before Allegro training
3. **Multi-model Support:** ✓ Allegro only - remove MACE and PaiNN exporters from cleaned code
4. **Execution:** ✓ Unified training script that auto-detects single vs multi-node from environment
5. **Config Format:** ✓ Keep YAML structure (config_allegro_exp2.yaml as template) with improved validation
6. **Module Structure:** Follow the proposed structure with data/, models/, training/, evaluation/, export/

---

## Testing Constraints

**Environment:** Currently on HPC login node (CPU only)
- **Cannot test:** Full training runs, GPU operations, multi-node distributed training
- **Can test:** Module imports, class instantiation, small single-structure tests
- **Workflow:** Implement changes → user submits SLURM jobs → collect results → iterate

**Testing Strategy:**
1. Develop and test locally with minimal data (1-2 structures)
2. Alert user when larger testing is needed
3. User submits batch jobs for real validation
4. Review results and iterate

---

## Progress Tracking

**Status:** Phase 3 COMPLETE ✓ - Core Pipeline Ready (Issues Documented)

### Phase 1: Foundation - Shared Utilities ✓

**Completed Modules:**
- [x] `config/manager.py` - ConfigManager class with YAML loading and convenient accessors
- [x] `models/topology.py` - TopologyBuilder and topology generation functions
- [x] `data/loader.py` - DatasetLoader class with NPZ loading and species mapping
- [x] `data/preprocessor.py` - CoordinatePreprocessor for centering and parking

**Files Created:** 9 files (4 modules + 4 __init__.py + 1 test script)

### Phase 2: Energy Models ✓

**Completed Modules:**
- [x] `models/prior_energy.py` - PriorEnergy class with bonds, angles, dihedrals, repulsive terms
- [x] `models/allegro_model.py` - AllegroModel wrapper for ML force field
- [x] `models/combined_model.py` - CombinedModel that composes Prior + Allegro

**Key Features Implemented:**
- ✓ Configurable prior on/off via `model.use_priors` in YAML
- ✓ Configurable energy term weights via `model.priors.weights` in YAML
- ✓ Three operational modes: Pure Allegro, Pure Prior, Combined
- ✓ Energy component breakdown for analysis
- ✓ Force component computation via autodiff
- ✓ Clean separation of concerns (topology, prior, ML, combined)

**Files Created:** 3 new modules + updated __init__.py

**Testing Status:**
- Module structure verified ✓
- Functional testing requires proper environment with JAX/chemutils
- Ready for integration testing with Phase 1

### Phase 3: Training Infrastructure ✓

**Completed Modules:**
- [x] `training/optimizers.py` - Optimizer factory supporting 6 optimizer types
- [x] `training/trainer.py` - Full training orchestration with prior pre-training

**Key Features Implemented:**
- ✓ Multi-stage training (e.g., AdaBelief → Yogi)
- ✓ Optional prior pre-training via `training.pretrain_prior` boolean
- ✓ Full pipeline automation via `train_full_pipeline()`
- ✓ Checkpoint saving and loading
- ✓ Model parameter export
- ✓ Frame evaluation and analysis
- ✓ NumpyDataLoader compatibility patch
- ✓ Multi-GPU support via batch_per_device

**Files Created:** 2 new modules + updated __init__.py

**Supported Optimizers:**
1. AdaBelief (default stage 1)
2. Yogi (default stage 2)
3. Adam
4. Lion
5. Polyak SGD
6. Fromage

**Configuration Added:**
```yaml
training:
  pretrain_prior: false  # NEW: Enable prior pre-training
  pretrain_prior_epochs: 50
  pretrain_prior_optimizer: "adam"
  stage1_optimizer: "adabelief"  # NEW: Configurable stage optimizers
  stage2_optimizer: "yogi"
  epochs_adabelief: 100
  epochs_yogi: 50
```

**Next:** Fix documented issues → Phase 4 (Evaluation & Export)

**Next Steps:**
1. Discuss and finalize OOP architecture with user
2. Start with Phase 1 (Core Energy Functions)
3. Update this document after each major milestone

---

## Technical Notes

### Key Dependencies:
- JAX / JAX-MD (core computation, autodiff)
- chemtrain (force matching trainer)
- chemutils / allegro (Allegro model implementation)
- aggforce (optimal force mapping for CG)
- MDTraj (PDB/trajectory handling)

### Hardware Requirements:
- Multi-GPU support (4-8 GPUs typical)
- Multi-node JAX distributed (optional)
- CUDA 12 on JUWELS Booster HPC system

### Energy Terms in Prior:
1. **Bonds:** Harmonic stretching between consecutive beads
2. **Angles:** Fourier series in theta (bending)
3. **Dihedrals:** Periodic cosine series in phi (torsion)
4. **Repulsive:** Soft-sphere between beads separated by ≥6 residues

### Training Strategy:
1. AdaBelief optimizer (warmup + cosine decay)
2. Yogi optimizer (fine-tuning)
3. Force matching objective (L2 on forces)
4. Validation split monitoring
5. Checkpointing and early stopping

---

## Code Review: Issues and Inconsistencies

**Last Review:** 2026-01-23 (Phase 3 completion)

### Critical Issues (Must Fix)

#### ISSUE 1: Prior Energy Weights Structure Mismatch
**Location:** `models/prior_energy.py:182-188`

**Problem:**
Code assumes YAML structure:
```yaml
model:
  priors:
    weights:
      bond: 0.5
      angle: 0.1
      repulsive: 0.25
      dihedral: 0.15
```

But actual YAML (config_allegro_exp2.yaml) has:
```yaml
model:
  priors:
    r0: 3.8375435
    kr: 154.50629
    # ... (no weights subsection)
```

**Impact:** PriorEnergy initialization will fail or use wrong defaults

**Fix Required:**
- Option A: Add `weights` section to config_allegro_exp2.yaml template
- Option B: Make weights optional with hardcoded defaults (0.5, 0.1, 0.25, 0.15)
- Option C: Add separate config keys: `bond_weight`, `angle_weight`, etc. at priors level

**Recommendation:** Option B (backwards compatible) + document Option A for new configs

---

#### ISSUE 2: Missing Config Methods
**Location:** `config/manager.py`

**Problem:**
PriorEnergy and Trainer expect these config methods that don't exist:
- `config.get("model", "use_priors")` - Not implemented as method
- `config.get("model", "allegro_size")` - Not implemented as method
- `config.get("training", "pretrain_prior")` - Not implemented as method
- `config.get("training", "stage1_optimizer")` - Not implemented as method
- `config.get("training", "stage2_optimizer")` - Not implemented as method

**Impact:** Code will work with generic `get()` but lacks validation and documentation

**Fix Required:**
Add convenience methods to ConfigManager:
```python
def use_priors(self) -> bool:
    return self.get("model", "use_priors", default=True)

def get_allegro_size(self) -> str:
    return self.get("model", "allegro_size", default="default")

def pretrain_prior_enabled(self) -> bool:
    return self.get("training", "pretrain_prior", default=False)
```

---

#### ISSUE 3: Unused theta0 and k_theta Parameters
**Location:** `models/prior_energy.py:157-177`, config_allegro_exp2.yaml:58-59

**Problem:**
Original config has `theta0` and `k_theta` for harmonic angle potential:
```yaml
theta0: 1.8335507
k_theta: 8.271271714604836
```

But PriorEnergy uses Fourier series (`a`, `b`) instead, ignoring these parameters.

Original code has commented-out harmonic angle energy:
```python
# E_angle = 0.5 * k_theta * jnp.sum(angle_valid * (theta - theta0) ** 2)
```

**Impact:** Config parameters are loaded but never used. Misleading documentation.

**Fix Required:**
- Option A: Remove unused parameters from config and loader
- Option B: Support both Fourier and harmonic angle potentials (config switch)
- Option C: Document that Fourier supersedes harmonic

**Recommendation:** Option C + eventually Option A

---

### Moderate Issues (Should Fix)

#### ISSUE 4: Missing YAML Config Keys for New Features
**Location:** Multiple files

**Problem:**
New features assume YAML keys that don't exist in template:
- `model.use_priors` (for enabling/disabling priors)
- `model.allegro_size` (for selecting allegro variant)
- `training.pretrain_prior` (for prior pre-training)
- `training.pretrain_prior_epochs`
- `training.pretrain_prior_optimizer`
- `training.stage1_optimizer`
- `training.stage2_optimizer`

**Impact:** Features work with defaults but aren't configurable

**Fix Required:**
Create updated config template `config_template_clean.yaml` with all new options documented

---

#### ISSUE 5: Hardcoded Allegro Size Selection
**Location:** `models/allegro_model.py:47-49`

**Problem:**
```python
allegro_size = config.get("model", "allegro_size", default="default")
self.allegro_config = config.get_allegro_config(size=allegro_size)
```

But config has three variants (`allegro`, `allegro_large`, `allegro_med`) and no `allegro_size` key.

**Impact:** Will always use "default" → "allegro" config. Cannot select other sizes.

**Fix Required:**
Add to YAML:
```yaml
model:
  allegro_size: "default"  # or "large" or "med"
```

---

#### ISSUE 6: NumpyDataLoader Patch Applied Globally
**Location:** `training/trainer.py:61-76`

**Problem:**
The dataloader patch modifies the class globally:
```python
_NDL._get_indices = _patched_get_indices
```

This affects all instances, not just trainer instances. Could cause issues in multi-trainer scenarios.

**Impact:** Low (unlikely to have multiple trainers), but not clean

**Fix Required:**
Consider applying patch only once at module level or using a context manager

---

### Minor Issues (Nice to Have)

#### ISSUE 7: Inconsistent Print Statements
**Location:** Multiple files

**Problem:**
Mix of `print()` statements for logging:
- Some prefixed with `[ModuleName]`
- Some without prefix
- No log levels (info, debug, warning)

**Impact:** Hard to filter or redirect logs

**Fix Required:**
Add proper logging with Python's `logging` module:
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Message")
```

---

#### ISSUE 8: Species Mapping Inconsistency
**Location:** `data/loader.py`, `models/allegro_model.py`

**Problem:**
Species handling varies:
- DatasetLoader: species are amino acid IDs (0 to N_species-1)
- AllegroModel: expects species 0-indexed but does `species - 1` in some places
- Confusion between "species ID" and "amino acid ID"

**Current code** in allegro_model.py:
```python
species_model = species - 1  # Line 42 in original
```

But this subtraction doesn't exist in the clean code - good!

**Impact:** Minimal - needs verification with real data

**Fix Required:**
Document expected species format and ensure consistency

---

#### ISSUE 9: Missing Type Hints in Some Functions
**Location:** `models/prior_energy.py` helper functions

**Problem:**
Helper functions like `_angular_fourier_energy` have partial type hints.

**Impact:** Reduced IDE support and documentation

**Fix Required:**
Add complete type hints to all functions

---

#### ISSUE 10: Box Parameter Handling
**Location:** Multiple files

**Problem:**
Box parameter is used inconsistently:
- Sometimes passed from config: `config.get("system", "box")`
- Sometimes computed dynamically: `preprocessor.compute_box_extent()`
- Original code has hardcoded box: `box: [406.13, 450.15002, 452.7]`

Question: Should we use config box or computed box?

**Impact:** Might use wrong box size

**Fix Required:**
Clarify box size strategy:
- Option A: Always compute from data (ignore config box)
- Option B: Use config box if provided, else compute
- Option C: Require config box (fail if missing)

**Current behavior:** AllegroModel uses config box, preprocessor computes its own

---

### Documentation Gaps

#### GAP 1: Missing Example Config
Need to create `config_template_clean.yaml` showing all new options

#### GAP 2: Missing Usage Examples
Need to add example scripts showing:
- How to train with priors
- How to train without priors
- How to do prior pre-training
- How to use different Allegro sizes

#### GAP 3: No Migration Guide
Need to document how to convert old configs to new format

---

### Testing Gaps

#### TEST 1: No Unit Tests
Need tests for:
- Topology generation (verify bond/angle/dihedral indices)
- Prior energy computation (compare with analytical values)
- Config loading (verify defaults and validation)

#### TEST 2: No Integration Tests
Need to verify:
- Full training pipeline runs without errors
- Output matches original implementation
- Multi-GPU training works
- Prior pre-training works

---

**END OF CONTEXT DOCUMENT**
