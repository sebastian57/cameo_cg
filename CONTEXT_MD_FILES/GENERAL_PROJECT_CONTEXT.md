# General Project Context

> **Purpose**: Concise reference for understanding this coarse-grained protein ML force field project.
> **Last Updated**: 2026-01-28

---

## 1. Project Goal

Train a machine learning force field for **1-bead-per-residue coarse-grained proteins** using the Allegro architecture + physics-based priors. The model learns to predict CG forces from CG positions, enabling fast MD simulations of protein dynamics.

---

## 2. Architecture Overview

```
All-atom MD → Force Projection (aggforce) → CG Dataset → Training → MLIR Export → LAMMPS MD
     ↓                    ↓                      ↓            ↓
 GROMACS/etc        F_CG = project(F_AA)    NPZ files    Allegro+Prior
```

### 2.1 Coarse-Graining Scheme
- **Mapping**: 1 bead per residue at Cα position
- **Force projection**: Uses `aggforce` library for constraint-aware optimal projection
- **Species**: 20 amino acid types (or 4-category grouping: positive/negative/polar/nonpolar)

### 2.2 Model Components

| Component | Purpose | Trainable |
|-----------|---------|-----------|
| **Allegro** | ML equivariant neural network | Yes |
| **Prior Energy** | Physics-based terms (bonds, angles, dihedrals, repulsion) | Partially (via LBFGS pretrain) |
| **Combined** | E_total = E_allegro + E_prior | Yes |

### 2.3 Prior Energy Terms

```
E_prior = w_bond × E_bond + w_angle × E_angle + w_dih × E_dihedral + w_rep × E_repulsive

E_bond = 0.5 × kr × Σ(r - r0)²           # Harmonic bonds (consecutive beads)
E_angle = Σ[a_n cos(nθ) + b_n sin(nθ)]   # Fourier series (triplets)
E_dihedral = Σ k_n(1 + cos(nφ - γ_n))    # Periodic torsion (quadruplets)
E_repulsive = ε × Σ(σ/r)^4               # Soft-sphere (separation ≥6)
```

**Known Gap**: No repulsion for residues with separation 2-5 (excluded volume). Code exists but is commented out in `prior_energy.py`.

---

## 3. Training Pipeline

### 3.1 Data Flow
```
config.yaml → DatasetLoader → CoordinatePreprocessor → CombinedModel → Trainer → Export
```

### 3.2 Training Stages
1. **Prior Pretrain** (optional): LBFGS optimization of prior params only
2. **Stage 1**: AdaBelief optimizer (fast initial learning)
3. **Stage 2**: Yogi optimizer (fine-tuning, optional)

### 3.3 Loss Function
```python
Loss = γ_F × MSE(F_pred, F_ref) + γ_U × MSE(E_pred, E_ref)
# Typically γ_F=1.0, γ_U=0.0 (force-matching only)
```

**chemtrain reports MSE, not RMSE**. To get RMSE: `sqrt(MSE)`.

### 3.4 Multi-Node Training

| Config | Value | Notes |
|--------|-------|-------|
| Processes per node | 1 | NOT 1 per GPU |
| GPUs per process | 4 | CUDA_VISIBLE_DEVICES=0,1,2,3 |
| Gradient sync | shard_map + lax.pmean | Across ALL GPUs in mesh |

```bash
# 2-node training (8 GPUs total)
sbatch --nodes=2 scripts/run_training.sh config.yaml
```

After `jax.distributed.initialize()`, `jax.devices()` returns ALL GPUs across nodes. chemtrain's `Mesh(jax.devices(), ...)` spans them all.

### 3.5 Multi-Node Implementation

**IMPORTANT: Manual Initialization Required**

JAX's automatic SLURM detection assumes **1 GPU per task**, which doesn't work for our "1 process per node with 4 GPUs" architecture. We must use **manual initialization** with `local_device_ids`:

```python
# In train.py - MANUAL initialization for multi-GPU per process
jax.distributed.initialize(
    coordinator_address=f"{coordinator_address}:{port}",
    num_processes=num_processes,
    process_id=process_id,
    local_device_ids=[0, 1, 2, 3],  # Explicitly use all 4 GPUs
)
```

**Why automatic detection fails:**
- JAX auto-detection uses `SLURM_LOCALID` → assumes 1 GPU per task
- With `--ntasks-per-node=1`, each task gets `SLURM_LOCALID=0` → only GPU 0
- Result: 2 total GPUs instead of 8, causing OOM errors

**Coordinator Address:**
- Must use FQDN (e.g., `jwb0262.juwels`) for cross-node resolution
- Port should be job-specific: `29400 + (SLURM_JOB_ID % 1000)`

**Troubleshooting:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Local GPUs: 1, Total GPUs: 2` | Auto-detection bug | Use manual init with `local_device_ids` |
| `DEADLINE_EXCEEDED` | Hostname not resolvable | Use FQDN (`.juwels` suffix) |
| OOM on single GPU | Not using all GPUs | Check `jax.device_count()` = 8 |

**Debugging Commands:**
```bash
# Print SLURM environment on each node
srun --nodes=2 --ntasks-per-node=1 bash -c 'echo "Host=$(hostname) PROCID=$SLURM_PROCID NTASKS=$SLURM_NTASKS"'
```

---

## 4. Key Files

| File | Purpose |
|------|---------|
| `scripts/train.py` | Main training entry point |
| `scripts/train_ensemble.py` | Ensemble training (multiple seeds) |
| `config_template.yaml` | All hyperparameters |
| `models/combined_model.py` | Allegro + Prior composition |
| `models/allegro_model.py` | Allegro wrapper |
| `models/prior_energy.py` | Physics-based energy terms |
| `models/topology.py` | Bond/angle/dihedral/repulsive pair indices |
| `training/trainer.py` | Training orchestration |
| `data_prep/cg_1bead.py` | All-atom → CG conversion |
| `data_prep/analyze_dataset.py` | Force diagnostics |

---

## 5. Force Magnitude Context

### 5.1 Typical Values (1-bead CG proteins)
- Mean |F|: 15-30 kcal/mol/Å
- Max |F|: 50-100 kcal/mol/Å
- RMS per component: 10-20 kcal/mol/Å

### 5.2 Interpreting Training Loss
```
MSE = mean((F_pred - F_ref)²)  # Per component, normalized
RMSE = sqrt(MSE)

For RMS_component ≈ 12 kcal/mol/Å:
- MSE ~3-6   → 15-20% relative error → Good
- MSE ~10-15 → 25-30% relative error → Fair
- MSE ~36    → 50% relative error    → Poor
```

### 5.3 Literature Comparison
| Model Type | Typical RMSE | Notes |
|------------|--------------|-------|
| All-atom ML (NequIP) | 0.5-2.0 | Full atomic resolution |
| CG 1-bead-per-residue | 5-15 | Irreducible error from CG |

---

## 6. Config Quick Reference

```yaml
# Key hyperparameters
model:
  cutoff: 12.0              # Neighbor list cutoff (Å)
  allegro_size: "default"   # Model size variant
  use_priors: true          # Enable physics priors

training:
  batch_per_device: 8       # Batch size per GPU
  epochs_adabelief: 100     # Stage 1 epochs
  epochs_yogi: 0            # Stage 2 epochs (0 = skip)
  val_fraction: 0.1         # Validation split

ensemble:
  enabled: false            # Multi-seed training
  n_models: 5               # Number of models
  save_all_models: false    # Save best only
```

---

## 7. Data Format

### 7.1 Input NPZ Structure
```python
{
    "R": float32[n_frames, n_atoms, 3],      # Positions (Å)
    "F": float32[n_frames, n_atoms, 3],      # Forces (kcal/mol/Å)
    "species": int32[n_frames, n_atoms],     # AA type IDs
    "mask": float32[n_frames, n_atoms],      # 1=real, 0=padded
    "Z": int32[n_atoms],                     # Atomic numbers (unused for CG)
}
```

### 7.2 Padding Strategy
- N_max = max atoms across all proteins
- Smaller proteins padded with zeros
- Padded atoms "parked" at 0.95 × box_extent
- Mask excludes padded atoms from loss

---

## 8. Export Format

- **MLIR**: StableHLO representation for LAMMPS ML-IAP
- **PKL**: Python pickle of trained parameters
- **Metadata JSON** (ensemble): Loss statistics, best model info

---

## 9. Known Issues & Gaps

1. **Excluded volume gap**: Residues at separation 2-5 have no repulsion (code exists but commented out)
2. **Force imbalance**: CG forces don't sum to zero (expected with aggforce projection)
3. **Multi-protein batching**: Different proteins in same batch must be padded to same size

### 9.1 Multi-Protein Padding & NaN Prevention (CRITICAL)

**Problem**: When training with multi-protein datasets (padded atoms), prior energy calculations can produce NaN losses.

**Root Causes** (TWO issues, both must be addressed):

1. **Forward pass NaN**: Multiplication-based masking propagates NaN
   - `0 × NaN = NaN` when masking angle/dihedral energy
   - Fixed in `prior_energy.py` using `jnp.where` for energy summation

2. **Backward pass NaN**: Undefined gradients for geometry computations
   - `d(norm)/dR = v/||v||` is undefined when `v = 0` (parked atoms at same location)
   - `d(atan2)/d(x,y)` is undefined when `x = y = 0` (dihedral of parked atoms)
   - These NaN propagate during gradient computation even when multiplied by 0 (`0 × NaN = NaN`)
   - **Fixed by applying `stop_gradient` to intermediate results (r, theta, phi) for invalid entries**

**Fix 1** (`models/combined_model.py`): Apply `stop_gradient` to padded atom coordinates

```python
mask_3d = mask[:, None]
R_masked = jnp.where(mask_3d > 0, R, jax.lax.stop_gradient(R))
E_prior = self.prior.compute_total_energy(R_masked, mask)
```

**Fix 2** (`models/prior_energy.py`): Block gradients for invalid geometry computations

The key insight: `jnp.where` in the energy sum doesn't prevent NaN gradients because the
gradient of the underlying computation (norm, atan2) is computed BEFORE being multiplied
by the mask. Solution: apply `stop_gradient` to intermediate values for invalid entries.

```python
# For bonds:
r = jnp.where(bond_valid, r, jax.lax.stop_gradient(r))
E_bond = jnp.sum(jnp.where(bond_valid, bond_energy, 0.0))

# For angles:
theta = jnp.where(angle_valid, theta, jax.lax.stop_gradient(theta))
E_angle = jnp.sum(jnp.where(angle_valid, U_angle, 0.0))

# For dihedrals (atan2(0,0) has undefined gradients):
phi = jnp.where(dih_valid, phi, jax.lax.stop_gradient(phi))
E_dih = jnp.sum(jnp.where(dih_valid, U_dih, 0.0))
```

**Symptoms of this bug**:
- `train_loss: nan` from epoch 0
- Only occurs with multi-protein datasets (single-protein works fine)
- Multi-node training NOT the cause (red herring)

**Testing**:
- Single-protein dataset (e.g., `4zohB01`): Should work regardless
- Multi-protein dataset (e.g., `2g4q4z5k`): Now works after fixes

---

## 10. Commands Cheat Sheet

```bash
# Analyze dataset
python data_prep/analyze_dataset.py --npz path/to/data.npz

# Single-node training (4 GPUs)
sbatch scripts/run_training.sh config_template.yaml

# Multi-node training (8 GPUs)
sbatch --nodes=2 scripts/run_training.sh config_template.yaml

# Ensemble training
# Set ensemble.enabled: true in config, then:
python scripts/train_ensemble.py config_template.yaml
```
