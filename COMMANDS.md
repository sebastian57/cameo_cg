# Clean Code Base - Command Reference

Quick reference for common operations using the clean code base.

---

## Training

### Single-Node Training (4 GPUs)

Submit from the `clean_code_base/` directory:

```bash
cd /p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/clean_code_base
sbatch scripts/run_training.sh config.yaml
```

### Multi-Node Training (8 GPUs across 2 nodes)

```bash
sbatch --nodes=2 scripts/run_training.sh config.yaml
```

### Resume Training from Checkpoint

```bash
# Auto-detect latest checkpoint
sbatch scripts/run_training.sh config.yaml --resume auto

# Resume from specific checkpoint
sbatch scripts/run_training.sh config.yaml --resume ./checkpoints_allegro/epoch30.pkl
```

### Change Training Time or Account

```bash
# Shorter job (2 hours)
sbatch --time=02:00:00 scripts/run_training.sh config.yaml

# Different account
sbatch --account=cameo scripts/run_training.sh config.yaml
```

---

## Plotting Loss Curves

Use `evaluation/visualizer.py` to plot training and validation loss from log files:

```bash
cd /p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/clean_code_base

# Basic usage
python evaluation/visualizer.py <log_file> [config.yaml] [output.png]

# Example with config (adds annotations to plot)
python evaluation/visualizer.py \
    train_allegro_13172105.log \
    config_excluded_vol.yaml \
    loss_excluded_vol.png

# Minimal (just log file, output auto-named)
python evaluation/visualizer.py train_allegro_13172105.log
```

Output:
- Plot saved to: specified path or `loss_curve_<log_stem>.png`
- Data saved to: `<output_path>.txt` (same name, .txt extension)

**Note:** Loss is also automatically plotted at the end of training by `train.py`.

---

## Evaluating Trained Models

### Quick Force Evaluation (with plots)

The `evaluate_forces.py` script supports three evaluation modes to compare different model components:
- **`full`** (default): Evaluate complete model (ML + priors if configured)
- **`prior-only`**: Evaluate ONLY prior terms (parametric, spline, or trained priors)
- **`ml-only`**: Evaluate ONLY ML model (disable priors)

#### Full Model Evaluation (ML + Priors)

```bash
cd /p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/clean_code_base

# Evaluate full model on 10 random frames (default)
python scripts/evaluate_forces.py \
    exported_models/model_params.pkl \
    config.yaml

# Evaluate from training checkpoint (supports chemtrain checkpoint format)
python scripts/evaluate_forces.py \
    checkpoints_allegro/epoch00040.pkl \
    config.yaml \
    --frames 50

# Custom output directory
python scripts/evaluate_forces.py \
    exported_models/model_params.pkl \
    config.yaml \
    --frames 20 \
    --output ./my_eval_results/
```

**Note**: The script automatically detects and supports:
- Exported model files (`model_params.pkl`)
- Training checkpoints (`epoch*.pkl`, `stage_*.pkl`)
- Both chemtrain trainer format and direct params dict format

#### Prior-Only Evaluation

Evaluate physics-based priors without ML component. Supports three prior types:
- **Parametric priors**: Histogram-fitted parameters from config YAML
- **Spline priors**: Cubic spline PMFs from NPZ file
- **Trained priors**: Optimized prior parameters from training

```bash
# Evaluate parametric priors (from config YAML)
python scripts/evaluate_forces.py \
    config_preprior.yaml \
    --mode prior-only \
    --frames 10

# Evaluate spline priors (from config with spline_file specified)
python scripts/evaluate_forces.py \
    config_template.yaml \
    --mode prior-only \
    --frames 50

# Evaluate trained priors (from params.pkl if train_priors was enabled)
python scripts/evaluate_forces.py \
    exported_models/model_params.pkl \
    config.yaml \
    --mode prior-only

# Evaluate trained priors from checkpoint
python scripts/evaluate_forces.py \
    checkpoints_allegro/stage_adabelief_epoch50.pkl \
    config.yaml \
    --mode prior-only
```

**Note**: Prior-only mode is much faster than full evaluation since it skips the expensive ML computation entirely.

#### ML-Only Evaluation

Evaluate only the ML model, disabling priors even if configured:

```bash
# Force disable priors for pure ML evaluation
python scripts/evaluate_forces.py \
    exported_models/model_params.pkl \
    config.yaml \
    --mode ml-only \
    --frames 50
```

#### Output Files

Saved to `./force_eval/` by default, with mode-specific naming:
- **Full mode**: `<model_id>_force_*.png`
- **Prior-only**: `<model_id>_prior_only_{prior_type}_force_*.png`
- **ML-only**: `<model_id>_ml_only_force_*.png`

Files generated:
- `*_force_components.png` - Pred vs Ref scatter plots for X/Y/Z
- `*_force_distribution.png` - Force magnitude distributions
- `*_force_magnitude.png` - Magnitude comparison
- `*_force_metrics.txt` - Numerical RMSE/MAE values

### Full Dataset Evaluation

```bash
# Evaluate single frame
python scripts/evaluate.py config.yaml params.pkl --frame 0

# Evaluate entire dataset
python scripts/evaluate.py config.yaml params.pkl --full

# Custom output directory
python scripts/evaluate.py config.yaml params.pkl --full --output-dir ./full_eval/
```

---

## Dataset Analysis

Analyze NPZ datasets before training:

```bash
cd /p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/clean_code_base

python data_prep/analyze_dataset.py \
    --npz data_prep/datasets/2g4q4z5k_320K_kcalmol_1bead_notnorm_aggforce.npz
```

Output includes:
- Array shapes and dtypes
- NaN/Inf checks
- Force magnitude statistics
- Training loss predictions

---

## Data Preparation

### Coarse-Graining All-Atom Data

Convert all-atom trajectories to 1-bead-per-residue CG representation:

```bash
cd /p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/clean_code_base

python data_prep/cg_1bead.py \
    --npz data_prep/raw_data/protein_allatom.npz \
    --pdb data_prep/raw_data/protein_topology.pdb \
    --output data_prep/datasets/protein_cg.npz
```

### Prior Fitting (Legacy + Spline)

Fit legacy parametric priors (histogram/Fourier path):

```bash
cd /p/project1/cameo/schmidt36/cameo_cg

python data_prep/prior_fitting_script.py \
    --data /path/to/combined_dataset.npz \
    --out_yaml data_prep/fitted_priors.yaml \
    --plots_dir data_prep/plots \
    --T 320.0
```

Fit spline priors as an add-on (also writes legacy YAML above):

```bash
cd /p/project1/cameo/schmidt36/cameo_cg

python data_prep/prior_fitting_script.py \
    --data /path/to/combined_dataset.npz \
    --out_yaml data_prep/fitted_priors.yaml \
    --plots_dir data_prep/plots \
    --T 320.0 \
    --spline \
    --spline_out data_prep/datasets/fitted_priors_spline.npz \
    --residue_specific_angles \
    --angle_min_samples 500 \
    --kde_bandwidth_factor 1.0 \
    --spline_grid_points 500
```

---

## Checking Job Status

```bash
# Check your running jobs
squeue -u $USER

# Check specific job details
scontrol show job <JOB_ID>

# Cancel a job
scancel <JOB_ID>

# View SLURM output in real-time
tail -f outputs/slurm-<JOB_ID>.out

# View training log in real-time
tail -f outputs/train_allegro_<JOB_ID>.log
```

**Note:** All SLURM outputs and training logs are now saved in the `outputs/` directory for better organization.

---

## Configuration Files

Available example configurations in `clean_code_base/`:

| Config File | Description |
|-------------|-------------|
| `config_allegro_only.yaml` | Pure Allegro model (no priors, `use_priors: false`) |
| `config_excluded_vol.yaml` | Allegro + priors, no prior pretraining (`pretrain_prior: false`) |
| `config_preprior.yaml` | Allegro + priors, with LBFGS prior pretraining (`pretrain_prior: true`) |

Key configuration options:

```yaml
model:
  use_priors: true/false      # Enable physics-based priors
  cutoff: 10.0                # Neighbor list cutoff (Angstrom)
  allegro_size: "default"     # Options: "default", "large", "med"

training:
  pretrain_prior: true/false  # LBFGS pretraining of prior weights
  epochs_adabelief: 30        # Stage 1 epochs
  epochs_yogi: 50             # Stage 2 epochs (0 to skip)
  batch_per_device: 2         # Batch size per GPU
```

---

## Useful Paths

```bash
# Clean code base
/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/clean_code_base

# Datasets
/p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/clean_code_base/data_prep/datasets/

# Training outputs (SLURM logs, training logs)
./outputs/

# Checkpoints (during training)
./checkpoints_allegro/

# Exported models (after training)
./exported_models/

# Environment activation
source /p/project1/cameo/schmidt36/load_modules.sh
source /p/project1/cameo/schmidt36/clean_booster_env/bin/activate
```

---

## Debugging

### Check GPU Allocation

```bash
# On compute node
nvidia-smi -L
echo $CUDA_VISIBLE_DEVICES
```

### Verify JAX Configuration

```python
import jax
print(f"Devices: {jax.devices()}")
print(f"Process count: {jax.process_count()}")
print(f"Process index: {jax.process_index()}")
```

### Common Issues

1. **QOSMaxWallDurationPerJobLimit**: Reduce `--time` (try 2-4 hours)
2. **Multi-node failures**: Check SLURM output for coordinator setup issues
3. **Memory errors**: Reduce `batch_per_device` or `max_frames`
