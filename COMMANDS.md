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

```bash
cd /p/project1/cameo/schmidt36/chemtrain-deploy/external/chemtrain/clean_code_base

# Evaluate on 10 random frames (default)
python scripts/evaluate_forces.py \
    exported_models/model_params.pkl \
    config.yaml

# Evaluate on 50 frames
python scripts/evaluate_forces.py \
    exported_models/model_params.pkl \
    config.yaml \
    --frames 50

# Custom output directory
python scripts/evaluate_forces.py \
    exported_models/model_params.pkl \
    config.yaml \
    --frames 20 \
    --output ./my_eval_results/
```

Output (saved to `./force_eval/` by default):
- `<model_id>_force_components.png` - Pred vs Ref scatter plots for X/Y/Z
- `<model_id>_force_distribution.png` - Force magnitude distributions
- `<model_id>_force_magnitude.png` - Magnitude comparison
- `<model_id>_force_metrics.txt` - Numerical RMSE/MAE values

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

---

## Checking Job Status

```bash
# Check your running jobs
squeue -u $USER

# Check specific job details
scontrol show job <JOB_ID>

# Cancel a job
scancel <JOB_ID>

# View job output in real-time
tail -f slurm-<JOB_ID>.out
```

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
