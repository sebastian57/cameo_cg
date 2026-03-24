# cameo_cg — Command Reference

Quick reference for common operations.

---

## Repository Layout

```
cameo_cg/
├── config/            # ConfigManager + type definitions
├── configs/           # base_config.yaml (all options documented)
├── data/              # DatasetLoader, CoordinatePreprocessor
├── data_prep/         # CG mapping, prior fitting, dataset tools
├── models/            # ML model wrappers, PriorEnergy, topology
├── training/          # Trainer, optimizers, prior residual
├── export/            # ModelExporter (MLIR for LAMMPS)
├── analysis_tests/    # All evaluation, analysis, and test scripts
├── scripts/           # train.py, SLURM wrappers, shell scripts
├── utils/             # Shared JAX setup utilities
├── new_conf.yaml      # Working example config
└── COMMANDS.md        # This file
```

---

## Training

### Single-Node Training (4 GPUs)

```bash
sbatch scripts/run_training.sh new_conf.yaml
```

### Multi-Node Training

```bash
sbatch --nodes=2 scripts/run_training.sh new_conf.yaml
```

### Resume from Checkpoint

```bash
sbatch scripts/run_training.sh new_conf.yaml --resume auto
sbatch scripts/run_training.sh new_conf.yaml --resume ./checkpoints/epoch30.pkl
```

### Suite of Experiments (Array Jobs)

Place multiple config files in a directory, then:

```bash
bash scripts/submit_suite.sh --input_dir ./my_configs/ --name my_experiment
```

This creates a SLURM array job — one task per config file. Results are
written to each config's `paths.output_dir`.

### Run Analysis Over a Suite

```bash
sbatch scripts/run_analysis.sh /path/to/suite/output/
```

Or directly:

```bash
python analysis_tests/analyze_suite.py /path/to/suite/output/ --detailed-force-eval
```

---

## Evaluation

All evaluation scripts live in `analysis_tests/`.

### Force Evaluation

Three modes: `full` (ML + priors), `prior-only`, `ml-only`.

```bash
# Full model evaluation
python analysis_tests/evaluate_forces.py \
    exported_models/model_params.pkl new_conf.yaml --frames 50

# Prior-only
python analysis_tests/evaluate_forces.py \
    new_conf.yaml --mode prior-only --frames 10

# ML-only
python analysis_tests/evaluate_forces.py \
    exported_models/model_params.pkl new_conf.yaml --mode ml-only
```

### Full Dataset Evaluation

```bash
python analysis_tests/evaluate.py new_conf.yaml params.pkl --full
```

### Plotting Loss Curves

```bash
python analysis_tests/visualizer.py train.log new_conf.yaml loss_plot.png
```

Loss curves are also auto-plotted at the end of training.

---

## Data Preparation

### Coarse-Graining

```bash
python data_prep/cg_1bead.py \
    --npz raw_data/protein_allatom.npz \
    --pdb raw_data/protein_topology.pdb \
    --output data_prep/datasets/protein_cg.npz
```

### Prior Fitting

```bash
# Parametric + spline priors
python data_prep/prior_fitting_script.py \
    --data /path/to/dataset.npz \
    --out_yaml data_prep/fitted_priors.yaml \
    --plots_dir data_prep/plots \
    --T 320.0 \
    --spline \
    --spline_out data_prep/datasets/fitted_priors_spline.npz
```

### Dataset Analysis

```bash
python data_prep/analyze_dataset.py --npz data_prep/datasets/my_data.npz
```

---

## Configuration

Reference config with all options documented: `configs/base_config.yaml`

Working example: `new_conf.yaml`

### Key Sections

```yaml
debug:
  neighbor_logging: false
  shape_trace: false
  model_logging: false

paths:
  output_dir: ./outputs/my_run
  checkpoint_dir: null       # null -> {output_dir}/checkpoints
  export_dir: null           # null -> {output_dir}/exports

data:
  path: data_prep/datasets/my_data.npz
  batch_mode: standard       # standard | tiled

model:
  ml_model: allegro          # allegro | allegro_cueq | allegro_cueq_fast | mace | painn
  use_priors: true
  cutoff: 10.0
  allegro:                   # direct hyperparameters (no size indirection)
    num_types: 22
    max_ell: 2
    num_layers: 3
    ...

training:
  stages:                    # ordered list — add more stages freely
    - optimizer: adabelief
      epochs: 80
    - optimizer: lamb
      epochs: 0
  batch_per_device: 2
  compute_dtype: float32     # float32 | bfloat16
```

---

## SLURM Job Management

```bash
squeue -u $USER                         # list jobs
scontrol show job <JOB_ID>              # details
scancel <JOB_ID>                        # cancel
tail -f outputs/slurm-<JOB_ID>.out      # live output
```

---

## Debugging

### Debug Config Section

All debug flags default to off. Override via YAML or environment variables:

```yaml
debug:
  neighbor_logging: true     # or: CHEMTRAIN_DEBUG_NEIGHBOR=1
  shape_trace: true          # or: CHEMTRAIN_DEBUG_SHAPE_TRACE=1
  model_logging: true        # jax.debug.print in compiled blocks
```

### GPU / JAX Checks

```bash
nvidia-smi -L
echo $CUDA_VISIBLE_DEVICES
```

```python
import jax
print(jax.devices(), jax.process_count(), jax.process_index())
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `JAX_ENABLE_X64` | `0` | Enable 64-bit precision |
| `JAX_INIT_TIMEOUT` | `1800` | Distributed init timeout (seconds) |
| `CHEMTRAIN_DEBUG_NEIGHBOR` | `0` | Neighbor list debug logging |
| `CHEMTRAIN_DEBUG_SHAPE_TRACE` | `0` | One-shot shape trace |

### Common Issues

1. **QOSMaxWallDurationPerJobLimit**: Reduce `--time`
2. **Multi-node failures**: Check coordinator setup in SLURM output
3. **Memory errors**: Reduce `batch_per_device` or `max_frames`
