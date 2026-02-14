# cameo_cg

**Coarse-Grained Protein ML Force Field Framework**

Hybrid ML + physics force field for 1-bead-per-residue coarse-grained protein simulations, combining Allegro/MACE/PaiNN equivariant neural networks with physics-based priors.

## Quick Start

```bash
# Training
sbatch scripts/run_training.sh config.yaml

# Evaluation
python scripts/evaluate_forces.py exported_models/params.pkl config.yaml

# Monitoring
tail -f outputs/train_allegro_<JOB_ID>.log
```

## Features

- **Hybrid architecture**: ML (Allegro/MACE/PaiNN) + physics-based priors (bonds, angles, dihedrals, repulsion)
- **Force matching**: Direct prediction of CG forces from CG positions
- **Multi-node training**: JAX distributed + chemtrain for multi-GPU/multi-node parallelism
- **Three prior types**: Parametric (histogram-fitted), spline (KDE-based), or trained (optimized)
- **Three evaluation modes**: Full model, prior-only, or ML-only analysis
- **LAMMPS deployment**: MLIR export for production MD simulations

## Pipeline

```
H5 trajectories → NPZ extraction → CG mapping → Prior fitting → Training → MLIR export → LAMMPS
```

See [data_prep/run_pipeline.py](data_prep/run_pipeline.py) for the complete offline preprocessing pipeline.

## Documentation

- **[UPDATED_PROJECT_CONTEXT.md](UPDATED_PROJECT_CONTEXT.md)**: Comprehensive technical reference (primary documentation)
- **[COMMANDS.md](COMMANDS.md)**: Quick reference for common operations
- **[CONTEXT_MD_FILES/](CONTEXT_MD_FILES/)**: Additional context files (bug fixes, plans, scientific notes)

## Directory Structure

```
cameo_cg/
├── scripts/          # Training, evaluation, SLURM submission
├── models/           # CombinedModel, Allegro/MACE/PaiNN wrappers, priors
├── training/         # Multi-stage trainer, optimizer factory
├── data_prep/        # H5→NPZ→CG→priors pipeline
├── evaluation/       # Metrics, visualizations, scaling analysis
├── outputs/          # Training logs and SLURM outputs (auto-generated)
├── checkpoints_allegro/  # Training checkpoints
└── exported_models/  # MLIR + params.pkl
```

## Recent Updates (2026-02-12)

### New Features
- **outputs/ directory**: Organized logs (SLURM + training) with git-ignore
- **Prior-only evaluation**: Measure prior contribution independently
- **Checkpoint loading**: Auto-detection of trainer state format
- **Enhanced JAX init**: Try/except error handling + diagnostics
- **Pipeline refactoring**: Framework-consistent logging + type hints

### Improvements
- Unbuffered output (`-u`) + rank labeling (`-l`) for multi-node debugging
- Spline prior support (KDE → Bayesian interpolation → cubic splines)
- Prior-only mode skips ML computation (5-10x faster)
- Comprehensive type hints across data_prep
- Enhanced docstrings with Args/Returns/Examples

## Environment

Requires JUWELS Booster modules:
```bash
source /p/project1/cameo/schmidt36/load_modules.sh
source /p/project1/cameo/schmidt36/clean_booster_env/bin/activate
```

## Citation

Part of the CAMEO project. Based on modified chemtrain framework with Allegro/MACE/PaiNN from chemutils.

---

**Last updated**: 2026-02-12
