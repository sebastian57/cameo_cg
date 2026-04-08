# cameo_cg_pkgflow

`cameo_cg_pkgflow` contains the training, prior-energy, export, and analysis code used for the CAMEO coarse-grained protein force-field workflow. It is the main working repository for training Allegro, Allegro-cuEq, MACE, and PaiNN-based coarse-grained models and exporting them for downstream MD use.

## Recommended Workflow

Use the repository root for shared code, scripts, and documentation.
Use `local_work/` for everything experiment-specific.

That means:
- run shell launchers from the repository root
- keep training configs in `local_work/<experiment>/`
- let training outputs be created under `local_work/outputs/`
- keep temporary notes, copied checkpoints, debug artifacts, and one-off analysis products in `local_work/`
- only create files outside `local_work/` if they are intended to be shared and potentially committed

`local_work/` is ignored by git by default.

## Quick Start

Create a local workspace, copy the shared base config into it, edit that config there, and submit from the repository root.

```bash
mkdir -p local_work/example_run
cp configs/base_config.yaml local_work/example_run/example_config.yaml

sbatch ./scripts/run_training.sh local_work/example_run/example_config.yaml
```

For a single run, outputs are written to:

```text
local_work/outputs/YYYYMMDD_example_config/
```

That run directory contains the copied input config, the resolved runtime config, training logs, checkpoints, exports, profiles, and the SLURM log for the run.

## Important Path Behavior

The launchers are designed so that:
- the submitted config can live in `local_work/`
- single-run outputs default to `local_work/outputs/YYYYMMDD_<config_name>/`
- set `paths.output_dir` in a config to force an explicit run directory
- suite outputs are written under `local_work/outputs/` by default
- relative dataset and spline-prior paths can be resolved relative to either the config directory or the repository root
- launchers should still be invoked from the repository root, for example `sbatch ./scripts/run_training.sh ...`

## Environment Setup

Environment and deployment setup is documented in:
- `env_setup/SETUP_ENV.md`
- `env_setup/interactive_job.md`
- `env_setup/LAMMPS_build.md`
- `env_setup/CONNECTOR_REBUILD.md`

### chemtrain Layout

The expected local layout is:
- `chemtrain-deploy/` cloned from the upstream online source
- `chemtrain_cameo/` cloned from your own source
- `chemtrain_cameo/` placed at `chemtrain-deploy/external/chemtrain/chemtrain_cameo`

This repository expects the active Python environment to import `chemtrain` from that local editable `chemtrain_cameo` checkout.

### Environment Variables For Training

The launchers no longer fall back to hard-coded old venv paths. Before training, set the environment variables explicitly:
- `CAMEO_CG_PROJECT_ROOT`: repository root for this checkout
- `CAMEO_ACTIVE_VENV`: optional explicit override for any model type
- `CAMEO_CUEQ_VENV`: required for `allegro_cueq*` models when `CAMEO_ACTIVE_VENV` is not set
- `CAMEO_STANDARD_VENV`: required for non-cueq models when `CAMEO_ACTIVE_VENV` is not set
- `CAMEO_LAMMPS_BUILD_DIR`: optional override for a local LAMMPS build location

A typical setup looks like:

```bash
export CAMEO_CG_PROJECT_ROOT=/path/to/cameo_cg_pkgflow
export CAMEO_CUEQ_VENV=/path/to/your/cueq_venv
export CAMEO_STANDARD_VENV=/path/to/your/standard_venv
```

Or force one specific environment for a run:

```bash
export CAMEO_ACTIVE_VENV=/path/to/your/venv
```

If the required variable is missing, `scripts/slurm_env.sh` now exits with a short `Python Venv not set at ...` error.

## Repository Structure

The most important directories are:
- `scripts/`: SLURM launchers and top-level training/export entry points
- `config/`: config loading and path helper logic
- `configs/`: shared reference configs, especially `configs/base_config.yaml`
- `models/`: ML backbones, combined model, prior energy, and topology code
- `training/`: trainer wrappers, optimizers, and prior-residual support
- `export/`: MLIR export and re-export tooling
- `analysis_tests/`: evaluation and post-training analysis scripts
- `data/`: runtime dataset loading and preprocessing helpers
- `data_prep/`: offline dataset generation, coarse-graining, and prior fitting
- `env_setup/`: environment, module, and deployment setup helpers
- `local_work/`: ignored local workspace for configs, outputs, and temporary work
