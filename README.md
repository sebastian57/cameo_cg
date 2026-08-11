# cameo_cg

`cameo_cg` trains and evaluates coarse-grained energy models, exports them for
simulation, runs JAX-MD validation, and supports enhanced-sampling data
acquisition. Allegro/cuEquivariance is the main model path; MACE and PaiNN
remain available through the same config-driven interface.

## Start here

1. Recreate or activate the Jupiter environment with
   [`env_setup/SETUP_ENV.md`](env_setup/SETUP_ENV.md).
2. Read [`WORKFLOW.md`](WORKFLOW.md) for the end-to-end data, training, MD, and
   sampling flow.
3. Use [`COMMANDS.md`](COMMANDS.md) for exact launch commands.
4. Use [`md_setup/README.md`](md_setup/README.md) for JAX-MD and LAMMPS details.

Run launchers from the repository root. Keep experiment-specific configs,
outputs, copied checkpoints, and scratch analysis under `local_work/`, which is
ignored by git.

## Jupiter quick start

The normal `.bashrc` setup defines the repository and venv paths:

```bash
source ~/.bashrc
cd "$CAMEO_CG_PROJECT_ROOT"
mkdir -p local_work/example_fm
cp configs/base_config.yaml local_work/example_fm/config.yaml
# Edit paths.dataset_path and the model/training sections.
sbatch scripts/run_training.sh local_work/example_fm/config.yaml
```

Single-run output defaults to:

```text
local_work/outputs/YYYYMMDD_<config-name>/
```

Set `paths.output_dir` in the YAML when an exact output directory is needed.
The run contains the submitted and resolved configs, logs, checkpoints,
exports, and profiling/analysis artifacts. Register and inspect runs with:

```bash
python3 runs/registry.py sync
python3 runs/registry.py status
```

## Repository layout

| Path | Purpose |
|---|---|
| `configs/` | Shared reference training and MD configs |
| `config/` | YAML loading, validation, and path resolution |
| `data/`, `data_prep/` | Runtime loading and offline preprocessing/assembly |
| `models/` | Allegro, MACE, PaiNN, priors, topology, combined energy |
| `training/` | Force matching, mSAM, REM, optimizers, batching/tiling |
| `md/`, `md_setup/` | JAX-MD runtime/analysis and simulation documentation |
| `sampling/` | TICA, teacher, inversion biases and GROMACS campaigns |
| `analysis_tests/` | Model/run evaluation and plots |
| `export/` | MLIR export and re-export |
| `scripts/` | Slurm launchers and top-level automation |
| `runs/` | Lightweight run registry; generated state is ignored |
| `local_work/` | Local experiment workspace and outputs; ignored |

## Path and environment rules

- Use config-relative or repository-relative paths in YAML; avoid personal
  absolute paths in shared configs.
- `CAMEO_CG_PROJECT_ROOT` locates this checkout.
- `CAMEO_STANDARD_VENV` and `CAMEO_CUEQ_VENV` select the normal environments.
- `CAMEO_ACTIVE_VENV` overrides selection for one shell/job.
- `CAMEO_MD_PROJECT_ROOT` identifies the optional separate MD workspace.
- `CAMEO_LAMMPS_BUILD_DIR` identifies the local LAMMPS build.
- Training and JAX-MD Slurm launchers share `scripts/slurm_env.sh`; JAX-MD
  resolves the referenced training config before choosing a venv.

## Supported workflows

The same training launcher handles standard force matching, mSAM, and relative
entropy/REM according to the YAML configuration. Data enhancement is a loop:
train a baseline, run safe MD and analysis, acquire edge/transition structures
with teacher/TICA/local-inversion campaigns, collect and validate data, assemble
a balanced dataset, then retrain or fine-tune.

The repository does not infer safe production-MD settings. Begin with the short
`configs/example_md.yaml` smoke configuration and only extend duration after
checking forces, temperature, geometry, and neighbor-list behavior.
