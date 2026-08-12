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

## New-user checklist

1. Create a workspace with this sibling layout:

   ```text
   <workspace>/
   ├── cameo_cg/
   ├── aggforce/
   ├── chemtrain-deploy/
   │   └── external/
   │       ├── chemutils/
   │       └── chemtrain/
   │           └── chemtrain_cameo/
   └── venv_cameocg_jupiter2026/
   ```

2. Clone `cameo_cg`, `aggforce`, `chemtrain-deploy`, and `chemtrain_cameo`.
3. Place the ChemTrain checkout at
   `chemtrain-deploy/external/chemtrain/chemtrain_cameo`; use the Chemutils
   checkout already under `chemtrain-deploy/external/chemutils`.
4. Follow [Jupiter environment setup](env_setup/SETUP_ENV.md) to load the 2026
   modules and create the Python 3.13 system-site-packages venv.
5. Complete that guide's editable installs and supported numerical-package
   cross-check. The package list and exact commands live only there.
6. Add the documented `CAMEO_*` variables to `.bashrc`, then reload it.
7. Run the setup guide's import/device and repository-entry-point checks.
8. Place or link a coarse-grained NPZ dataset under `local_work/`.
9. Copy the first-FM starter, edit its required inputs, and submit it:

   ```bash
   cd "$CAMEO_CG_PROJECT_ROOT"
   mkdir -p local_work/first_fm
   cp configs/example_training.yaml local_work/first_fm/config.yaml
   # Edit data.path, paths.output_dir, model.allegro.num_types, and mapping-specific model/prior choices.
   sbatch scripts/run_training.sh local_work/first_fm/config.yaml
   ```

   The starter retains a production-scale schedule; shorten its optimizer-stage
   epoch counts before using it as a smoke run.
10. Use the cross-reference below to find later workflows without scanning the
    repository.

| Need | Read | Use |
|---|---|---|
| Installation | [Environment setup](env_setup/SETUP_ENV.md) | Its layout, venv, package, shell, and verification sections |
| Data preprocessing | [Workflow: prepare data](WORKFLOW.md#2-prepare-data) | [Command reference: data preprocessing](COMMANDS.md#data-preprocessing) |
| First FM | [Workflow: configure a run](WORKFLOW.md#3-configure-a-model-and-training-run) | [`configs/example_training.yaml`](configs/example_training.yaml) |
| Full training settings | [Workflow: batching and losses](WORKFLOW.md#4-choose-batching-and-loss-semantics) | [`configs/base_config.yaml`](configs/base_config.yaml), the complete stable annotated lookup |
| mSAM / REM | [Workflow: FM and REM](WORKFLOW.md#5-train-and-inspect-force-matching) | [`COMMANDS.md`](COMMANDS.md) and the disabled method blocks in `configs/base_config.yaml` |
| Analysis / export | [Workflow: evaluation and provenance](WORKFLOW.md#6-evaluate-before-md) | [Analysis and export commands](COMMANDS.md#model-evaluation-and-run-analysis) |
| JAX-MD | [MD setup and safety](md_setup/README.md#jax-md-smoke-run) | [`configs/example_md.yaml`](configs/example_md.yaml), the annotated MD lookup and safe smoke template |
| LAMMPS | [LAMMPS/MLIR path](md_setup/README.md#lammpsmlir-path) | [Export and analysis commands](COMMANDS.md#mlir-export) |
| Enhanced sampling | [Workflow: acquire difficult configurations](WORKFLOW.md#8-acquire-difficult-configurations) | [Campaign commands](COMMANDS.md#enhanced-sampling-campaigns) and [`sampling/campaigns/`](sampling/campaigns/) |

## Run outputs

After the first submission, the launcher writes single-run output by default to:

```text
local_work/outputs/YYYYMMDD_<config-name>/
```

Set `paths.output_dir` in the YAML when an exact output directory is needed.
The run contains the submitted and resolved configs, logs, checkpoints,
exports, and profiling/analysis artifacts. Use the
[run-registry commands](COMMANDS.md#run-registry-and-monitoring) to register and
inspect runs.

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
