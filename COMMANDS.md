# Command Reference

## Core Pattern

Run commands from the repository root, but keep configs and outputs in `local_work/`.

```bash
mkdir -p local_work/example_run
cp configs/base_config.yaml local_work/example_run/example_config.yaml
sbatch ./scripts/run_training.sh local_work/example_run/example_config.yaml
```

Single-run outputs appear under:

```text
local_work/outputs/YYYYMMDD_example_config/
```

## Training

Single node:

```bash
sbatch ./scripts/run_training.sh local_work/example_run/example_config.yaml
```

Two nodes:

```bash
sbatch --nodes=2 ./scripts/run_training.sh local_work/example_run/example_config.yaml
```

Resume from the latest checkpoint:

```bash
sbatch ./scripts/run_training.sh local_work/example_run/example_config.yaml --resume auto
```

Resume from a specific checkpoint:

```bash
sbatch ./scripts/run_training.sh   local_work/example_run/example_config.yaml   --resume local_work/outputs/YYYYMMDD_example_config/checkpoints/stage_sgd_nesterov_epoch2.pkl
```

## Suite Submission

Prepare a config directory under `local_work/`, then submit:

```bash
bash ./scripts/submit_suite.sh   --input_dir local_work/my_suite/configs   --name my_suite
```

Suite outputs are written under `local_work/outputs` by default:

```text
local_work/outputs/YYYYMMDD_training_suite_my_suite/
```

## Monitoring

Follow the main training log:

```bash
tail -f local_work/outputs/YYYYMMDD_example_config/train_<jobid>.log
```

Follow the SLURM log:

```bash
tail -f local_work/outputs/YYYYMMDD_example_config/slurm-<jobid>.out
```

Cluster helpers:

```bash
squeue -u $USER
scontrol show job <jobid>
scancel <jobid>
```

## Export

Post-hoc MLIR re-export:

```bash
python export/reexport_mlir.py   /path/to/model_params.pkl   /path/to/model_config.yaml   --mode combined   --prior-source config   --output-name model_with_priors
```

Batch re-export:

```bash
sbatch export/run_reexport.sh   /path/to/model_params.pkl   /path/to/export_config.yaml   --mode combined   --prior-source config   --output-name model_with_priors
```

## Evaluation And Analysis

Force evaluation:

```bash
python analysis_tests/evaluate_forces.py   /path/to/model_params.pkl   /path/to/config.yaml   --frames 50
```

Suite analysis:

```bash
sbatch scripts/run_analysis.sh /path/to/suite/output
```

```bash
sbatch scripts/run_analysis.sh \
  --input-dir /p/project1/cameo/schmidt36/cameo_md/outputs/<your_run_dir> \
  --name <analysis_name> \
  --detailed-force-eval \
  --complete-eval \
  --detailed-batch-size 8 \
  --complete-eval-batch-size 4
```


Direct suite analysis:

```bash
python analysis_tests/analyze_suite.py /path/to/suite/output --detailed-force-eval
```

## Data Preparation

Coarse-grain a trajectory:

```bash
python data_prep/cg_1bead.py   --npz raw_data/protein_allatom.npz   --pdb raw_data/protein_topology.pdb   --output data_prep/datasets/protein_cg.npz
```

Fit priors:

```bash
python data_prep/prior_fitting_script.py   --data /path/to/dataset.npz   --out_yaml data_prep/fitted_priors.yaml   --plots_dir data_prep/plots   --T 320.0   --spline   --spline_out data_prep/datasets/fitted_priors_spline.npz
```

## Environment

Primary setup docs:
- `env_setup/SETUP_ENV.md`
- `env_setup/interactive_job.md`
- `env_setup/LAMMPS_build.md`
- `CONNECTOR_REBUILD.md`

Override the selected environment for a run:

```bash
export CAMEO_ACTIVE_VENV=/path/to/venv
sbatch ./scripts/run_training.sh local_work/example_run/example_config.yaml
```

## Repository Map

- `scripts/`: launchers and top-level entry points
- `config/`: config manager and path helpers
- `configs/`: shared reference configs
- `models/`: ML and prior-energy code
- `training/`: trainer wrappers and optimizer setup
- `export/`: deployment export code
- `analysis_tests/`: evaluation scripts
- `data/`: runtime data loading
- `data_prep/`: offline preprocessing
- `env_setup/`: environment setup helpers
- `local_work/`: ignored local experiment workspace
