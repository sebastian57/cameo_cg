# ML Training Testing Suite

This directory contains reproducible training-test configs, a suite-specific
Slurm launcher, a directory-based batch submit helper, and a post-run analysis
script.

Conventions:
- Config prefix: `training_testing_`
- Single-run training layout: `ml_training_testing_suite/outputs/YYYYMMDD_run-name/`
- Grouped training layout: `ml_training_testing_suite/outputs/YYYYMMDD_training_testing_suite-name/<run-name>/`
- Analysis layout for either case: sibling `..._analysis/` directory

Single-config workflow:

```bash
cd /p/project1/cameo/schmidt36/cameo_cg
sbatch ml_training_testing_suite/run_training_testing.slurm   ml_training_testing_suite/training_testing_baseline.yaml
```

Directory-of-configs workflow:

```bash
cd /p/project1/cameo/schmidt36/cameo_cg
./ml_training_testing_suite/submit_training_testing_suite.sh   --input_dir configs_examples   --name example_name
```

That creates a grouped output root like:
- `ml_training_testing_suite/outputs/YYYYMMDD_training_testing_example_name/`
- each config run then lives inside its own child run directory under that root

To analyze one completed single run:

```bash
source /p/project1/cameo/schmidt36/load_modules.sh
source /p/project1/cameo/schmidt36/env_cueq_allegro_opt/bin/activate
python ml_training_testing_suite/analyze_training_testing_suite.py   /p/project1/cameo/schmidt36/cameo_cg/ml_training_testing_suite/outputs/YYYYMMDD_run-name
```

To analyze a grouped batch output directory:

```bash
source /p/project1/cameo/schmidt36/load_modules.sh
source /p/project1/cameo/schmidt36/env_cueq_allegro_opt/bin/activate
python ml_training_testing_suite/analyze_training_testing_suite.py   /p/project1/cameo/schmidt36/cameo_cg/ml_training_testing_suite/outputs/YYYYMMDD_training_testing_example_name
```

Each analysis writes a sibling directory containing:
- `summary.csv`
- `tail_loss_plots/`
- `force_eval_plots/`

Optional detailed held-out force evaluation:

```bash
source /p/project1/cameo/schmidt36/load_modules.sh
source /p/project1/cameo/schmidt36/env_cueq_allegro_opt/bin/activate
python ml_training_testing_suite/analyze_training_testing_suite.py   /p/project1/cameo/schmidt36/cameo_cg/ml_training_testing_suite/outputs/YYYYMMDD_run-name   --detailed-force-eval
```

When that flag is enabled, the analysis directory also contains:
- `detailed_force_eval/`

Inside `detailed_force_eval/<run-name>/` the analyzer writes:
- `metrics.json`
- `metrics.csv`
- `shuffle_rmses.csv`
- `baseline_rmse_comparison.png`
- `shuffle_rmse_distribution.png`
- `cosine_similarity_hist.png`
