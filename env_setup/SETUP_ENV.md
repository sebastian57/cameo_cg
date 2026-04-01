# Environment Setup

Use this workflow when you need to recreate the `cameo_cg_cleanup` Python environment on a new HPC system where:

- the non-Python system stack is already available
- internet access is not available from the compute hardware
- package installation must therefore happen from local clones

## Overview

The clean local setup is:

1. Clone `cameo_cg_cleanup`
2. Clone `chemtrain-deploy`
3. Clone `chemtrain` inside `chemtrain-deploy/external/chemtrain`
4. Start an interactive job on a compute node
5. Create and activate the Python environment
6. Install local editable `chemtrain` and `chemutils`
7. Install the remaining pinned packages from the portable requirements file in this repo

## Repository Layout

Create a layout like this:

```text
work/
├── cameo_cg_cleanup/
└── chemtrain-deploy/
    └── external/
        ├── chemtrain/
        └── chemutils/
```

Notes:

- `chemutils/` is already part of the `chemtrain-deploy` repository under `external/chemutils`
- `chemtrain/` should be a separate clone placed inside `chemtrain-deploy/external/chemtrain`

## 1. Clone The Repositories

Run these on a machine where you can access the repositories:

```bash
git clone <cameo_cg_cleanup_repo_url>
git clone https://github.com/tummfm/chemtrain-deploy.git
git clone https://github.com/tummfm/chemtrain.git chemtrain-deploy/external/chemtrain
```

If you want to match the currently used commits exactly, do this only if the latest versions do not work for you:

```bash
cd chemtrain-deploy
git checkout d88382e081683bdd0d4f5282e63fca4ce58793c2

cd external/chemtrain
git checkout 9cad115c715b3f9df7813410153c2fc192a8240c
```

## 2. Start An Interactive Compute Job

Create the environment on the hardware that will actually run the jobs.

Load the required system modules first, then start an interactive job on a compute node using your site-specific scheduler commands.

After the job starts, continue the setup there.

## 3. Load Modules And Create The Virtual Environment

From the `env_setup` directory of `cameo_cg_cleanup`:

```bash
source load_modules.sh
```

Then create the Python environment where you want it to live:

```bash
python3.12 -m venv env_cueq
source env_cueq/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

Python 3.12 is recommended to match the current environment.

## 4. Install chemtrain And chemutils Locally

From the `chemtrain-deploy` directory:

```bash
cd /path/to/chemtrain-deploy

pip install -e "external/chemtrain_cameo[all]"
pip install -e "external/chemutils"
```

This avoids any network access during installation.

## 5. Install The Remaining Packages

After `chemtrain` and `chemutils` are installed locally, install the remaining pinned packages from the portable requirements file shipped with `cameo_cg_cleanup`:

```bash
cd /path/to/cameo_cg_cleanup

pip install -r env_setup/requirements_cueq_env.txt
```

This requirements file is already cleaned for local/offline use:

- it does not contain the Git-based editable installs for `chemtrain` and `chemutils`
- it replaces the machine-local `z3-solver @ file://...` entry with `z3-solver==4.13.0.0`

Important pinned packages in this environment include:

- `jax==0.9.1`
- `jaxlib==0.9.1`
- `jax-cuda12-pjrt==0.9.1`
- `jax-cuda12-plugin==0.9.1`
- `cuequivariance==0.9.0`
- `cuequivariance-jax==0.9.0`
- `cuequivariance-ops-cu12==0.9.0`
- `cuequivariance-ops-jax-cu12==0.9.0`

## 6. Sanity Check

Run:

```bash
python -c "import jax, chemtrain, chemutils, flax, e3nn_jax, cuequivariance; print('ok')"
```

If you want to verify the JAX version explicitly:

```bash
python -c "import jax; print(jax.__version__)"
```

## 7. Repo Runtime Setup

For normal repo usage, set:

```bash
export CONFIG_FILE=/path/to/cameo_cg_cleanup/configs/base_config.yaml
```

Then run commands from inside the `cameo_cg_cleanup` repository.

## Summary

The minimal local install order is:

```bash
source /path/to/cameo_cg_cleanup/env_setup/load_modules.sh

python3.12 -m venv env_cueq
source env_cueq/bin/activate
python -m pip install --upgrade pip setuptools wheel

cd /path/to/chemtrain-deploy
pip install -e "external/chemtrain[all]"
pip install -e "external/chemutils"

cd /path/to/cameo_cg_cleanup
pip install -r env_setup/requirements_cueq_env.txt
```
