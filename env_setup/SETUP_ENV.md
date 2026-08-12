# Jupiter environment setup

This is the source of truth for the Python and shell environment used by
`cameo_cg`. The instructions target Jupiter's 2026 software stack. Other
clusters need equivalent Python, CUDA, compiler, and scheduler setup.

`env_setup/interactive_job.md` is retained as a legacy scheduler reference;
it is not the environment installation guide.

## Expected directory layout

Keep the editable repositories beside one another:

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

The active installation imports ChemTrain from the `chemtrain_cameo`
checkout, Chemutils from `chemtrain-deploy/external/chemutils`, and AggForce
from the sibling `aggforce` checkout. Do not install a second PyPI ChemTrain
over the editable checkout.

## 1. Load Jupiter modules

Run environment creation on a GPU compute node when possible, then load the
same module family in jobs:

```bash
cd "$CAMEO_CG_PROJECT_ROOT"
source env_setup/load_modules_2026.sh
```

The helper loads Python 3.13.5, CUDA 13, GCC 14.3, and the build/runtime
modules used by the current environment. It deliberately does not load the
system JAX module.

## 2. Create the venv

```bash
python -m venv --system-site-packages /path/to/venv_cameocg_jupiter2026
source /path/to/venv_cameocg_jupiter2026/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

The validated Jupiter venv uses `include-system-site-packages = true`.

## 3. Install the main packages

Install the project's editable repositories first:

```bash
python -m pip install -e /path/to/aggforce
python -m pip install -e /path/to/chemtrain-deploy/external/chemutils
python -m pip install -e '/path/to/chemtrain-deploy/external/chemtrain/chemtrain_cameo[all]'
```

The currently validated key versions are:

- JAX, JAXlib, and CUDA 12 plugin/PJRT: 0.10.1
- JAX-MD: 0.2.28
- Flax: 0.12.7
- Optax: 0.2.8
- e3nn-jax: 0.21.0
- cuequivariance, cuequivariance-jax, and cu12 ops: 0.10.0
- ChemTrain: editable `chemtrain_cameo` checkout
- Chemutils and AggForce: editable local checkouts

Also install the normal scientific/IO tools used by the repo (`numpy`,
`scipy`, `h5py`, `pyyaml`, `matplotlib`, `mdtraj`, `ase`, `pytest`). CUDA
wheels may require a package mirror or prepared wheel cache.

`env_setup/requirements_git.txt` is a complete `pip freeze` snapshot of the
working Jupiter venv. It contains machine paths and VCS revisions and exists
for comparison and recovery, not as a portable one-command installer.

## 4. Configure persistent paths

Add one managed block to `~/.bashrc` (change the prefix for another checkout):

```bash
# >>> cameo_cg env >>>
export CAMEO_LAMMPS_BUILD_DIR=/e/project1/cameo/schmidt36/lammps/build
export CAMEO_CG_PROJECT_ROOT=/e/project1/cameo/schmidt36/cameo_cg
export CAMEO_CUEQ_VENV=/e/project1/cameo/schmidt36/venv_cameocg_jupiter2026
export CAMEO_STANDARD_VENV=/e/project1/cameo/schmidt36/venv_cameocg_jupiter2026
export CAMEO_MD_PROJECT_ROOT=/e/project1/cameo/schmidt36/cameo_md
export PATH="$HOME/.local/bin:$PATH"
# <<< cameo_cg env <<<
```

Reload with `source ~/.bashrc`. To update or migrate the managed block, run:

```bash
bash scripts/configure_user_env.sh
```

`CAMEO_ACTIVE_VENV` is an optional per-command override. Otherwise,
`scripts/slurm_env.sh` selects `CAMEO_CUEQ_VENV` for `allegro_cueq*` configs
and `CAMEO_STANDARD_VENV` for other models. JAX-MD configs point to a training
config; the helper resolves it before selecting the venv.

## 5. Verify imports and accelerator discovery

```bash
source env_setup/load_modules_2026.sh
source "$CAMEO_STANDARD_VENV/bin/activate"
python - <<'PY'
import jax, jax_md, chemtrain, chemutils, aggforce
import flax, optax, e3nn_jax, cuequivariance
print('JAX:', jax.__version__, jax.__file__)
print('devices:', jax.devices())
PY
```

The August 2026 Jupiter jobs report JAX 0.10.1 from the venv's Python 3.13
`site-packages`, not the system JAX module.

For a quick repository check:

```bash
cd "$CAMEO_CG_PROJECT_ROOT"
python -m pytest -q tests/test_run_registry_launchers.py
python scripts/train.py --help
python scripts/run_md.py --help
```

After these checks pass, return to the
[README new-user checklist](../README.md#new-user-checklist) to prepare data and
submit the first force-matching run.

## LAMMPS connector

LAMMPS has a separate compiled dependency chain. See `md_setup/README.md` for
normal use. `env_setup/LAMMPS_build.md` and `env_setup/CONNECTOR_REBUILD.md`
are rebuild/legacy records and should not override this runtime environment.

## Common failures

- **Wrong JAX:** load `load_modules_2026.sh`, reactivate the venv, and print
  both `jax.__version__` and `jax.__file__`.
- **Venv variable unset:** export `CAMEO_STANDARD_VENV` and
  `CAMEO_CUEQ_VENV`, or set `CAMEO_ACTIVE_VENV` for that command.
- **Wrong ChemTrain:** inspect
  `python -c 'import chemtrain; print(chemtrain.__file__)'` and reinstall the
  intended checkout editable.
- **cuEquivariance failure:** ensure the core, JAX adapter, and CUDA ops use
  the same version and that the Jupiter modules were loaded.
- **Direct ChemTrain trainer import fails on `jax.tree_map`:** supported repo
  entry points call `utils.jax_setup.apply_jax_compat_shims()` before importing
  ChemTrain/JAX-MD. Do the same in new standalone entry points; do not infer
  that the working JAX version comes from the system module.
- **Root-level `pytest` collection fails on `active_learning`:** that analysis
  test belongs to an optional sibling package. The maintained repository tests
  can be run with `python -m pytest -q tests`; install the optional package only
  when working on that separate analysis path.
- **Login/compute mismatch:** submit through the repository launchers; they
  source the shared module/environment resolver.
