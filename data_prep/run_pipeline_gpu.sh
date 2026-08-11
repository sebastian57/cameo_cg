#!/bin/bash
#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=booster
#SBATCH --output=/dev/null

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -n "${CAMEO_CG_PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="${CAMEO_CG_PROJECT_ROOT}"
else
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
fi

if [[ -n "${CAMEO_ACTIVE_VENV:-}" ]]; then
  VENV_DIR="${CAMEO_ACTIVE_VENV}"
else
  VENV_DIR="${CAMEO_STANDARD_VENV:-}"
fi
if [[ -z "${VENV_DIR}" ]]; then
  echo "ERROR: Set CAMEO_STANDARD_VENV (or CAMEO_ACTIVE_VENV) before submitting." >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# Editable run configuration (all paths can be absolute or project-root relative)
# -----------------------------------------------------------------------------
H5_DIR="${H5_DIR:-data_prep/datasets/dataset_1604_25pro}"
DATE_TAG="$(date +%Y%m%d)"
OUT_DIR="${OUT_DIR:-data_prep/datasets/pipeline_${DATE_TAG}}"
NFRAMES="${NFRAMES:-2500}"
TEMP_GROUPS=(${TEMP_GROUPS:-320})
PRIOR_FIT_T="${PRIOR_FIT_T:-320}"

ENABLE_SPLINE="${ENABLE_SPLINE:-0}"
RESIDUE_SPECIFIC_ANGLES="${RESIDUE_SPECIFIC_ANGLES:-0}"
NORMALIZE_FORCES="${NORMALIZE_FORCES:-0}"
USE_4WAY_GROUPING="${USE_4WAY_GROUPING:-0}"
VERBOSE="${VERBOSE:-0}"
MAPPING="${MAPPING:-1bead}"
N_BUCKETS="${N_BUCKETS:-}"
BUCKET_BOUNDARIES="${BUCKET_BOUNDARIES:-}"
NO_COMBINE="${NO_COMBINE:-0}"
SKIP_PRIOR_FITTING="${SKIP_PRIOR_FITTING:-0}"
# -----------------------------------------------------------------------------

if [[ "${MAPPING}" != "1bead" && "${MAPPING}" != "backbone_cb" ]]; then
  echo "ERROR: MAPPING must be 1bead or backbone_cb (got ${MAPPING})." >&2
  exit 1
fi
if [[ -n "${N_BUCKETS}" && -n "${BUCKET_BOUNDARIES}" ]]; then
  echo "ERROR: Set only one of N_BUCKETS or BUCKET_BOUNDARIES." >&2
  exit 1
fi
if [[ "${NO_COMBINE}" == "1" && ( -n "${N_BUCKETS}" || -n "${BUCKET_BOUNDARIES}" ) ]]; then
  echo "ERROR: NO_COMBINE=1 cannot be combined with bucket options." >&2
  exit 1
fi

if [[ "${H5_DIR}" != /* ]]; then
  H5_DIR="${PROJECT_ROOT}/${H5_DIR}"
fi
if [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${PROJECT_ROOT}/${OUT_DIR}"
fi

mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/slurm-pipeline-${SLURM_JOB_ID:-local}.out"
exec > >(tee -a "${LOG_FILE}") 2>&1

cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/env_setup/load_modules_2026.sh"
source "${VENV_DIR}/bin/activate"
PYTHON_BIN="$(command -v python)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: No python found after activating ${VENV_DIR}." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
export JAX_PLATFORMS=cuda
export JAX_PLATFORM_NAME=cuda
unset ROCM_PATH HSA_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=2

python - <<'PYCHK'
import importlib
import sys
modules = ["aggforce", "qpsolvers", "osqp", "jinja2"]
for name in modules:
    importlib.import_module(name)
import qpsolvers
print("Python:", sys.executable)
print("JAX_PLATFORMS:", __import__("os").environ.get("JAX_PLATFORMS"))
print("Available qpsolvers solvers:", qpsolvers.available_solvers)
try:
    import jax
    print("JAX default backend:", jax.default_backend())
    print("JAX devices:", jax.devices())
except Exception as exc:
    print("WARNING: JAX CUDA preflight failed; continuing anyway:", exc)
PYCHK

CMD=(
  "${PYTHON_BIN}" data_prep/run_pipeline.py
  --h5_dir "${H5_DIR}"
  --out_dir "${OUT_DIR}"
  --nframes "${NFRAMES}"
  --temp "${TEMP_GROUPS[@]}"
  --T "${PRIOR_FIT_T}"
  --mapping "${MAPPING}"
)

if [[ -n "${N_BUCKETS}" ]]; then
  CMD+=(--n_buckets "${N_BUCKETS}")
fi
if [[ -n "${BUCKET_BOUNDARIES}" ]]; then
  read -r -a BUCKET_BOUNDARY_VALUES <<< "${BUCKET_BOUNDARIES}"
  CMD+=(--bucket_boundaries "${BUCKET_BOUNDARY_VALUES[@]}")
fi
if [[ "${NO_COMBINE}" == "1" ]]; then
  CMD+=(--no_combine)
fi
if [[ "${SKIP_PRIOR_FITTING}" == "1" ]]; then
  CMD+=(--skip_prior_fitting)
fi

if [[ "${ENABLE_SPLINE}" == "1" ]]; then
  CMD+=(--spline)
fi
if [[ "${RESIDUE_SPECIFIC_ANGLES}" == "1" ]]; then
  CMD+=(--residue_specific_angles)
fi
if [[ "${NORMALIZE_FORCES}" == "1" ]]; then
  CMD+=(--normalize_forces)
fi
if [[ "${USE_4WAY_GROUPING}" == "1" ]]; then
  CMD+=(--use_4way_grouping)
fi
if [[ "${VERBOSE}" == "1" ]]; then
  CMD+=(--verbose)
fi

source "${PROJECT_ROOT}/runs/registry_hook.sh"
run_registry_start data-preparation "" "${OUT_DIR}"
run_registry_install_exit_trap

echo "============================================================"
echo "Pipeline job configuration"
echo "============================================================"
echo "Project root: ${PROJECT_ROOT}"
echo "Venv:         ${VENV_DIR}"
echo "H5 dir:       ${H5_DIR}"
echo "Out dir:      ${OUT_DIR}"
echo "Frames:       ${NFRAMES}"
echo "Temp groups:  ${TEMP_GROUPS[*]}"
echo "Prior-fit T:  ${PRIOR_FIT_T}"
echo "Mapping:      ${MAPPING}"
echo "N buckets:    ${N_BUCKETS:-none}"
echo "Boundaries:   ${BUCKET_BOUNDARIES:-none}"
echo "No combine:   ${NO_COMBINE}"
echo "Skip priors:  ${SKIP_PRIOR_FITTING}"
echo "Aggforce:     enabled"
echo "Spline:       ${ENABLE_SPLINE}"
echo "CUDA visible: ${CUDA_VISIBLE_DEVICES}"
echo "Command:      ${CMD[*]}"
echo "============================================================"

"${CMD[@]}"
