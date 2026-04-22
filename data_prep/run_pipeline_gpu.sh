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
# -----------------------------------------------------------------------------

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

module purge
module load Stages/2025 StdEnv/2025
module load GCC/13.3.0 Python/3.12.3
module load CUDA/12 ParaStationMPI cuDNN/9.5.0.50-CUDA-12 NCCL/default-CUDA-12
module load jax/0.4.34-CUDA-12
module load CMake/3.30.3 Ninja/1.12.1 Clang/18.1.8 UCX/default UCC/default git/2.45.1 HDF5/1.14.5-serial

source "${VENV_DIR}/bin/activate"

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
  python3 data_prep/run_pipeline.py
  --h5_dir "${H5_DIR}"
  --out_dir "${OUT_DIR}"
  --nframes "${NFRAMES}"
  --temp "${TEMP_GROUPS[@]}"
  --T "${PRIOR_FIT_T}"
)

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
echo "Aggforce:     enabled"
echo "Spline:       ${ENABLE_SPLINE}"
echo "CUDA visible: ${CUDA_VISIBLE_DEVICES}"
echo "Command:      ${CMD[*]}"
echo "============================================================"

"${CMD[@]}"
