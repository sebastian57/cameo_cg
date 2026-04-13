#!/bin/bash -x

#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --output=/dev/null

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -n "${CAMEO_CG_PROJECT_ROOT:-}" ]]; then
    PROJECT_ROOT="${CAMEO_CG_PROJECT_ROOT}"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
fi

if [[ -n "${CAMEO_ACTIVE_VENV:-}" ]]; then
    VENV_DIR="${CAMEO_ACTIVE_VENV}"
else
    VENV_DIR="${CAMEO_CUEQ_VENV:-}"
fi
if [[ -z "${VENV_DIR}" ]]; then
    echo "ERROR: Set CAMEO_CUEQ_VENV (or CAMEO_ACTIVE_VENV) before submitting." >&2
    exit 1
fi

LMP_BIN="${CAMEO_LMP_BIN:-lmp}"
OUTPUT_DIR="${PROJECT_ROOT}/local_work/outputs"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/slurm-lammps-${SLURM_JOB_ID:-local}.out"
exec > >(tee -a "${LOG_FILE}") 2>&1

source "${PROJECT_ROOT}/env_setup/load_modules.sh"
source "${VENV_DIR}/bin/activate"
export PYTHON_BIN="$(command -v python)"
source "${PROJECT_ROOT}/env_setup/set_lammps_paths.sh"

INPUT_FILE="${PROJECT_ROOT}/md_setup/inp_lammps_trained.in"
cd "${PROJECT_ROOT}"

echo "[MLIR preflight] disabled for the modernized JAX 0.9.x deployment path"

export CUDA_VISIBLE_DEVICES=0,1,2,3

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_HOME=/p/software/juwelsbooster/stages/2025/software/CUDA/12
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_cuda_data_dir=${CUDA_HOME}"

srun "${LMP_BIN}" -in "${INPUT_FILE}"
