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
PROJECT_ROOT="${CAMEO_CG_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
VENV_DIR="${CAMEO_ACTIVE_VENV:-${CAMEO_CUEQ_VENV:-}}"
if [[ -z "${VENV_DIR}" || ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "ERROR: Set CAMEO_CUEQ_VENV (or CAMEO_ACTIVE_VENV) to a valid venv." >&2
    exit 1
fi

DATA_FILE="${CAMEO_LAMMPS_DATA_FILE:-}"
MODEL_FILE="${CAMEO_LAMMPS_MODEL_FILE:-}"
if [[ -z "${DATA_FILE}" || ! -f "${DATA_FILE}" ]]; then
    echo "ERROR: Set CAMEO_LAMMPS_DATA_FILE to an existing LAMMPS data file." >&2
    exit 1
fi
if [[ -z "${MODEL_FILE}" || ! -f "${MODEL_FILE}" ]]; then
    echo "ERROR: Set CAMEO_LAMMPS_MODEL_FILE to an existing MLIR model." >&2
    exit 1
fi

LMP_BIN="${CAMEO_LMP_BIN:-lmp}"
INPUT_FILE="${CAMEO_LAMMPS_INPUT_FILE:-${PROJECT_ROOT}/md_setup/inp_lammps_trained.in}"
OUTPUT_DIR="${CAMEO_LAMMPS_OUTPUT_DIR:-${PROJECT_ROOT}/local_work/outputs/lammps_md}"
TEMPERATURE="${CAMEO_LAMMPS_TEMPERATURE:-320.0}"
TIMESTEP_FS="${CAMEO_LAMMPS_TIMESTEP_FS:-1.0}"
RUN_STEPS="${CAMEO_LAMMPS_RUN_STEPS:-10000}"
DUMP_FILE="${CAMEO_LAMMPS_DUMP_FILE:-${OUTPUT_DIR}/trajectory.dump}"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/slurm-lammps-${SLURM_JOB_ID:-local}.out"
exec > >(tee -a "${LOG_FILE}") 2>&1

source "${PROJECT_ROOT}/env_setup/load_modules_2026.sh"
source "${VENV_DIR}/bin/activate"
export PYTHON_BIN="$(command -v python)"
source "${PROJECT_ROOT}/env_setup/set_lammps_paths_2026.sh"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_cuda_data_dir=${CUDA_HOME}"

source "${PROJECT_ROOT}/runs/registry_hook.sh"
run_registry_start lammps "" "${OUTPUT_DIR}"
run_registry_install_exit_trap

srun "${LMP_BIN}" -in "${INPUT_FILE}" \
    -var data_file "${DATA_FILE}" \
    -var model_file "${MODEL_FILE}" \
    -var temperature "${TEMPERATURE}" \
    -var timestep_fs "${TIMESTEP_FS}" \
    -var run_steps "${RUN_STEPS}" \
    -var dump_file "${DUMP_FILE}"
