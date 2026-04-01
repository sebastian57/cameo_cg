#!/bin/bash -x

#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
VENV_DIR="${CAMEO_CUEQ_VENV:-/p/project1/cameo/schmidt36/env_cueq_allegro_opt}"
LMP_BIN="${CAMEO_LMP_BIN:-lmp}"

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
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_cuda_data_dir=$CUDA_HOME"

srun "${LMP_BIN}" -in "${INPUT_FILE}"
