#!/bin/bash

#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=develbooster
#SBATCH --output=/p/project1/cameo/schmidt36/cameo_cg/ml_training_testing_suite/outputs/slurm-analysis-%j.out

set -Eeuo pipefail

RUN_ANALYSIS_TRACE="${RUN_ANALYSIS_TRACE:-0}"
if [[ "${RUN_ANALYSIS_TRACE}" == "1" ]]; then
    export PS4='[run_analysis.slurm:${LINENO}] '
    set -x
fi

run_analysis_err_trap() {
    local rc=$?
    echo "[run_analysis.slurm] ERROR rc=${rc} line=${BASH_LINENO[0]} cmd=${BASH_COMMAND}" >&2
    exit "${rc}"
}
trap run_analysis_err_trap ERR

SCRIPT_DIR="${CAMEO_TRAINING_TESTING_SUITE_DIR:-}"
if [[ -n "${SCRIPT_DIR}" && -d "${SCRIPT_DIR}" ]]; then
    SCRIPT_DIR="$(cd "${SCRIPT_DIR}" && pwd -P)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
fi

INPUT_DIR="${1:-}"
if [[ -z "${INPUT_DIR}" ]]; then
    echo "Usage: sbatch ml_training_testing_suite/run_analysis.slurm <run_dir> [extra analyze args]"
    echo "  <run_dir>: single run directory or directory containing multiple run directories"
    exit 1
fi
shift || true

if [[ "${INPUT_DIR}" != /* && -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/${INPUT_DIR}" ]]; then
    INPUT_DIR="${SLURM_SUBMIT_DIR}/${INPUT_DIR}"
fi
INPUT_DIR="$(cd "${INPUT_DIR}" && pwd -P)"

if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "ERROR: Input directory not found: ${INPUT_DIR}"
    exit 1
fi

source /p/project1/cameo/schmidt36/load_modules.sh
source /p/project1/cameo/schmidt36/env_cueq_allegro_opt/bin/activate
source /p/project1/cameo/schmidt36/set_lammps_paths.sh

PYTHON_BIN="$(command -v python)"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "ERROR: No python found after activating env_cueq_allegro_opt"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0

CUDA_ROOT="$("${PYTHON_BIN}" -c 'import os; from jax_plugins import xla_cuda12; print(os.path.dirname(xla_cuda12.__file__))')"
SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${CUDA_ROOT}:${SITE_PACKAGES}/nvidia/cudnn/lib:${SITE_PACKAGES}/nvidia/cuda_runtime/lib:${SITE_PACKAGES}/nvidia/cublas/lib:${SITE_PACKAGES}/nvidia/cusolver/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME=/p/software/juwelsbooster/stages/2025/software/CUDA/12
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME} --xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy

JOB_TAG="${SLURM_JOB_ID:-local}"
ANALYSIS_SCRIPT="${SCRIPT_DIR}/analyze_training_testing_suite.py"

echo "============================================================"
echo "ML Training Testing Suite — Analysis"
echo "============================================================"
echo "Input dir:         ${INPUT_DIR}"
echo "Analysis script:   ${ANALYSIS_SCRIPT}"
echo "Job tag:           ${JOB_TAG}"
echo "CUDA device:       ${CUDA_VISIBLE_DEVICES}"
echo "Module Environment"
echo "============================================================"
module list || true
echo "============================================================"

"${PYTHON_BIN}" -u "${ANALYSIS_SCRIPT}" "${INPUT_DIR}" --detailed-force-eval --devices-per-run 1 "$@"
