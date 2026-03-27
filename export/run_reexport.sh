#!/bin/bash

#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Booster jobs use a 1-process-per-node pattern and request the full 4-GPU
# node allocation even when the script only uses CUDA_VISIBLE_DEVICES=0.
#SBATCH --gpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=develbooster
#SBATCH --output=slurm-reexport-%j.out

# =============================================================================
# SLURM wrapper for post-hoc MLIR re-export.
#
# Relative input paths are resolved from the directory where `sbatch` is run.
# Intended usage: submit this from export/ so local filenames are relative to
# export/ and the script can locate ../scripts/slurm_env.sh reliably.
# =============================================================================

set -Eeuo pipefail

run_reexport_err_trap() {
    local rc=$?
    echo "[run_reexport.sh] ERROR rc=${rc} line=${BASH_LINENO[0]} cmd=${BASH_COMMAND}" >&2
    exit "${rc}"
}
trap run_reexport_err_trap ERR

SUBMIT_WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
SUBMIT_WORKDIR="$(cd "${SUBMIT_WORKDIR}" && pwd -P)"
cd "${SUBMIT_WORKDIR}"

# On the compute node BASH_SOURCE points at a SLURM spool copy, so derive the
# real script directory from the submit directory when submitting from export/.
if [[ -f "${SUBMIT_WORKDIR}/reexport_mlir.py" && -f "${SUBMIT_WORKDIR}/run_reexport.sh" ]]; then
    SCRIPT_DIR="${SUBMIT_WORKDIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
fi
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
SHARED_SCRIPT_DIR="${PROJECT_ROOT}/scripts"

if [[ ! -f "${SHARED_SCRIPT_DIR}/slurm_env.sh" ]]; then
    echo "ERROR: Could not locate shared SLURM env script: ${SHARED_SCRIPT_DIR}/slurm_env.sh" >&2
    echo "       Submit this launcher from the export/ directory of the repo." >&2
    exit 1
fi

resolve_file() {
    local path="$1"
    if [[ -z "${path}" ]]; then
        return 1
    fi
    if [[ "${path}" == /* ]]; then
        printf "%s
" "${path}"
    else
        printf "%s
" "${SUBMIT_WORKDIR}/${path}"
    fi
}

PARAMS_FILE="${1:-}"
CONFIG_FILE_INPUT="${2:-}"
if [[ $# -lt 2 ]]; then
    cat <<'EOF'
Usage: sbatch export/run_reexport.sh <params.pkl> <config.yaml> [extra args]

Relative paths are resolved from the directory where sbatch is run.
Recommended: cd export/ first, then submit from there.

Example:
  sbatch run_reexport.sh       model_ml_only.pkl       config_prior_export.yaml       --mode combined --prior-source config --output-name model_with_priors
EOF
    exit 1
fi
shift 2
EXTRA_ARGS=("$@")

PARAMS_FILE="$(resolve_file "${PARAMS_FILE}")"
CONFIG_FILE="$(resolve_file "${CONFIG_FILE_INPUT}")"

if [[ ! -f "${PARAMS_FILE}" ]]; then
    echo "ERROR: params file not found: ${PARAMS_FILE}" >&2
    exit 1
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi

export CONFIG_FILE
source "${SHARED_SCRIPT_DIR}/slurm_env.sh"

export CUDA_VISIBLE_DEVICES=0

SAVE_DIR="${PROJECT_ROOT}/saved_models"
mkdir -p "${SAVE_DIR}"
LOGFILE="${SAVE_DIR}/reexport_${SLURM_JOB_ID:-local}.log"
exec > >(tee -a "${LOGFILE}") 2>&1

echo "============================================================"
echo "Post-hoc MLIR Re-export"
echo "============================================================"
echo "Submit dir:  ${SUBMIT_WORKDIR}"
echo "Script dir:  ${SCRIPT_DIR}"
echo "Project dir: ${PROJECT_ROOT}"
echo "Params:      ${PARAMS_FILE}"
echo "Config:      ${CONFIG_FILE}"
echo "Output dir:  ${SAVE_DIR}"
echo "Extra args:  ${EXTRA_ARGS[*]:-none}"
echo "============================================================"

"${PYTHON_BIN}" -u "${PROJECT_ROOT}/export/reexport_mlir.py"     "${PARAMS_FILE}"     "${CONFIG_FILE}"     "${EXTRA_ARGS[@]}"
