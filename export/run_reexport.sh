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

# On the compute node BASH_SOURCE can point to a SLURM spool copy. Prefer
# resolving the real export/ directory from the original submit directory.
if [[ -f "${SUBMIT_WORKDIR}/reexport_mlir.py" && -f "${SUBMIT_WORKDIR}/run_reexport.sh" ]]; then
    # Submitted from export/
    SCRIPT_DIR="${SUBMIT_WORKDIR}"
elif [[ -f "${SUBMIT_WORKDIR}/export/reexport_mlir.py" && -f "${SUBMIT_WORKDIR}/export/run_reexport.sh" ]]; then
    # Submitted from repo root via: sbatch export/run_reexport.sh ...
    SCRIPT_DIR="${SUBMIT_WORKDIR}/export"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
fi
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
SHARED_SCRIPT_DIR="${PROJECT_ROOT}/scripts"

if [[ ! -f "${SHARED_SCRIPT_DIR}/slurm_env.sh" ]]; then
    echo "ERROR: Could not locate shared SLURM env script: ${SHARED_SCRIPT_DIR}/slurm_env.sh" >&2
    echo "       Submit this launcher from the repo root or export/ directory." >&2
    exit 1
fi

resolve_file() {
    local path="$1"
    if [[ -z "${path}" ]]; then
        return 1
    fi
    if [[ "${path}" == /* ]]; then
        printf "%s\n" "${path}"
    else
        printf "%s\n" "${SUBMIT_WORKDIR}/${path}"
    fi
}

PARAMS_FILE="${1:-}"
CONFIG_FILE_INPUT="${2:-}"
if [[ $# -lt 2 ]]; then
    cat <<'USAGE'
Usage: sbatch export/run_reexport.sh <params.pkl> <config.yaml> [extra args]

Relative paths are resolved from the directory where sbatch is run.

Example:
  sbatch export/run_reexport.sh \
      /path/to/model_checkpoint.pkl \
      /path/to/config_runtime.yaml \
      --mode combined --prior-source config --output-name model_with_priors
USAGE
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

# ---------------------------------------------------------------------------
# Output routing:
# - If --output-dir is explicitly provided, respect it.
# - Otherwise, infer run directory from params/config and export into
#   <run_dir>/exports (matching training layout).
# ---------------------------------------------------------------------------
has_output_dir_arg=0
for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "${arg}" == "--output-dir" || "${arg}" == --output-dir=* ]]; then
        has_output_dir_arg=1
        break
    fi
done

PARAMS_DIR="$(cd "$(dirname "${PARAMS_FILE}")" && pwd -P)"
CONFIG_DIR="$(cd "$(dirname "${CONFIG_FILE}")" && pwd -P)"

is_parent_dir_of() {
    local parent="$1"
    local child="$2"
    [[ "${child}" == "${parent}" || "${child}" == "${parent}/"* ]]
}

COMMON_DIR="${PARAMS_DIR}"
while ! is_parent_dir_of "${COMMON_DIR}" "${CONFIG_DIR}"; do
    next_common="$(dirname "${COMMON_DIR}")"
    if [[ "${next_common}" == "${COMMON_DIR}" ]]; then
        break
    fi
    COMMON_DIR="${next_common}"
done

if [[ "${COMMON_DIR}" == "/" ]]; then
    RUN_BASE_DIR="${CONFIG_DIR}"
else
    RUN_BASE_DIR="${COMMON_DIR}"
fi

DEFAULT_EXPORT_DIR="${RUN_BASE_DIR}/exports"
if [[ ${has_output_dir_arg} -eq 0 ]]; then
    EXTRA_ARGS+=("--output-dir" "${DEFAULT_EXPORT_DIR}")
fi

export CONFIG_FILE
source "${SHARED_SCRIPT_DIR}/slurm_env.sh"

export CUDA_VISIBLE_DEVICES=0

LOG_DIR="${RUN_BASE_DIR}"
mkdir -p "${LOG_DIR}" "${DEFAULT_EXPORT_DIR}"
LOGFILE="${LOG_DIR}/reexport_${SLURM_JOB_ID:-local}.log"
exec > >(tee -a "${LOGFILE}") 2>&1

echo "============================================================"
echo "Post-hoc MLIR Re-export"
echo "============================================================"
echo "Submit dir:  ${SUBMIT_WORKDIR}"
echo "Script dir:  ${SCRIPT_DIR}"
echo "Project dir: ${PROJECT_ROOT}"
echo "Params:      ${PARAMS_FILE}"
echo "Config:      ${CONFIG_FILE}"
echo "Log dir:     ${LOG_DIR}"
echo "Export dir:  ${DEFAULT_EXPORT_DIR}"
echo "Extra args:  ${EXTRA_ARGS[*]:-none}"
echo "============================================================"

"${PYTHON_BIN}" -u "${PROJECT_ROOT}/export/reexport_mlir.py" \
    "${PARAMS_FILE}" \
    "${CONFIG_FILE}" \
    "${EXTRA_ARGS[@]}"
