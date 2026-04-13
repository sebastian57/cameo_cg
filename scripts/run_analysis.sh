#!/bin/bash

#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=develbooster
#SBATCH --output=/dev/null

# =============================================================================
# SLURM wrapper for suite analysis.
#
# Usage:
#   sbatch scripts/run_analysis.sh --input-dir <run_dir> [--name <label>] \
#       [--detailed-force-eval] [--complete-eval] [extra args]
# =============================================================================

set -Eeuo pipefail

# Prefer explicit override, then submit dir, then script-relative fallback.
SCRIPT_SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -n "${CAMEO_CG_PROJECT_ROOT:-}" ]]; then
    PROJECT_ROOT="${CAMEO_CG_PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "${SCRIPT_SRC_DIR}/.." && pwd -P)"
fi
SCRIPT_DIR="${PROJECT_ROOT}/scripts"

# Validate root so we never silently run from the wrong repo.
if [[ ! -f "${SCRIPT_DIR}/slurm_env.sh" || ! -f "${PROJECT_ROOT}/analysis_tests/analyze_suite.py" ]]; then
    echo "ERROR: PROJECT_ROOT is not a valid cameo_cg checkout: ${PROJECT_ROOT}" >&2
    echo "       Submit from cameo_cg root or set CAMEO_CG_PROJECT_ROOT explicitly." >&2
    exit 1
fi

# --- Parse named arguments ---
INPUT_DIR=""
RUN_NAME=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir|--input_dir)
            INPUT_DIR="$2"; shift 2 ;;
        --name)
            RUN_NAME="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "${INPUT_DIR}" ]]; then
    echo "Usage: sbatch scripts/run_analysis.sh --input-dir <run_dir> [--name <label>] [extra args]"
    echo ""
    echo "Extra args are forwarded directly to analyze_suite.py, e.g.:"
    echo "  --detailed-force-eval   Run detailed baseline/correlation eval"
    echo "  --complete-eval         Run the full 7-module diagnostic suite"
    echo "  --skip-force-eval       Skip basic force eval"
    echo "  --include-incomplete    Include incomplete runs"
    exit 1
fi

# Resolve relative input path against submit dir when available, else project root.
if [[ "${INPUT_DIR}" != /* ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/${INPUT_DIR}" ]]; then
        INPUT_DIR="${SLURM_SUBMIT_DIR}/${INPUT_DIR}"
    elif [[ -d "${PROJECT_ROOT}/${INPUT_DIR}" ]]; then
        INPUT_DIR="${PROJECT_ROOT}/${INPUT_DIR}"
    fi
fi
INPUT_DIR="$(cd "${INPUT_DIR}" && pwd -P)"

OUTPUT_ROOT="${PROJECT_ROOT}/local_work/outputs"
mkdir -p "${OUTPUT_ROOT}"

if [[ -n "${RUN_NAME}" ]]; then
    ANALYSIS_LABEL="${RUN_NAME}"
else
    ANALYSIS_LABEL="$(basename "${INPUT_DIR}")"
fi
if [[ "${ANALYSIS_LABEL}" != *_analysis ]]; then
    ANALYSIS_LABEL="${ANALYSIS_LABEL}_analysis"
fi

ANALYSIS_DIR="${OUTPUT_ROOT}/${ANALYSIS_LABEL}"
mkdir -p "${ANALYSIS_DIR}"

ANALYSIS_LOG="${ANALYSIS_DIR}/slurm-analysis-${SLURM_JOB_ID:-local}.out"
exec > >(tee -a "${ANALYSIS_LOG}") 2>&1

# --- Environment setup (shared with training scripts) ---
_found_config="$(find -L "${INPUT_DIR}" -maxdepth 4 -name "config_runtime.yaml" | sort | head -1)"
if [[ -z "${_found_config}" ]]; then
    _found_config="$(find -L "${INPUT_DIR}" -maxdepth 4 -name "config_input.yaml" | sort | head -1)"
fi
if [[ -z "${_found_config}" ]]; then
    echo "ERROR: Could not find config_runtime.yaml or config_input.yaml under ${INPUT_DIR}" >&2
    exit 1
fi
export CONFIG_FILE="${_found_config}"
unset _found_config
source "${SCRIPT_DIR}/slurm_env.sh"

export CUDA_VISIBLE_DEVICES=0

echo "============================================================"
echo "Suite Analysis"
echo "============================================================"
echo "Project root: ${PROJECT_ROOT}"
echo "Input dir:    ${INPUT_DIR}"
[[ -n "${RUN_NAME}" ]] && echo "Run name:     ${RUN_NAME}"
echo "Output dir:   ${ANALYSIS_DIR}"
echo "Extra args:   ${EXTRA_ARGS[*]:-none}"
echo "============================================================"

"${PYTHON_BIN}" -u "${PROJECT_ROOT}/analysis_tests/analyze_suite.py" \
    "${INPUT_DIR}" \
    --analysis-dir "${ANALYSIS_DIR}" \
    --devices-per-run 1 \
    "${EXTRA_ARGS[@]}"
