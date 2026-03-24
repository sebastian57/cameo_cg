#!/bin/bash

#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=develbooster
#SBATCH --output=outputs/slurm-analysis-%j.out

# =============================================================================
# SLURM wrapper for suite analysis.
#
# Usage:
#   sbatch scripts/run_analysis.sh --input-dir <run_dir> [--name <label>] \
#       [--detailed-force-eval] [--complete-eval] [extra args]
#
# Examples:
#   sbatch scripts/run_analysis.sh --input-dir outputs/20250101_my_run
#   sbatch scripts/run_analysis.sh --input-dir runs/ --name full_eval \
#       --detailed-force-eval --complete-eval
# =============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

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

# Resolve relative paths
if [[ "${INPUT_DIR}" != /* && -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/${INPUT_DIR}" ]]; then
    INPUT_DIR="${SLURM_SUBMIT_DIR}/${INPUT_DIR}"
fi
INPUT_DIR="$(cd "${INPUT_DIR}" && pwd -P)"

# --- Environment setup (shared with training scripts) ---
export CONFIG_FILE="${INPUT_DIR}/config.yaml"
source "${SCRIPT_DIR}/slurm_env.sh"

export CUDA_VISIBLE_DEVICES=0

echo "============================================================"
echo "Suite Analysis"
echo "============================================================"
echo "Input dir:  ${INPUT_DIR}"
[[ -n "${RUN_NAME}" ]] && echo "Run name:   ${RUN_NAME}"
echo "Extra args: ${EXTRA_ARGS[*]:-none}"
echo "============================================================"

"${PYTHON_BIN}" -u "${PROJECT_ROOT}/analysis_tests/analyze_suite.py" \
    "${INPUT_DIR}" \
    --devices-per-run 1 \
    "${EXTRA_ARGS[@]}"
