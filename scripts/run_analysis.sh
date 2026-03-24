#!/bin/bash

#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=develbooster
#SBATCH --output=outputs/slurm-analysis-%j.out

# =============================================================================
# SLURM wrapper for suite analysis.
#
# Usage:
#   sbatch scripts/run_analysis.sh <run_dir> [extra analyze_suite.py args]
# =============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

INPUT_DIR="${1:-}"
if [[ -z "${INPUT_DIR}" ]]; then
    echo "Usage: sbatch scripts/run_analysis.sh <run_dir> [extra args]"
    exit 1
fi
shift || true

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
echo "============================================================"

"${PYTHON_BIN}" -u "${PROJECT_ROOT}/analysis_tests/analyze_suite.py" "${INPUT_DIR}" --detailed-force-eval --devices-per-run 1 "$@"
