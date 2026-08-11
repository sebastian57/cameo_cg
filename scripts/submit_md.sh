#!/bin/bash
# SLURM submission script for CAMEO CG JAX-MD simulations.
#
# Usage:
#   sbatch scripts/submit_md.sh [md_config.yaml]
#   sbatch scripts/submit_md.sh configs/md_1pro_4zoh.yaml
#
#SBATCH --job-name=cameo_md
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/cameo_md_%j.out
#SBATCH --error=slurm/cameo_md_%j.err

set -euo pipefail

CONFIG="${1:-configs/example_md.yaml}"
PROJECT_ROOT="${CAMEO_CG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${PROJECT_ROOT}/${CONFIG}"
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: MD config not found: ${CONFIG}" >&2
  exit 1
fi
export CONFIG_FILE="${CONFIG}"
source "${PROJECT_ROOT}/scripts/slurm_env.sh"

echo "============================================================"
echo "CAMEO CG JAX-MD Simulation"
echo "Config      : $CONFIG"
echo "Project root: $PROJECT_ROOT"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "============================================================"

mkdir -p "$PROJECT_ROOT/slurm"
cd "$PROJECT_ROOT"

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

source "${PROJECT_ROOT}/runs/registry_hook.sh"
run_registry_start md "${CONFIG}"
run_registry_install_exit_trap

"${PYTHON_BIN}" scripts/run_md.py "$CONFIG" "$SLURM_JOB_ID"
