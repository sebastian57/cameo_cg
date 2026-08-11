#!/bin/bash
# SLURM submission script for parallel multi-replica CAMEO CG MD.
#
# Runs n_replicas replicas in GPU-sized waves using scripts/run_md_parallel.py.
# Use PROCS_PER_GPU to oversubscribe small systems; default 4 on 4 GPUs = 16
# concurrent replicas per wave.
#
# Usage:
#   sbatch scripts/submit_md_parallel.sh configs/my_md.yaml
#   sbatch --gres=gpu:8 scripts/submit_md_parallel.sh configs/my_md.yaml
#
#SBATCH --job-name=cameo_md_par
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --output=slurm/cameo_md_par_%j.out
#SBATCH --error=slurm/cameo_md_par_%j.err

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
echo "CAMEO CG JAX-MD Parallel MD"
echo "Config      : $CONFIG"
echo "Project root: $PROJECT_ROOT"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"

mkdir -p "$PROJECT_ROOT/slurm"
cd "$PROJECT_ROOT"

# Multiple tiny-replica JAX processes share each GPU; avoid one process
# preallocating most memory before its siblings start.
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.20}

# Count allocated GPUs from SLURM
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
PROCS_PER_GPU=${PROCS_PER_GPU:-4}
WAVE_SIZE=$((N_GPUS * PROCS_PER_GPU))
echo "Launching MD replicas with $N_GPUS GPU(s), $PROCS_PER_GPU process(es)/GPU, wave size $WAVE_SIZE"
echo "XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION"

source "${PROJECT_ROOT}/runs/registry_hook.sh"
run_registry_start md "${CONFIG}"
run_registry_install_exit_trap

"${PYTHON_BIN}" scripts/run_md_parallel.py "$CONFIG" \
    --n-gpus "$N_GPUS" \
    --procs-per-gpu "$PROCS_PER_GPU" \
    --job-id "$SLURM_JOB_ID"
