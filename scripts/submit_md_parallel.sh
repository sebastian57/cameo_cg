#!/bin/bash
# SLURM submission script for parallel multi-replica CAMEO CG MD.
#
# Runs n_replicas replicas simultaneously, one per GPU, using
# scripts/run_md_parallel.py.  Set n_replicas in the YAML to match
# the GPU count requested below (--gres=gpu:N).
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

CONFIG="${1:-configs/md_1pro_4zoh.yaml}"
PROJECT_ROOT="${CAMEO_CG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

echo "============================================================"
echo "CAMEO CG JAX-MD Parallel MD"
echo "Config      : $CONFIG"
echo "Project root: $PROJECT_ROOT"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

source "$PROJECT_ROOT/../load_modules.sh"
source "$PROJECT_ROOT/../venv_cameocg_jupiter/bin/activate"

mkdir -p "$PROJECT_ROOT/slurm"
cd "$PROJECT_ROOT"

# Allocate most GPU memory to JAX upfront; leave some headroom for cuEquivariance.
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

# Count allocated GPUs from SLURM
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
echo "Launching $N_GPUS replicas..."

python scripts/run_md_parallel.py "$CONFIG" \
    --n-gpus "$N_GPUS" \
    --job-id "$SLURM_JOB_ID"
