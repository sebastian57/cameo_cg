#!/bin/bash -x

#SBATCH --account=atmlaml #cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=booster
#SBATCH --output=outputs/slurm-%j.out

# =============================================================================
# Unified SLURM script for Allegro training (works for 1 or N nodes)
# =============================================================================
#
# IMPORTANT: Submit this job from the clean_code_base/ directory
#
# ARCHITECTURE:
#   - 1 process per NODE (not per GPU!)
#   - Each process sees 4 local GPUs
#   - chemtrain uses pmap internally to distribute across local GPUs
#   - JAX distributed coordinates gradient sync across NODES
#
# Memory model:
#   - Data loaded ONCE per node (not per GPU)
#   - pmap splits batches across 4 local GPUs
#   - For 2 nodes: 2 processes, each with 4 GPUs = 8 total GPUs
#
# Usage:
#   Single-node (1 node, 4 GPUs):
#     sbatch scripts/run_training.sh config.yaml
#
#   Multi-node (2 nodes, 8 GPUs):
#     sbatch --nodes=2 scripts/run_training.sh config.yaml
#
#   Resume from latest checkpoint:
#     sbatch scripts/run_training.sh config.yaml --resume auto
#
#   Resume from specific checkpoint:
#     sbatch scripts/run_training.sh config.yaml --resume ./checkpoints_allegro/epoch30.pkl
#
# =============================================================================

CONFIG_FILE="$1"
shift  # Remove config file from arguments

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Usage: sbatch run_training.sh <config.yaml> [--resume auto|<checkpoint.pkl>]"
    exit 1
fi

# Parse --resume flag
RESUME_FLAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            if [[ -n "$2" ]]; then
                RESUME_FLAG="--resume $2"
                shift 2
            else
                echo "ERROR: --resume requires an argument (auto or checkpoint path)"
                exit 1
            fi
            ;;
        *)
            echo "WARNING: Unknown argument: $1"
            shift
            ;;
    esac
done

source /p/project1/cameo/schmidt36/load_modules.sh
source /p/project1/cameo/schmidt36/clean_booster_env/bin/activate
source /p/project1/cameo/schmidt36/set_lammps_paths.sh

export CC=$(which gcc)
export CXX=$(which g++)
export CLANG_CUDA_COMPILER_PATH=$(which gcc)

# CUDA setup for JAX
CUDA_ROOT=$(python -c "import os; from jax_plugins import xla_cuda12; print(os.path.dirname(xla_cuda12.__file__))")
export LD_LIBRARY_PATH=$CUDA_ROOT:$(python -c "import site; print(site.getsitepackages()[0])")/nvidia/cuda_runtime/lib:$(python -c "import site; print(site.getsitepackages()[0])")/nvidia/cublas/lib:$(python -c "import site; print(site.getsitepackages()[0])")/nvidia/cusolver/lib:$LD_LIBRARY_PATH

export CUDA_HOME=/p/software/juwelsbooster/stages/2025/software/CUDA/12
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME --xla_gpu_autotune_level=0"

# ===== JAX Distributed Setup =====
# JAX automatically detects SLURM environment (nodes, process IDs, coordinator)
# No manual coordinator setup needed - jax.distributed.initialize() handles it

# Memory settings for multi-GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

# GPU visibility: defaults to all 4 GPUs; can be overridden externally for scaling tests
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# ===== Verification =====
echo "============================================================"
echo "Module Environment"
echo "============================================================"
module list
echo ""
echo "============================================================"
echo "SLURM Job Configuration"
echo "============================================================"
echo "Config file:    $CONFIG_FILE"
echo "Job ID:         $SLURM_JOB_ID"
echo "Nodes:          $SLURM_NNODES"
echo "Tasks/node:     1 (1 process per node)"
echo "GPUs per node:  4 (pmap distributes across local GPUs)"
echo "Total GPUs:     $((SLURM_NNODES * 4))"
echo "CUDA_HOME:      $CUDA_HOME"
echo "CUDA_VISIBLE:   $CUDA_VISIBLE_DEVICES"
echo "============================================================"

# Print device info from each node
echo "Verifying GPU allocation per node..."
srun --ntasks-per-node=1 bash -c 'echo "Host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L'

# For multi-node: verify SLURM process IDs (JAX auto-detects coordinator)
if [[ $SLURM_NNODES -gt 1 ]]; then
    echo ""
    echo "============================================================"
    echo "Multi-node JAX Distributed Verification"
    echo "============================================================"
    srun --ntasks-per-node=1 bash -c 'echo "Node=$(hostname) SLURM_PROCID=$SLURM_PROCID SLURM_NTASKS=$SLURM_NTASKS"'
    echo "JAX will auto-detect coordinator from SLURM environment"
    echo "============================================================"
fi

# ===== Determine paths =====
# Use SLURM_SUBMIT_DIR (directory from which job was submitted)
# Assumes you submit from clean_code_base/ directory
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    CLEAN_CODE_BASE_DIR="$SLURM_SUBMIT_DIR"
else
    # Fallback: use current directory
    CLEAN_CODE_BASE_DIR="$(pwd)"
fi

SCRIPT_DIR="${CLEAN_CODE_BASE_DIR}/scripts"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"

echo "Submit directory: ${CLEAN_CODE_BASE_DIR}"
echo "Training script:  ${TRAIN_SCRIPT}"

# Verify script exists
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: Training script not found at ${TRAIN_SCRIPT}"
    echo "Please submit this job from the clean_code_base/ directory"
    exit 1
fi

# ===== Prepare Output Directory =====
# Create outputs directory for logs (relative to submit directory)
OUTPUTS_DIR="${CLEAN_CODE_BASE_DIR}/outputs"
mkdir -p "${OUTPUTS_DIR}"

# ===== Run Training =====
# Launch 1 process per NODE - chemtrain's pmap handles local multi-GPU
LOGFILE="${OUTPUTS_DIR}/train_allegro_${SLURM_JOB_ID}.log"

echo "============================================================"
echo "Starting training with $SLURM_NNODES node(s), 4 GPUs each..."
echo "Log file: ${LOGFILE}"
if [[ -n "$RESUME_FLAG" ]]; then
    echo "Resume mode: ${RESUME_FLAG}"
fi
echo "============================================================"

# srun launches 1 task per node, each sees 4 local GPUs
# shellcheck disable=SC2086  # Word splitting intended for RESUME_FLAG
srun -l --ntasks-per-node=1 python3 -u "${TRAIN_SCRIPT}" \
    "$CONFIG_FILE" "${SLURM_JOB_ID}" ${RESUME_FLAG} 2>&1 | tee "${LOGFILE}"

echo "============================================================"
echo "Training complete. Log: ${LOGFILE}"
echo "============================================================"
