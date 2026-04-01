#!/bin/bash -x

#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --output=outputs/slurm-%j.out

# =============================================================================
# Profiling SLURM script — focused compute vs communication baseline
# =============================================================================
#
# PURPOSE:
#   Run a short training with focused profiling so we can compare compute vs
#   communication timing with minimal extra instrumentation overhead.
#
# PROFILING LAYERS ENABLED:
#   1. UpdateFnInternal text timing (put_state, put_batch, dispatch, block_loss)
#   2. UpdateFnComponents split (local grad, collectives, optimizer)
#   3. BatchProfiler gap/barrier summary (configured in YAML)
#   4. GPU telemetry (nvidia-smi CSV at 1 Hz)
#
# USAGE:
#   1-node (4 GPUs, default):
#     sbatch scripts/run_profiling.sh config_profile_compare.yaml
#
#   2-node (8 GPUs):
#     sbatch --nodes=2 scripts/run_profiling.sh config_profile_compare.yaml
#
#   Resume is not supported for profiling runs (always starts fresh).
#
# OUTPUTS (in outputs/ and ./profiles_compare/):
#   - outputs/slurm-<JOB_ID>.out          — SLURM log (verification output)
#   - outputs/train_allegro_<JOB_ID>.log  — training log with all timing data
#   - outputs/gpu_telemetry_<JOB_ID>_<host>.csv — GPU utilization 1 Hz samples
#   - profiles_compare/stage_sgd_nesterov_rank<R>_epoch*/ — JAX XLA traces
#
# =============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
export PROJECT_ROOT

CONFIG_FILE="${1:-config_profile_compare.yaml}"

if [[ -z "${CONFIG_FILE}" ]]; then
    echo "Usage: sbatch scripts/run_profiling.sh [config.yaml]"
    echo "  Default config: config_profile_compare.yaml"
    exit 1
fi

if [[ "${CONFIG_FILE}" != /* ]]; then
    if [[ -f "${PROJECT_ROOT}/${CONFIG_FILE}" ]]; then
        CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
    else
        CONFIG_FILE="$(pwd -P)/${CONFIG_FILE}"
    fi
fi

source "${SCRIPT_DIR}/slurm_env.sh"

is_truthy() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

# ===== Gradient accumulation =====
# Keep K=8 to match production runs (so per-step timings are comparable).
export CHEMTRAIN_GRAD_ACCUM_STEPS="${CHEMTRAIN_GRAD_ACCUM_STEPS:-8}"

# ===== Focused profiling flags =====
export CHEMTRAIN_PROFILE_JAX_TRACE=0
export CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL=0
export CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL_BLOCK=0
export CHEMTRAIN_PROFILE_RANK0_ONLY=1
export CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL_RANK0_ONLY=1
export CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL_LIMIT=0
export CHEMTRAIN_PROFILE_EDGE_COUNTS=0
export CHEMTRAIN_PROFILE_EDGE_COUNT_SAMPLES=1
export CHEMTRAIN_PROFILE_EDGE_COUNT_STRIDE=1
export CHEMTRAIN_PROFILE_EDGE_COUNT_RANK0_ONLY=1
export CHEMTRAIN_PROFILE_TASK_TIMING=0
export CHEMTRAIN_PROFILE_BATCH_BREAKDOWN=0
export CHEMTRAIN_PROFILE_BATCH_BREAKDOWN_LIMIT=0
export CHEMTRAIN_PROFILE_DATALOADER_NEXT=0
export CHEMTRAIN_PROFILE_GPU_TELEMETRY=0
export CHEMTRAIN_PROFILE_UPDATE_FN_COMPONENTS=0
export CHEMTRAIN_PROFILE_UPDATE_FN_LOCAL_SPLIT=0
export CHEMTRAIN_DISABLE_GRAD_NORM=0
export CHEMTRAIN_DISABLE_TRAIN_TARGET_LOSS_SYNC=0

# ===== Verification =====
echo "============================================================"
echo "Module Environment"
echo "============================================================"
module list
echo ""
echo "============================================================"
echo "SLURM Profiling Run Configuration"
echo "============================================================"
echo "Config file:    ${CONFIG_FILE}"
echo "Job ID:         ${SLURM_JOB_ID:-local}"
echo "Nodes:          ${SLURM_NNODES:-1}"
echo "GPUs/node:      4 (pmap distributes across local GPUs)"
echo "Total GPUs:     $(( ${SLURM_NNODES:-1} * 4 ))"
echo "Grad accum K:   ${CHEMTRAIN_GRAD_ACCUM_STEPS}"
echo ""
echo "Profiling flags:"
echo "  UPDATE_FN_INTERNAL:       ${CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL}"
echo "  UPDATE_FN_INTERNAL_BLOCK: ${CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL_BLOCK}"
echo "  RANK0_ONLY:               ${CHEMTRAIN_PROFILE_RANK0_ONLY}"
echo "  TASK_TIMING:              ${CHEMTRAIN_PROFILE_TASK_TIMING}"
echo "  BATCH_BREAKDOWN:          ${CHEMTRAIN_PROFILE_BATCH_BREAKDOWN}"
echo "  GPU_TELEMETRY:            ${CHEMTRAIN_PROFILE_GPU_TELEMETRY}"
echo "  UPDATE_FN_COMPONENTS:     ${CHEMTRAIN_PROFILE_UPDATE_FN_COMPONENTS}"
echo "  UPDATE_FN_LOCAL_SPLIT:    ${CHEMTRAIN_PROFILE_UPDATE_FN_LOCAL_SPLIT}"
echo "  EDGE_COUNTS:              ${CHEMTRAIN_PROFILE_EDGE_COUNTS}"
echo "  EDGE_COUNT_SAMPLES:       ${CHEMTRAIN_PROFILE_EDGE_COUNT_SAMPLES}"
echo "  EDGE_COUNT_STRIDE:        ${CHEMTRAIN_PROFILE_EDGE_COUNT_STRIDE}"
echo "  EDGE_COUNT_RANK0_ONLY:    ${CHEMTRAIN_PROFILE_EDGE_COUNT_RANK0_ONLY}"
echo "  DATALOADER_NEXT:          ${CHEMTRAIN_PROFILE_DATALOADER_NEXT}"
echo "  JAX_TRACE_EXPORT:         ${CHEMTRAIN_PROFILE_JAX_TRACE}"
echo "  JAX traces:               profiles_compare/ (config-driven)"
echo "============================================================"

echo "Verifying GPU allocation per node..."
srun --ntasks-per-node=1 bash -c 'echo "Host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L'

if [[ ${SLURM_NNODES:-1} -gt 1 ]]; then
    echo ""
    echo "============================================================"
    echo "Multi-node JAX Distributed Verification"
    echo "============================================================"
    srun --ntasks-per-node=1 bash -c 'echo "Node=$(hostname) SLURM_PROCID=$SLURM_PROCID SLURM_NTASKS=$SLURM_NTASKS"'
    echo "JAX will auto-detect coordinator from SLURM environment"

    COORD_NODE=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -1)
    COORD_PORT=$((29400 + (SLURM_JOB_ID % 1000)))
    echo "Coordinator node: ${COORD_NODE}.juwels  Port: ${COORD_PORT}"
    echo "Testing inter-node ping from each node to coordinator..."
    srun --ntasks-per-node=1 bash -c "ping -c 2 -W 5 ${COORD_NODE}.juwels > /dev/null 2>&1 && echo \"\$(hostname): ping to ${COORD_NODE}.juwels OK\" || echo \"WARNING: \$(hostname): ping to ${COORD_NODE}.juwels FAILED\""
    echo "============================================================"
fi

TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train.py"
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: Training script not found at ${TRAIN_SCRIPT}" >&2
    exit 1
fi

echo "Project root:    ${PROJECT_ROOT}"
echo "Training script: ${TRAIN_SCRIPT}"

OUTPUTS_DIR="${PROJECT_ROOT}/outputs"
mkdir -p "${OUTPUTS_DIR}"

GPU_TELEMETRY_SRUN_PID=""
if is_truthy "${CHEMTRAIN_PROFILE_GPU_TELEMETRY}"; then
    echo "Starting per-node GPU telemetry sampling (1 Hz)..."
    GPU_TELEMETRY_PREFIX="${OUTPUTS_DIR}/gpu_telemetry_${SLURM_JOB_ID:-local}"
    srun --overlap --ntasks-per-node=1 bash -lc "
        out='${GPU_TELEMETRY_PREFIX}_\$(hostname).csv'
        echo 'timestamp,index,util_gpu,util_mem,power_w,sm_clock_mhz,mem_clock_mhz,pci_gen,pci_width' > \"\$out\"
        while true; do
            nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits >> \"\$out\" || true
            sleep 1
        done
    " &
    GPU_TELEMETRY_SRUN_PID=$!
fi

cleanup() {
    local rc=$?
    if [[ -n "${GPU_TELEMETRY_SRUN_PID}" ]]; then
        kill "${GPU_TELEMETRY_SRUN_PID}" >/dev/null 2>&1 || true
    fi
    exit ${rc}
}
trap cleanup EXIT INT TERM

LOGFILE="${OUTPUTS_DIR}/train_allegro_${SLURM_JOB_ID:-local}.log"
srun -u -l --ntasks-per-node=1 "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}" "${CONFIG_FILE}" 2>&1 | tee -a "${LOGFILE}"
