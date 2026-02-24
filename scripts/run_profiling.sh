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

CONFIG_FILE="${1:-config_profile_compare.yaml}"

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Usage: sbatch scripts/run_profiling.sh [config.yaml]"
    echo "  Default config: config_profile_compare.yaml"
    exit 1
fi

source /p/project1/cameo/schmidt36/load_modules.sh
source /p/project1/cameo/schmidt36/clean_booster_env/bin/activate
source /p/project1/cameo/schmidt36/set_lammps_paths.sh

is_truthy() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

export CC=$(which gcc)
export CXX=$(which g++)
export CLANG_CUDA_COMPILER_PATH=$(which gcc)

# CUDA setup for JAX
CUDA_ROOT=$(python -c "import os; from jax_plugins import xla_cuda12; print(os.path.dirname(xla_cuda12.__file__))")
export LD_LIBRARY_PATH=$CUDA_ROOT:$(python -c "import site; print(site.getsitepackages()[0])")/nvidia/cuda_runtime/lib:$(python -c "import site; print(site.getsitepackages()[0])")/nvidia/cublas/lib:$(python -c "import site; print(site.getsitepackages()[0])")/nvidia/cusolver/lib:$LD_LIBRARY_PATH

export CUDA_HOME=/p/software/juwelsbooster/stages/2025/software/CUDA/12
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME --xla_gpu_autotune_level=0"

# Memory settings
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

# GPU visibility
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"

# Force NCCL to use InfiniBand
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN

# ===== Gradient accumulation =====
# Keep K=8 to match production runs (so per-step timings are comparable).
export CHEMTRAIN_GRAD_ACCUM_STEPS="${CHEMTRAIN_GRAD_ACCUM_STEPS:-8}"

# ===== Focused profiling flags =====
# JAX trace export can generate huge protobufs on long/high-event runs.
# Force OFF for full training runs.
export CHEMTRAIN_PROFILE_JAX_TRACE=0

# Layer 1: Per-update detailed timing (forced off for full training).
export CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL=0
export CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL_BLOCK=0

# Profiling rank selection (kept conservative; profiling flags are forced off).
export CHEMTRAIN_PROFILE_RANK0_ONLY=1
export CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL_RANK0_ONLY=1

# Disable detailed per-step internal logs.
export CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL_LIMIT=0

# Layer 2: BatchProfiler gap/barrier ratio (enabled via config YAML).
# No separate env var needed; controlled by profiling.batch_profiler_enabled in YAML.

# Layer 2b: sampled edge-count diagnostics (forced off for full training).
export CHEMTRAIN_PROFILE_EDGE_COUNTS=0
export CHEMTRAIN_PROFILE_EDGE_COUNT_SAMPLES=1
export CHEMTRAIN_PROFILE_EDGE_COUNT_STRIDE=1
export CHEMTRAIN_PROFILE_EDGE_COUNT_RANK0_ONLY=1

# Disable extra instrumentation that inflates overhead/noise for this baseline.
export CHEMTRAIN_PROFILE_TASK_TIMING=0
export CHEMTRAIN_PROFILE_BATCH_BREAKDOWN=0
export CHEMTRAIN_PROFILE_BATCH_BREAKDOWN_LIMIT=0
export CHEMTRAIN_PROFILE_DATALOADER_NEXT=0

# Layer 3: GPU telemetry via nvidia-smi at 1 Hz (forced off for full training).
export CHEMTRAIN_PROFILE_GPU_TELEMETRY=0

# Layer 4: Component-level update timing (forced off for full training).
export CHEMTRAIN_PROFILE_UPDATE_FN_COMPONENTS=0
export CHEMTRAIN_PROFILE_UPDATE_FN_LOCAL_SPLIT=0

# Keep gradient norm and train-target-loss sync (both off = less host overhead,
# but we want them on for profiling so results match production semantics).
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
echo "Config file:    $CONFIG_FILE"
echo "Job ID:         $SLURM_JOB_ID"
echo "Nodes:          $SLURM_NNODES"
echo "GPUs/node:      4 (pmap distributes across local GPUs)"
echo "Total GPUs:     $((SLURM_NNODES * 4))"
echo "Grad accum K:   $CHEMTRAIN_GRAD_ACCUM_STEPS"
echo ""
echo "Profiling flags:"
echo "  UPDATE_FN_INTERNAL:       $CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL"
echo "  UPDATE_FN_INTERNAL_BLOCK: $CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL_BLOCK"
echo "  RANK0_ONLY:               $CHEMTRAIN_PROFILE_RANK0_ONLY"
echo "  TASK_TIMING:              $CHEMTRAIN_PROFILE_TASK_TIMING"
echo "  BATCH_BREAKDOWN:          $CHEMTRAIN_PROFILE_BATCH_BREAKDOWN"
echo "  GPU_TELEMETRY:            $CHEMTRAIN_PROFILE_GPU_TELEMETRY"
echo "  UPDATE_FN_COMPONENTS:     $CHEMTRAIN_PROFILE_UPDATE_FN_COMPONENTS"
echo "  UPDATE_FN_LOCAL_SPLIT:    $CHEMTRAIN_PROFILE_UPDATE_FN_LOCAL_SPLIT"
echo "  EDGE_COUNTS:              $CHEMTRAIN_PROFILE_EDGE_COUNTS"
echo "  EDGE_COUNT_SAMPLES:       $CHEMTRAIN_PROFILE_EDGE_COUNT_SAMPLES"
echo "  EDGE_COUNT_STRIDE:        $CHEMTRAIN_PROFILE_EDGE_COUNT_STRIDE"
echo "  EDGE_COUNT_RANK0_ONLY:    $CHEMTRAIN_PROFILE_EDGE_COUNT_RANK0_ONLY"
echo "  DATALOADER_NEXT:          $CHEMTRAIN_PROFILE_DATALOADER_NEXT"
echo "  JAX_TRACE_EXPORT:         $CHEMTRAIN_PROFILE_JAX_TRACE"
echo "  JAX traces:               profiles_compare/ (config-driven)"
echo "============================================================"

# Print device info from each node
echo "Verifying GPU allocation per node..."
srun --ntasks-per-node=1 bash -c 'echo "Host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L'

# Multi-node: verify connectivity
if [[ $SLURM_NNODES -gt 1 ]]; then
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
    srun --ntasks-per-node=1 bash -c \
        "ping -c 2 -W 5 ${COORD_NODE}.juwels > /dev/null 2>&1 \
         && echo \"\$(hostname): ping to ${COORD_NODE}.juwels OK\" \
         || echo \"WARNING: \$(hostname): ping to ${COORD_NODE}.juwels FAILED\""
    echo "============================================================"
fi

# ===== Determine paths =====
if [[ -n "$SLURM_SUBMIT_DIR" ]]; then
    CLEAN_CODE_BASE_DIR="$SLURM_SUBMIT_DIR"
else
    CLEAN_CODE_BASE_DIR="$(pwd)"
fi

SCRIPT_DIR="${CLEAN_CODE_BASE_DIR}/scripts"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"

echo "Submit directory: ${CLEAN_CODE_BASE_DIR}"
echo "Training script:  ${TRAIN_SCRIPT}"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: Training script not found at ${TRAIN_SCRIPT}"
    echo "Please submit this job from the cameo_cg/ directory"
    exit 1
fi

# ===== Prepare Output Directory =====
OUTPUTS_DIR="${CLEAN_CODE_BASE_DIR}/outputs"
mkdir -p "${OUTPUTS_DIR}"

# ===== GPU Telemetry =====
GPU_TELEMETRY_SRUN_PID=""
if is_truthy "${CHEMTRAIN_PROFILE_GPU_TELEMETRY}"; then
    echo "Starting per-node GPU telemetry sampling (1 Hz)..."
    GPU_TELEMETRY_PREFIX="${OUTPUTS_DIR}/gpu_telemetry_${SLURM_JOB_ID}"
    srun --overlap --ntasks-per-node=1 bash -lc "
        out='${GPU_TELEMETRY_PREFIX}_\$(hostname).csv'
        echo 'timestamp,index,util_gpu,util_mem,power_w,sm_clock_mhz,mem_clock_mhz,pci_gen,pci_width' > \"\$out\"
        while true; do
            nvidia-smi \
                --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current \
                --format=csv,noheader,nounits >> \"\$out\"
            sleep 1
        done
    " &
    GPU_TELEMETRY_SRUN_PID=$!
    echo "GPU telemetry sampler PID: ${GPU_TELEMETRY_SRUN_PID}"
fi

cleanup_background_jobs() {
    if [[ -n "${GPU_TELEMETRY_SRUN_PID}" ]]; then
        kill "${GPU_TELEMETRY_SRUN_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup_background_jobs EXIT

# ===== Run Training =====
LOGFILE="${OUTPUTS_DIR}/train_allegro_${SLURM_JOB_ID}.log"

echo "============================================================"
echo "Starting profiling run with $SLURM_NNODES node(s), 4 GPUs each..."
echo "Log file: ${LOGFILE}"
echo "============================================================"

srun -l --ntasks-per-node=1 python3 -u "${TRAIN_SCRIPT}" \
    "$CONFIG_FILE" "${SLURM_JOB_ID}" 2>&1 | tee "${LOGFILE}"

echo "============================================================"
echo "Profiling run complete."
echo ""
echo "Results:"
echo "  Training log:   ${LOGFILE}"
echo "  SLURM output:   ${OUTPUTS_DIR}/slurm-${SLURM_JOB_ID}.out"
echo "  GPU telemetry:  ${OUTPUTS_DIR}/gpu_telemetry_${SLURM_JOB_ID}_*.csv"
echo "  JAX XLA traces: ${CLEAN_CODE_BASE_DIR}/profiles_compare/"
echo ""
echo "To view JAX traces: open https://ui.perfetto.dev and load the .pb.gz files"
echo "============================================================"
