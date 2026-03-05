#!/bin/bash -x

#SBATCH --account=cameo
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
#   Multi-protein bucketed training:
#     sbatch scripts/run_training.sh config.yaml --multi-protein-dir /path/to/03_bucketed_npz
#
# =============================================================================

CONFIG_FILE="$1"
shift  # Remove config file from arguments

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Usage: sbatch run_training.sh <config.yaml> [--multi-protein-dir <bucket_dir>] [--resume auto|<checkpoint.pkl>] [extra train.py args]"
    exit 1
fi

# Parse training flags
RESUME_VALUE=""
MULTI_PROTEIN_DIR=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            if [[ -n "$2" ]]; then
                RESUME_VALUE="$2"
                shift 2
            else
                echo "ERROR: --resume requires an argument (auto or checkpoint path)"
                exit 1
            fi
            ;;
        --multi-protein-dir)
            if [[ -n "$2" ]]; then
                MULTI_PROTEIN_DIR="$2"
                shift 2
            else
                echo "ERROR: --multi-protein-dir requires a directory path"
                exit 1
            fi
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                EXTRA_ARGS+=("$1")
                shift
            done
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -n "$MULTI_PROTEIN_DIR" && -n "$RESUME_VALUE" ]]; then
    echo "WARNING: --resume is not supported in --multi-protein-dir mode; ignoring --resume ${RESUME_VALUE}"
    RESUME_VALUE=""
fi

source /p/project1/cameo/schmidt36/load_modules.sh
source /p/project1/cameo/schmidt36/clean_booster_env/bin/activate
# Inject cuequivariance packages via PYTHONPATH rather than activating the overlay
# venv (activating it would replace clean_booster_env and lose jax_sgmc etc.).
# export PYTHONPATH="/p/project1/cameo/schmidt36/cueq_allegro/cueq_overlay_env/lib/python3.12/site-packages:${PYTHONPATH:-}"
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

# ===== JAX Distributed Setup =====
# JAX automatically detects SLURM environment (nodes, process IDs, coordinator)
# No manual coordinator setup needed - jax.distributed.initialize() handles it

# Memory settings for multi-GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

# GPU visibility: defaults to all 4 GPUs; can be overridden externally for scaling tests
# Strip trailing whitespace that SLURM may inject when setting CUDA_VISIBLE_DEVICES automatically
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"

# Force NCCL (GPU-GPU gradient sync) to use InfiniBand, not Ethernet.
# Without this, NCCL may pick the wrong interface and stall during collective ops.
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
if is_truthy "${CHEMTRAIN_PROFILE_NCCL_INFO:-0}"; then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,COLL
else
    unset NCCL_DEBUG_SUBSYS
fi

# JAX distributed can fail or hang when proxy env vars are set in multi-node jobs.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy

# Normal training defaults (no profiling).
# Keep K=8 fixed for this run to match prior microbatch behavior.
export CHEMTRAIN_GRAD_ACCUM_STEPS=1
export CHEMTRAIN_GRAD_ACCUM_MODE="${CHEMTRAIN_GRAD_ACCUM_MODE:-stack_scan}"
export CHEMTRAIN_DEBUG_MICROBATCH_GRAD_NORMS="${CHEMTRAIN_DEBUG_MICROBATCH_GRAD_NORMS:-0}"
export CHEMTRAIN_DISABLE_GRAD_NORM=0
export CHEMTRAIN_DISABLE_TRAIN_TARGET_LOSS_SYNC=0
export CHEMTRAIN_PROFILE_DATALOADER_NEXT=0
export CHEMTRAIN_PROFILE_RANK0_ONLY=1
export CHEMTRAIN_PROFILE_BATCH_BREAKDOWN=0
export CHEMTRAIN_PROFILE_UPDATE_BREAKDOWN=0
export CHEMTRAIN_PROFILE_TASK_TIMING=0
export CHEMTRAIN_PROFILE_BATCH_BREAKDOWN_LIMIT=0
export CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL=0
export CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL_BLOCK=0
export CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL_RANK0_ONLY=1
export CHEMTRAIN_PROFILE_UPDATE_FN_INTERNAL_LIMIT=0
export CHEMTRAIN_PROFILE_GPU_TELEMETRY=0
export CHEMTRAIN_DEBUG_SHAPE_TRACE="${CHEMTRAIN_DEBUG_SHAPE_TRACE:-0}"
export CHEMTRAIN_DEBUG_NEIGHBOR="${CHEMTRAIN_DEBUG_NEIGHBOR:-1}"
export CHEMTRAIN_DEBUG_NEIGHBOR_RANK0_ONLY="${CHEMTRAIN_DEBUG_NEIGHBOR_RANK0_ONLY:-1}"
export CHEMTRAIN_DEBUG_COMPILE_SIGNATURE="${CHEMTRAIN_DEBUG_COMPILE_SIGNATURE:-1}"
export CHEMTRAIN_DEBUG_COMPILE_SIGNATURE_RANK0_ONLY="${CHEMTRAIN_DEBUG_COMPILE_SIGNATURE_RANK0_ONLY:-1}"
export JAX_LOG_COMPILES="${JAX_LOG_COMPILES:-1}"
export CHEMTRAIN_DEBUG_MICROBATCH_GRAD_NORMS=0
export CHEMTRAIN_SEGMENT_SUM_MODE="${CHEMTRAIN_SEGMENT_SUM_MODE:-chunked}"
export CHEMTRAIN_SEGMENT_SUM_CHUNK_EDGES="${CHEMTRAIN_SEGMENT_SUM_CHUNK_EDGES:-65536}"
export CHEMTRAIN_SEGMENT_SUM_DEBUG="${CHEMTRAIN_SEGMENT_SUM_DEBUG:-0}"
# Keep runtime precision in FP32 for non-profiling training unless config
# explicitly overrides inside the trainer.
export CHEMTRAIN_COMPUTE_DTYPE=float32
export CHEMTRAIN_PARAM_DTYPE=float32
export CHEMTRAIN_REDUCE_DTYPE=float32

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
echo "Grad accum K:   $CHEMTRAIN_GRAD_ACCUM_STEPS"
echo "Accum mode:     $CHEMTRAIN_GRAD_ACCUM_MODE"
echo "Shape trace:    $CHEMTRAIN_DEBUG_SHAPE_TRACE"
echo "Nbr debug:      $CHEMTRAIN_DEBUG_NEIGHBOR"
echo "Compile sig:    $CHEMTRAIN_DEBUG_COMPILE_SIGNATURE"
echo "JAX compiles:   $JAX_LOG_COMPILES"
echo "Edge agg mode:  $CHEMTRAIN_SEGMENT_SUM_MODE"
echo "Edge chunk:     $CHEMTRAIN_SEGMENT_SUM_CHUNK_EDGES"
echo "Edge agg debug: $CHEMTRAIN_SEGMENT_SUM_DEBUG"
if [[ -n "${CHEMTRAIN_NEIGHBOR_LIST_FORMAT:-}" ]]; then
    echo "Nbr list fmt:   ${CHEMTRAIN_NEIGHBOR_LIST_FORMAT} (env override)"
fi
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
    echo "JAX coordinator will be selected explicitly (host/port exported to train.py)"
    echo "============================================================"

    # Verify inter-node network reachability before starting training.
    # Fail fast if we cannot find a coordinator address reachable from all nodes.
    COORD_NODE=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -1)
    COORD_PORT=$((29400 + (SLURM_JOB_ID % 1000)))
    # Use a dedicated probe port for TCP reachability checks.
    TCP_PROBE_PORT=$((20000 + (SLURM_JOB_ID % 20000)))

    # Candidate coordinator addresses in priority order.
    # Prefer InfiniBand hostname first (JUWELS convention: append 'i').
    COORD_CANDIDATES=("${COORD_NODE}i.juwels" "${COORD_NODE}.juwels" "${COORD_NODE}")
    COORD_IP_GETENT=$(getent ahostsv4 "${COORD_NODE}i.juwels" 2>/dev/null | awk 'NR==1 {print $1}')
    if [[ -n "${COORD_IP_GETENT}" ]]; then
        COORD_CANDIDATES+=("${COORD_IP_GETENT}")
    fi
    COORD_IP_GETENT=$(getent ahostsv4 "${COORD_NODE}.juwels" 2>/dev/null | awk 'NR==1 {print $1}')
    if [[ -n "${COORD_IP_GETENT}" ]]; then
        COORD_CANDIDATES+=("${COORD_IP_GETENT}")
    fi
    COORD_IP_REMOTE=$(srun --nodes=1 --ntasks=1 -w "${COORD_NODE}" \
        bash -lc "hostname -I | awk '{print \$1}'" 2>/dev/null | tail -n1)
    if [[ -n "${COORD_IP_REMOTE}" ]]; then
        COORD_CANDIDATES+=("${COORD_IP_REMOTE}")
    fi

    check_host_all_nodes_tcp() {
        local host="$1"
        local listener_step_pid=""

        echo "Testing inter-node TCP reachability to candidate coordinator: ${host}:${TCP_PROBE_PORT}"

        # Start a short-lived TCP listener on the coordinator node.
        srun --overlap --nodes=1 --ntasks=1 -w "${COORD_NODE}" bash -lc \
            "python3 -u -c '
import socket
import time
import sys

port = int(\"${TCP_PROBE_PORT}\")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind((\"0.0.0.0\", port))
except Exception as e:
    print(f\"listener bind failed on port {port}: {e}\", flush=True)
    sys.exit(2)
s.listen(16)
print(\"listener ready\", flush=True)
deadline = time.time() + 25
while time.time() < deadline:
    s.settimeout(1.0)
    try:
        conn, _ = s.accept()
        conn.close()
    except socket.timeout:
        pass
s.close()
	'" >/dev/null 2>&1 &
        listener_step_pid=$!

        # Give listener time to bind before client checks.
        sleep 1

        # Verify each node can establish a TCP connection to the candidate host.
        srun --overlap --ntasks-per-node=1 bash -lc \
            "python3 -u -c '
import socket
import sys

host = \"${host}\"
port = int(\"${TCP_PROBE_PORT}\")
try:
    sock = socket.create_connection((host, port), timeout=5.0)
    sock.close()
    print(f\"{socket.gethostname()}: tcp connect to {host}:{port} OK\", flush=True)
except Exception as e:
    print(f\"{socket.gethostname()}: tcp connect to {host}:{port} FAILED ({e})\", flush=True)
    sys.exit(1)
'"
        local rc=$?

        # Ensure background listener is stopped before continuing.
        kill "${listener_step_pid}" >/dev/null 2>&1 || true
        wait "${listener_step_pid}" >/dev/null 2>&1 || true
        return "${rc}"
    }

    # Pick first candidate reachable from all nodes.
    COORD_HOST_SELECTED=""
    declare -A _seen_coord
    for cand in "${COORD_CANDIDATES[@]}"; do
        [[ -z "${cand}" ]] && continue
        if [[ -n "${_seen_coord[$cand]}" ]]; then
            continue
        fi
        _seen_coord[$cand]=1
        if check_host_all_nodes_tcp "${cand}"; then
            COORD_HOST_SELECTED="${cand}"
            break
        fi
    done

    if [[ -z "${COORD_HOST_SELECTED}" ]]; then
        echo "ERROR: No coordinator address is reachable from all nodes."
        echo "Tried candidates: ${COORD_CANDIDATES[*]}"
        exit 1
    fi

    export CHEMTRAIN_COORDINATOR_HOST="${COORD_HOST_SELECTED}"
    export CHEMTRAIN_COORDINATOR_PORT="${COORD_PORT}"
    echo "Selected coordinator: ${CHEMTRAIN_COORDINATOR_HOST}:${CHEMTRAIN_COORDINATOR_PORT}"
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

cd "${CLEAN_CODE_BASE_DIR}"

# Resolve CONFIG_FILE to an absolute path so Python sees the right file
# regardless of the working directory on any remote node.
if [[ "${CONFIG_FILE}" != /* ]]; then
    CONFIG_FILE="${CLEAN_CODE_BASE_DIR}/${CONFIG_FILE}"
fi
if [[ -n "$MULTI_PROTEIN_DIR" && "${MULTI_PROTEIN_DIR}" != /* ]]; then
    MULTI_PROTEIN_DIR="${CLEAN_CODE_BASE_DIR}/${MULTI_PROTEIN_DIR}"
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

# Save the exact input config snapshot before starting Python/JAX init.
PRELAUNCH_CONFIG_COPY="${OUTPUTS_DIR}/config_${SLURM_JOB_ID}_prelaunch.yaml"
cp -f "${CONFIG_FILE}" "${PRELAUNCH_CONFIG_COPY}"
echo "Pre-launch config snapshot: ${PRELAUNCH_CONFIG_COPY}"

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
# Launch 1 process per NODE - chemtrain's pmap handles local multi-GPU
LOGFILE="${OUTPUTS_DIR}/train_allegro_${SLURM_JOB_ID}.log"

echo "============================================================"
echo "Starting training with $SLURM_NNODES node(s), 4 GPUs each..."
echo "Log file: ${LOGFILE}"
if [[ -n "$MULTI_PROTEIN_DIR" ]]; then
    echo "Multi-protein mode: bucket_dir=${MULTI_PROTEIN_DIR}"
fi
if [[ -n "$RESUME_VALUE" ]]; then
    echo "Resume mode: --resume ${RESUME_VALUE}"
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "Extra train.py args: ${EXTRA_ARGS[*]}"
fi
echo "============================================================"

# Build argument list for train.py
TRAIN_ARGS=("$CONFIG_FILE" "${SLURM_JOB_ID}")
if [[ -n "$MULTI_PROTEIN_DIR" ]]; then
    TRAIN_ARGS+=("--multi-protein-dir" "$MULTI_PROTEIN_DIR")
fi
if [[ -n "$RESUME_VALUE" ]]; then
    TRAIN_ARGS+=("--resume" "$RESUME_VALUE")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    TRAIN_ARGS+=("${EXTRA_ARGS[@]}")
fi

# srun launches 1 task per node, each sees 4 local GPUs
srun -l --ntasks-per-node=1 python3 -u "${TRAIN_SCRIPT}" \
    "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOGFILE}"

echo "============================================================"
echo "Training complete. Log: ${LOGFILE}"
echo "============================================================"
