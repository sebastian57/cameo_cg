#!/bin/bash

#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --partition=booster
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# =============================================================================
# Unified SLURM script for training (single run or array suite)
# =============================================================================
#
# ARCHITECTURE:
#   - 1 process per NODE (not per GPU!)
#   - Each process sees 4 local GPUs
#   - chemtrain uses pmap internally to distribute across local GPUs
#   - JAX distributed coordinates gradient sync across NODES
#
# Usage:
#   Single-run:
#     sbatch scripts/run_training.sh config.yaml
#
#   Multi-node:
#     sbatch --nodes=2 scripts/run_training.sh config.yaml
#
#   Resume:
#     sbatch scripts/run_training.sh config.yaml --resume auto
#
#   Multi-protein:
#     sbatch scripts/run_training.sh config.yaml --multi-protein-dir /path/to/npz
#
#   Array mode (submitted by scripts/submit_suite.sh):
#     sbatch --array 0-N%4 \
#       --export=ALL,CONFIG_LIST_FILE=manifest.txt,PARENT_OUTPUT_DIR=... \
#       scripts/run_training.sh
#
# =============================================================================

set -Eeuo pipefail

RUN_TRAINING_TRACE="${RUN_TRAINING_TRACE:-0}"
if [[ "${RUN_TRAINING_TRACE}" == "1" ]]; then
    export PS4='[run_training.sh:${LINENO}] '
    set -x
fi

run_training_err_trap() {
    local rc=$?
    echo "[run_training.sh] ERROR rc=${rc} line=${BASH_LINENO[0]} cmd=${BASH_COMMAND}" >&2
    exit "${rc}"
}
trap run_training_err_trap ERR

SCRIPT_DIR=""
PROJECT_ROOT="${CAMEO_CG_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
if [[ -n "${CAMEO_CG_PROJECT_ROOT:-}" ]]; then
    SCRIPT_DIR="${PROJECT_ROOT}/scripts"
fi

is_truthy() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

resolve_repo_root() {
    local hint=""
    local cand=""
    for hint in "$@"; do
        [[ -z "${hint}" ]] && continue
        if [[ -f "${hint}/scripts/slurm_env.sh" && -f "${hint}/scripts/train.py" ]]; then
            printf '%s\n' "$(cd "${hint}" && pwd -P)"
            return 0
        fi
        cand="${hint}"
        if [[ -f "${cand}" ]]; then
            cand="$(dirname "${cand}")"
        fi
        while [[ "${cand}" != "/" ]]; do
            if [[ -f "${cand}/scripts/slurm_env.sh" && -f "${cand}/scripts/train.py" ]]; then
                printf '%s\n' "$(cd "${cand}" && pwd -P)"
                return 0
            fi
            cand="$(dirname "${cand}")"
        done
    done
    return 1
}

# =============================================================================
# Parse arguments
# =============================================================================
CONFIG_FILE="${1:-}"
if [[ -n "${CONFIG_FILE}" ]]; then
    shift || true
fi

# Array mode: pick config from manifest
if [[ -z "${CONFIG_FILE}" && -n "${CONFIG_LIST_FILE:-}" ]]; then
    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        echo "ERROR: CONFIG_LIST_FILE provided but SLURM_ARRAY_TASK_ID is not set"
        exit 1
    fi
    CONFIG_FILE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${CONFIG_LIST_FILE}")"
fi

if [[ -z "${CONFIG_FILE}" ]]; then
    echo "Usage: sbatch scripts/run_training.sh <config.yaml> [--resume auto|<path>] [--multi-protein-dir <dir>]"
    exit 1
fi

RESUME_VALUE=""
MULTI_PROTEIN_DIR=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            RESUME_VALUE="${2:?--resume requires an argument}"; shift 2 ;;
        --multi-protein-dir)
            MULTI_PROTEIN_DIR="${2:?--multi-protein-dir requires a directory}"; shift 2 ;;
        --)
            shift; EXTRA_ARGS+=("$@"); break ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -n "$MULTI_PROTEIN_DIR" && -n "$RESUME_VALUE" ]]; then
    echo "WARNING: --resume not supported with --multi-protein-dir; ignoring"
    RESUME_VALUE=""
fi

# Resolve config to absolute path
if [[ "${CONFIG_FILE}" != /* ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/${CONFIG_FILE}" ]]; then
        CONFIG_FILE="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)/${CONFIG_FILE}"
    elif [[ -f "${PROJECT_ROOT}/${CONFIG_FILE}" ]]; then
        CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
    else
        CONFIG_FILE="$(pwd -P)/${CONFIG_FILE}"
    fi
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi
CONFIG_DIR="$(cd "$(dirname "${CONFIG_FILE}")" && pwd -P)"

PROJECT_ROOT="$(resolve_repo_root "${CONFIG_DIR}" "${SLURM_SUBMIT_DIR:-}" "${PROJECT_ROOT:-}" "$(pwd -P)" "$(dirname "${BASH_SOURCE[0]}")")" || {
    echo "ERROR: Could not locate project root containing scripts/slurm_env.sh and scripts/train.py"
    exit 1
}
SCRIPT_DIR="${PROJECT_ROOT}/scripts"

# Ensure a Python interpreter exists before loading the selected SLURM env.
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "ERROR: Could not locate python/python3 in PATH."
    exit 1
fi

# Resolve multi-protein dir
if [[ -n "$MULTI_PROTEIN_DIR" && "${MULTI_PROTEIN_DIR}" != /* ]]; then
    MULTI_PROTEIN_DIR="${PROJECT_ROOT}/${MULTI_PROTEIN_DIR}"
fi

# Auto-enable bucketed multi-protein mode only when explicitly requested
# in config (data.use_bucketed_dir: true) and data.path resolves to a directory.
if [[ -z "${MULTI_PROTEIN_DIR}" ]]; then
    AUTO_MULTI_DIR="$(${PYTHON_BIN} - <<'PYAUTO' "${CONFIG_FILE}" "${PROJECT_ROOT}"
from pathlib import Path
import sys
import yaml

cfg = Path(sys.argv[1]).resolve()
project_root = Path(sys.argv[2]).resolve()
try:
    data = yaml.safe_load(cfg.read_text()) or {}
except Exception:
    data = {}
data_cfg = data.get('data') or {}
flag_raw = data_cfg.get('use_bucketed_dir', False)
if isinstance(flag_raw, str):
    bucketed_mode = flag_raw.strip().lower() in {'1', 'true', 'yes', 'on'}
else:
    bucketed_mode = bool(flag_raw)
if not bucketed_mode:
    print('')
    raise SystemExit(0)
raw = data_cfg.get('path')
if not raw:
    print('__BUCKET_FLAG_SET_BUT_NO_PATH__')
    raise SystemExit(0)
p = Path(str(raw))
if p.is_absolute():
    cand = p
else:
    candidates = [cfg.parent / p, project_root / p]
    cand = None
    for c in candidates:
        if c.exists():
            cand = c
            break
    if cand is None:
        cand = project_root / p
if cand.exists() and cand.is_dir():
    print(str(cand.resolve()))
else:
    print('__BUCKET_FLAG_SET_BUT_NOT_DIR__')
PYAUTO
)"
    if [[ "${AUTO_MULTI_DIR}" == "__BUCKET_FLAG_SET_BUT_NO_PATH__" ]]; then
        echo "WARNING: data.use_bucketed_dir=true but data.path is empty; using normal DatasetLoader mode."
        AUTO_MULTI_DIR=""
    elif [[ "${AUTO_MULTI_DIR}" == "__BUCKET_FLAG_SET_BUT_NOT_DIR__" ]]; then
        echo "WARNING: data.use_bucketed_dir=true but data.path is not a directory; using normal DatasetLoader mode."
        AUTO_MULTI_DIR=""
    fi
    if [[ -n "${AUTO_MULTI_DIR}" ]]; then
        MULTI_PROTEIN_DIR="${AUTO_MULTI_DIR}"
        echo "[run_training.sh] Auto-enabled bucketed --multi-protein-dir from config (data.use_bucketed_dir=true): ${MULTI_PROTEIN_DIR}"
    fi
fi

if [[ -n "$MULTI_PROTEIN_DIR" && -n "$RESUME_VALUE" ]]; then
    echo "WARNING: --resume not supported with --multi-protein-dir; ignoring"
    RESUME_VALUE=""
fi

# =============================================================================
# Output directory setup
# =============================================================================
JOB_TAG="${SLURM_JOB_ID:-local}"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    JOB_TAG="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi

RUN_NAME="${RUN_NAME:-$(basename "${CONFIG_FILE%.*}")}"
DATE_TAG="$(date +%Y%m%d)"

if [[ -n "${PARENT_OUTPUT_DIR:-}" ]]; then
    # Suite mode: outputs under the parent suite directory
    OUTPUTS_ROOT="${PARENT_OUTPUT_DIR}"
    RUN_OUTPUT_DIR="${OUTPUTS_ROOT}/${RUN_NAME}"
else
    # Single-run mode:
    # 1) If paths.output_dir is set in config, use it (absolute or repo-root-relative).
    # 2) Otherwise default to local_work/outputs/<date>_<run_name>.
    CONFIG_OUTPUT_DIR="$("${PYTHON_BIN}" -c "
import sys, yaml
try:
    d = yaml.safe_load(open(sys.argv[1])) or {}
except Exception:
    d = {}
v = ((d.get('paths') or {}).get('output_dir'))
print('' if v is None else str(v).strip())
" "${CONFIG_FILE}" 2>/dev/null || echo '')"
    if [[ -n "${CONFIG_OUTPUT_DIR}" ]]; then
        if [[ "${CONFIG_OUTPUT_DIR}" == /* ]]; then
            RUN_OUTPUT_DIR="${CONFIG_OUTPUT_DIR}"
        else
            RUN_OUTPUT_DIR="${PROJECT_ROOT}/${CONFIG_OUTPUT_DIR}"
        fi
    else
        OUTPUTS_ROOT="${PROJECT_ROOT}/local_work/outputs"
        RUN_OUTPUT_DIR="${OUTPUTS_ROOT}/${DATE_TAG}_${RUN_NAME}"
    fi
fi

mkdir -p "${RUN_OUTPUT_DIR}"
RUN_OUTPUT_DIR="$(cd "${RUN_OUTPUT_DIR}" && pwd -P)"
RUN_EXPORT_DIR="${RUN_OUTPUT_DIR}/exports"
RUN_CHECKPOINT_DIR="${RUN_OUTPUT_DIR}/checkpoints"
RUN_PROFILE_DIR="${RUN_OUTPUT_DIR}/profiles"
mkdir -p "${RUN_EXPORT_DIR}" "${RUN_CHECKPOINT_DIR}" "${RUN_PROFILE_DIR}"

RUN_SLURM_OUT="${RUN_OUTPUT_DIR}/slurm-${JOB_TAG}.out"
RUN_SLURM_ERR="${RUN_OUTPUT_DIR}/slurm-${JOB_TAG}.err"
touch "${RUN_SLURM_OUT}" "${RUN_SLURM_ERR}"
exec > >(tee -a "${RUN_SLURM_OUT}") 2> >(tee -a "${RUN_SLURM_ERR}" >&2)

echo "[run_training.sh] Project root: ${PROJECT_ROOT}"
echo "[run_training.sh] Script dir:   ${SCRIPT_DIR}"
echo "[run_training.sh] Config file:  ${CONFIG_FILE}"
echo "[run_training.sh] Run dir:      ${RUN_OUTPUT_DIR}"

# =============================================================================
# Environment setup (shared helper)
# =============================================================================
source "${SCRIPT_DIR}/slurm_env.sh"

# Runtime config: inject resolved output paths into a copy of the config.
# Use job-unique filenames to avoid cross-job overwrite when multiple configs
# share the same paths.output_dir.
INPUT_CONFIG_COPY="${RUN_OUTPUT_DIR}/config_input_${JOB_TAG}.yaml"
RUNTIME_CONFIG="${RUN_OUTPUT_DIR}/config_runtime_${JOB_TAG}.yaml"
cp -f "${CONFIG_FILE}" "${INPUT_CONFIG_COPY}"
# Backward-compat alias used by older analysis utilities.
ln -sfn "$(basename "${INPUT_CONFIG_COPY}")" "${RUN_OUTPUT_DIR}/config_input.yaml"

"${PYTHON_BIN}" - <<'PYCFG' "${CONFIG_FILE}" "${RUNTIME_CONFIG}" "${RUN_EXPORT_DIR}" "${RUN_CHECKPOINT_DIR}" "${RUN_PROFILE_DIR}" "${PROJECT_ROOT}"
from pathlib import Path
import sys, yaml

src = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
export_dir = str(Path(sys.argv[3]).resolve())
checkpoint_dir = str(Path(sys.argv[4]).resolve())
profile_dir = str(Path(sys.argv[5]).resolve())
project_root = Path(sys.argv[6]).resolve()

data = yaml.safe_load(src.read_text()) or {}

paths = data.setdefault('paths', {})
paths['output_dir'] = str(Path(sys.argv[3]).resolve().parent)
paths['export_dir'] = export_dir
paths['checkpoint_dir'] = checkpoint_dir
paths['profile_dir'] = profile_dir
paths['slurm_dir'] = str(out.parent)

# Also set legacy keys for backward compat with older code paths
training = data.setdefault('training', {})
training['export_path'] = export_dir
training['checkpoint_path'] = checkpoint_dir
profiling = training.setdefault('profiling', {})
profiling['trace_dir'] = profile_dir

# Resolve relative input paths to absolute
model = data.setdefault('model', {})
priors = model.get('priors') or {}
spline_file = priors.get('spline_file')
if spline_file:
    sp = Path(spline_file)
    if not sp.is_absolute():
        for candidate in [src.parent / sp, project_root / sp]:
            if candidate.exists():
                priors['spline_file'] = str(candidate.resolve())
                break
        else:
            priors['spline_file'] = str((project_root / sp).resolve())
    model['priors'] = priors

data_path = ((data.get('data') or {}).get('path'))
if data_path:
    dp = Path(data_path)
    if not dp.is_absolute():
        for candidate in [src.parent / dp, project_root / dp]:
            if candidate.exists():
                data['data']['path'] = str(candidate.resolve())
                break
        else:
            data['data']['path'] = str((project_root / dp).resolve())

out.write_text(yaml.safe_dump(data, sort_keys=False))
PYCFG
# Backward-compat alias used by older analysis utilities.
ln -sfn "$(basename "${RUNTIME_CONFIG}")" "${RUN_OUTPUT_DIR}/config_runtime.yaml"

LOGFILE="${RUN_OUTPUT_DIR}/train_${JOB_TAG}.log"

# =============================================================================
# Read grad-accum settings from config (env vars win if set externally)
# =============================================================================
CONFIG_GRAD_ACCUM_STEPS="$("${PYTHON_BIN}" -c "
import yaml, sys
d = yaml.safe_load(open(sys.argv[1])) or {}
v = (d.get('training') or {}).get('grad_accum_steps')
print('' if v is None else v)
" "${CONFIG_FILE}" 2>/dev/null || echo '')"
CONFIG_GRAD_ACCUM_MODE="$("${PYTHON_BIN}" -c "
import yaml, sys
d = yaml.safe_load(open(sys.argv[1])) or {}
v = (d.get('training') or {}).get('grad_accum_mode')
print('' if v is None else v)
" "${CONFIG_FILE}" 2>/dev/null || echo '')"
export CHEMTRAIN_GRAD_ACCUM_STEPS="${CHEMTRAIN_GRAD_ACCUM_STEPS:-${CONFIG_GRAD_ACCUM_STEPS:-6}}"
export CHEMTRAIN_GRAD_ACCUM_MODE="${CHEMTRAIN_GRAD_ACCUM_MODE:-${CONFIG_GRAD_ACCUM_MODE:-stack_scan}}"

# --- Debug flags consumed by cameo_cg Python code (config.debug.* is primary) ---
export CHEMTRAIN_DEBUG_SHAPE_TRACE="${CHEMTRAIN_DEBUG_SHAPE_TRACE:-0}"
export CHEMTRAIN_DEBUG_NEIGHBOR="${CHEMTRAIN_DEBUG_NEIGHBOR:-0}"
export CHEMTRAIN_DEBUG_NEIGHBOR_RANK0_ONLY="${CHEMTRAIN_DEBUG_NEIGHBOR_RANK0_ONLY:-1}"

# --- Chemtrain-internal passthrough (not read by cameo_cg code) ---
export CHEMTRAIN_DEBUG_MICROBATCH_GRAD_NORMS="${CHEMTRAIN_DEBUG_MICROBATCH_GRAD_NORMS:-0}"
export CHEMTRAIN_DISABLE_GRAD_NORM="${CHEMTRAIN_DISABLE_GRAD_NORM:-0}"
export CHEMTRAIN_DISABLE_TRAIN_TARGET_LOSS_SYNC="${CHEMTRAIN_DISABLE_TRAIN_TARGET_LOSS_SYNC:-0}"
export CHEMTRAIN_SEGMENT_SUM_MODE="${CHEMTRAIN_SEGMENT_SUM_MODE:-chunked}"
export CHEMTRAIN_SEGMENT_SUM_CHUNK_EDGES="${CHEMTRAIN_SEGMENT_SUM_CHUNK_EDGES:-65536}"
export CHEMTRAIN_SEGMENT_SUM_DEBUG="${CHEMTRAIN_SEGMENT_SUM_DEBUG:-0}"

# =============================================================================
# Multi-node coordinator verification
# =============================================================================
if [[ ${SLURM_NNODES:-1} -gt 1 ]]; then
    echo "============================================================"
    echo "Multi-node JAX Distributed Verification"
    echo "============================================================"
    srun --ntasks-per-node=1 bash -c 'echo "Node=$(hostname) SLURM_PROCID=$SLURM_PROCID SLURM_NTASKS=$SLURM_NTASKS"'

    # Check if coordinator is manually set via environment variable
    if [[ -n "${CHEMTRAIN_COORDINATOR_HOST:-}" ]]; then
        echo "Using pre-set coordinator from CHEMTRAIN_COORDINATOR_HOST: ${CHEMTRAIN_COORDINATOR_HOST}"
        export CHEMTRAIN_COORDINATOR_PORT="${CHEMTRAIN_COORDINATOR_PORT:-$((29400 + (SLURM_JOB_ID % 1000)))}"
    else
        COORD_NODE=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -1)
        COORD_PORT=$((29400 + (SLURM_JOB_ID % 1000)))
        TCP_PROBE_PORT=$((20000 + (SLURM_JOB_ID % 20000)))
    
        COORD_CANDIDATES=("${COORD_NODE}")
        
        # Try DNS suffixes for this cluster (adjust if needed for your site)
        for host_suffix in "${COORD_NODE}i.juwels" "${COORD_NODE}.juwels"; do
            COORD_IP_GETENT="$(getent ahostsv4 "${host_suffix}" 2>/dev/null | awk 'NR==1 {print $1}' || true)"
            if [[ -n "${COORD_IP_GETENT}" ]]; then
                COORD_CANDIDATES+=("${host_suffix}" "${COORD_IP_GETENT}")
                break  # If DNS works, use that
            fi
        done
        
        # Fallback: get IP directly from coordinator node via srun
        COORD_IP_REMOTE=$(srun --nodes=1 --ntasks=1 -w "${COORD_NODE}" \
            bash -lc "hostname -I | awk '{print \$1}'" 2>/dev/null | tail -n1 || echo "")
        if [[ -n "${COORD_IP_REMOTE}" ]]; then
            COORD_CANDIDATES+=("${COORD_IP_REMOTE}")
        fi

        check_host_all_nodes_tcp() {
            local host="$1"
            echo "Testing TCP reachability to ${host}:${TCP_PROBE_PORT}"
            srun --overlap --nodes=1 --ntasks=1 -w "${COORD_NODE}" bash -lc "
                python3 -u -c '
import socket, time, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((\"0.0.0.0\", ${TCP_PROBE_PORT}))
s.listen(16)
deadline = time.time() + 25
while time.time() < deadline:
    s.settimeout(1.0)
    try:
        conn, _ = s.accept(); conn.close()
    except socket.timeout: pass
s.close()
'" >/dev/null 2>&1 &
            local listener_pid=$!
            sleep 1
            srun --overlap --ntasks-per-node=1 bash -lc "
                python3 -u -c '
import socket, sys
try:
    sock = socket.create_connection((\"${host}\", ${TCP_PROBE_PORT}), timeout=5.0)
    sock.close()
except Exception as e:
    sys.exit(1)
'"
            local rc=$?
            kill "${listener_pid}" >/dev/null 2>&1 || true
            wait "${listener_pid}" >/dev/null 2>&1 || true
            return "${rc}"
        }

        COORD_HOST_SELECTED=""
        declare -A _seen_coord
        for cand in "${COORD_CANDIDATES[@]}"; do
            [[ -z "${cand}" || -n "${_seen_coord[$cand]:-}" ]] && continue
            _seen_coord[$cand]=1
            if check_host_all_nodes_tcp "${cand}"; then
                COORD_HOST_SELECTED="${cand}"
                break
            fi
        done
        if [[ -z "${COORD_HOST_SELECTED}" ]]; then
            echo "ERROR: No coordinator address reachable from all nodes."
            echo "Debug info:"
            echo "  COORD_NODE: ${COORD_NODE}"
            echo "  Candidates tried: ${COORD_CANDIDATES[@]}"
            echo "  SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
            echo "Troubleshooting: Check DNS setup or try setting CHEMTRAIN_COORDINATOR_HOST manually"
            exit 1
        fi
        export CHEMTRAIN_COORDINATOR_HOST="${COORD_HOST_SELECTED}"
        export CHEMTRAIN_COORDINATOR_PORT="${COORD_PORT}"
        echo "Selected coordinator: ${CHEMTRAIN_COORDINATOR_HOST}:${CHEMTRAIN_COORDINATOR_PORT}"
    fi
    echo "============================================================"
fi

# =============================================================================
# GPU telemetry (optional)
# =============================================================================
GPU_TELEMETRY_SRUN_PID=""
if is_truthy "${CHEMTRAIN_PROFILE_GPU_TELEMETRY:-0}"; then
    GPU_TELEMETRY_PREFIX="${RUN_OUTPUT_DIR}/gpu_telemetry_${JOB_TAG}"
    srun --overlap --ntasks-per-node=1 bash -lc "
        out='${GPU_TELEMETRY_PREFIX}_\$(hostname).csv'
        echo 'timestamp,index,util_gpu,util_mem,power_w,sm_clock_mhz,mem_clock_mhz,pci_gen,pci_width' > \"\$out\"
        while true; do
            nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits >> \"\$out\"
            sleep 1
        done
    " &
    GPU_TELEMETRY_SRUN_PID=$!
fi

cleanup_background_jobs() {
    local rc=$?
    [[ -n "${GPU_TELEMETRY_SRUN_PID}" ]] && kill "${GPU_TELEMETRY_SRUN_PID}" >/dev/null 2>&1 || true
    echo "[run_training.sh] EXIT rc=${rc}" >&2
}
trap cleanup_background_jobs EXIT

# =============================================================================
# Launch training
# =============================================================================
cd "${PROJECT_ROOT}"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: Training script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

echo "============================================================"
echo "Config:       ${RUNTIME_CONFIG}"
echo "Run dir:      ${RUN_OUTPUT_DIR}"
echo "Job tag:      ${JOB_TAG}"
echo "Nodes:        ${SLURM_NNODES:-1}"
echo "GPUs/node:    4"
echo "Grad accum:   ${CHEMTRAIN_GRAD_ACCUM_STEPS} (${CHEMTRAIN_GRAD_ACCUM_MODE})"
echo "============================================================"

TRAIN_ARGS=("${RUNTIME_CONFIG}" "${JOB_TAG}")
[[ -n "$MULTI_PROTEIN_DIR" ]] && TRAIN_ARGS+=("--multi-protein-dir" "$MULTI_PROTEIN_DIR")
[[ -n "$RESUME_VALUE" ]] && TRAIN_ARGS+=("--resume" "$RESUME_VALUE")
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

srun -l --ntasks-per-node=1 "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}" \
    "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOGFILE}"

echo "============================================================"
echo "Training complete. Log: ${LOGFILE}"
echo "SLURM stdout:   ${RUN_SLURM_OUT}"
echo "SLURM stderr:   ${RUN_SLURM_ERR}"
echo "============================================================"
