#!/bin/bash

#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=booster
#SBATCH --output=/p/project1/cameo/schmidt36/cameo_cg/ml_training_testing_suite/outputs/slurm-bootstrap-%j.out

set -Eeuo pipefail

RUN_TRAINING_TRACE="${RUN_TRAINING_TRACE:-0}"
if [[ "${RUN_TRAINING_TRACE}" == "1" ]]; then
    export PS4='[run_training_testing.slurm:${LINENO}] '
    set -x
fi

run_training_testing_err_trap() {
    local rc=$?
    echo "[run_training_testing.slurm] ERROR rc=${rc} line=${BASH_LINENO[0]} cmd=${BASH_COMMAND}" >&2
    exit "${rc}"
}
trap run_training_testing_err_trap ERR

SCRIPT_DIR="${CAMEO_TRAINING_TESTING_SUITE_DIR:-}"
if [[ -n "${SCRIPT_DIR}" && -d "${SCRIPT_DIR}" ]]; then
    SCRIPT_DIR="$(cd "${SCRIPT_DIR}" && pwd -P)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
fi

PROJECT_ROOT="${CAMEO_CG_PROJECT_ROOT:-}"
if [[ -n "${PROJECT_ROOT}" && -d "${PROJECT_ROOT}" ]]; then
    PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd -P)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/train.py" ]]; then
    PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
fi
DEFAULT_OUTPUTS_ROOT="${SCRIPT_DIR}/outputs"

canonical_path() {
    local raw="$1"
    local base=""
    if [[ -z "${raw}" ]]; then
        return 1
    fi
    if [[ "${raw}" == /* ]]; then
        if [[ -e "${raw}" ]]; then
            (cd "$(dirname "${raw}")" && printf '%s/%s
' "$(pwd -P)" "$(basename "${raw}")")
        else
            printf '%s
' "${raw}"
        fi
        return 0
    fi
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -e "${SLURM_SUBMIT_DIR}/${raw}" ]]; then
        (cd "${SLURM_SUBMIT_DIR}" && printf '%s/%s
' "$(pwd -P)" "${raw}")
        return 0
    fi
    if [[ -e "${PROJECT_ROOT}/${raw}" ]]; then
        (cd "${PROJECT_ROOT}" && printf '%s/%s
' "$(pwd -P)" "${raw}")
        return 0
    fi
    base="$(basename "${raw}")"
    if [[ -d "$(dirname "${raw}")" ]]; then
        (cd "$(dirname "${raw}")" && printf '%s/%s
' "$(pwd -P)" "${base}")
        return 0
    fi
    printf '%s/%s
' "$(pwd -P)" "${raw}"
}

CONFIG_FILE="${1:-}"
if [[ -n "${CONFIG_FILE}" ]]; then
    shift || true
fi

if [[ -z "${CONFIG_FILE}" && -n "${CONFIG_LIST_FILE:-}" ]]; then
    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        echo "ERROR: CONFIG_LIST_FILE was provided but SLURM_ARRAY_TASK_ID is not set"
        exit 1
    fi
    if [[ ! -f "${CONFIG_LIST_FILE}" ]]; then
        echo "ERROR: Config manifest not found: ${CONFIG_LIST_FILE}"
        exit 1
    fi
    CONFIG_FILE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${CONFIG_LIST_FILE}")"
fi

if [[ -z "${CONFIG_FILE}" ]]; then
    echo "Usage: sbatch ml_training_testing_suite/run_training_testing.slurm <config.yaml> [extra train.py args]"
    echo "   or: sbatch --array ... --export=...,CONFIG_LIST_FILE=<manifest> ml_training_testing_suite/run_training_testing.slurm"
    exit 1
fi

CONFIG_FILE="$(canonical_path "${CONFIG_FILE}")"
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

OUTPUTS_ROOT="${PARENT_OUTPUT_DIR:-${DEFAULT_OUTPUTS_ROOT}}"
if [[ "${OUTPUTS_ROOT}" != /* ]]; then
    OUTPUTS_ROOT="${PROJECT_ROOT}/${OUTPUTS_ROOT}"
fi
mkdir -p "${OUTPUTS_ROOT}"
OUTPUTS_ROOT="$(cd "${OUTPUTS_ROOT}" && pwd -P)"

RUN_NAME="${RUN_NAME:-$(basename "${CONFIG_FILE%.*}")}"
DATE_TAG="$(date +%Y%m%d)"
if [[ -n "${PARENT_OUTPUT_DIR:-}" ]]; then
    RUN_OUTPUT_DIR="${OUTPUTS_ROOT}/${RUN_NAME}"
else
    RUN_OUTPUT_DIR="${OUTPUTS_ROOT}/${DATE_TAG}_${RUN_NAME}"
fi
RUN_EXPORT_DIR="${RUN_OUTPUT_DIR}/exports"
RUN_CHECKPOINT_DIR="${RUN_OUTPUT_DIR}/checkpoints"
RUN_PROFILE_DIR="${RUN_OUTPUT_DIR}/profiles"
mkdir -p "${RUN_OUTPUT_DIR}" "${RUN_EXPORT_DIR}" "${RUN_CHECKPOINT_DIR}" "${RUN_PROFILE_DIR}"
RUN_OUTPUT_DIR="$(cd "${RUN_OUTPUT_DIR}" && pwd -P)"
RUN_EXPORT_DIR="$(cd "${RUN_EXPORT_DIR}" && pwd -P)"
RUN_CHECKPOINT_DIR="$(cd "${RUN_CHECKPOINT_DIR}" && pwd -P)"
RUN_PROFILE_DIR="$(cd "${RUN_PROFILE_DIR}" && pwd -P)"

JOB_TAG="${SLURM_JOB_ID:-local}"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    JOB_TAG="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
BOOTSTRAP_SLURM_OUT="${OUTPUTS_ROOT}/slurm-bootstrap-${JOB_TAG}.out"
RUN_SLURM_OUT="${RUN_OUTPUT_DIR}/slurm-${JOB_TAG}.out"
LIVE_LOGFILE="${OUTPUTS_ROOT}/train_allegro_${JOB_TAG}.log"
RUN_LOGFILE="${RUN_OUTPUT_DIR}/train_allegro_${JOB_TAG}.log"
INPUT_CONFIG_COPY="${RUN_OUTPUT_DIR}/config_input.yaml"
RUNTIME_CONFIG="${RUN_OUTPUT_DIR}/config_runtime.yaml"
cp -f "${CONFIG_FILE}" "${INPUT_CONFIG_COPY}"

MODEL_TYPE="$(
    awk '
        /^[[:space:]]*#/ { next }
        /^[[:space:]]*ml_model:[[:space:]]*/ {
            line = $0
            sub(/^[[:space:]]*ml_model:[[:space:]]*/, "", line)
            sub(/[[:space:]]*#.*/, "", line)
            gsub(/[[:space:]"]/, "", line)
            print line
            exit
        }
    ' "${CONFIG_FILE}"
)"

if [[ -z "${MODEL_TYPE}" ]]; then
    echo "ERROR: Could not determine model.ml_model from config: ${CONFIG_FILE}"
    exit 1
fi

MODEL_TYPE_LOWER="${MODEL_TYPE,,}"
MODEL_TYPE_CANON="${MODEL_TYPE_LOWER//-/_}"
if [[ "${MODEL_TYPE_CANON}" == "allegro_cueq" || "${MODEL_TYPE_CANON}" == "allegro_cueq_fast" || "${MODEL_TYPE_CANON}" == "allegro_cueq_opt" || "${MODEL_TYPE_CANON}" == "allegro_cueq_b1" || "${MODEL_TYPE_CANON}" == "allegro_cueq_fast_1103" ]]; then
    SELECTED_VENV="/p/project1/cameo/schmidt36/env_cueq_allegro_opt"
    SELECTED_VENV_NAME="env_cueq_allegro_opt"
else
    SELECTED_VENV="/p/project1/cameo/schmidt36/clean_booster_env"
    SELECTED_VENV_NAME="clean_booster_env"
fi

source /p/project1/cameo/schmidt36/load_modules.sh
source "${SELECTED_VENV}/bin/activate"
source /p/project1/cameo/schmidt36/set_lammps_paths.sh

PYTHON_BIN="$(command -v python)"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "ERROR: No python found after activating ${SELECTED_VENV_NAME}"
    exit 1
fi

echo "Selected ml_model: ${MODEL_TYPE_CANON} -> activated ${SELECTED_VENV_NAME}"

"${PYTHON_BIN}" - <<'PYCFG' "${CONFIG_FILE}" "${RUNTIME_CONFIG}" "${RUN_EXPORT_DIR}" "${RUN_CHECKPOINT_DIR}" "${RUN_PROFILE_DIR}" "${PROJECT_ROOT}"
from pathlib import Path
import sys
import yaml

src = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
export_dir = str(Path(sys.argv[3]).resolve())
checkpoint_dir = str(Path(sys.argv[4]).resolve())
profile_dir = str(Path(sys.argv[5]).resolve())
project_root = Path(sys.argv[6]).resolve()

data = yaml.safe_load(src.read_text()) or {}
training = data.setdefault('training', {})
training['export_path'] = export_dir
training['checkpoint_path'] = checkpoint_dir
profiling = training.setdefault('profiling', {})
profiling['trace_dir'] = profile_dir

model = data.setdefault('model', {})
priors = model.get('priors') or {}
spline_file = priors.get('spline_file')
if spline_file:
    spline_path = Path(spline_file)
    if not spline_path.is_absolute():
        candidates = [
            project_root / spline_path,
            src.parent / spline_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                priors['spline_file'] = str(candidate.resolve())
                break
        else:
            priors['spline_file'] = str((project_root / spline_path).resolve())
    model['priors'] = priors

data_path = ((data.get('data') or {}).get('path'))
if data_path:
    data_path_obj = Path(data_path)
    if not data_path_obj.is_absolute():
        data['data']['path'] = str((project_root / data_path_obj).resolve())

out.write_text(yaml.safe_dump(data, sort_keys=False))
PYCFG

export CC="$(which gcc)"
export CXX="$(which g++)"
export CLANG_CUDA_COMPILER_PATH="$(which gcc)"
CUDA_ROOT="$("${PYTHON_BIN}" -c 'import os; from jax_plugins import xla_cuda12; print(os.path.dirname(xla_cuda12.__file__))')"
SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${CUDA_ROOT}:${SITE_PACKAGES}/nvidia/cudnn/lib:${SITE_PACKAGES}/nvidia/cuda_runtime/lib:${SITE_PACKAGES}/nvidia/cublas/lib:${SITE_PACKAGES}/nvidia/cusolver/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME=/p/software/juwelsbooster/stages/2025/software/CUDA/12
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME} --xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy

cleanup_suite_outputs() {
    local rc=$?
    if [[ -f "${BOOTSTRAP_SLURM_OUT}" ]]; then
        cp -f "${BOOTSTRAP_SLURM_OUT}" "${RUN_SLURM_OUT}" 2>/dev/null || true
    fi
    rm -f "${LIVE_LOGFILE}" 2>/dev/null || true
    exit "${rc}"
}
trap cleanup_suite_outputs EXIT

cd "${PROJECT_ROOT}"

echo "============================================================"
echo "ML Training Testing Suite"
echo "============================================================"
echo "Input config:      ${CONFIG_FILE}"
echo "Runtime config:    ${RUNTIME_CONFIG}"
echo "Run name:          ${RUN_NAME}"
echo "Date tag:          ${DATE_TAG}"
echo "Array task id:     ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "Config manifest:   ${CONFIG_LIST_FILE:-n/a}"
echo "Outputs root:      ${OUTPUTS_ROOT}"
echo "Run output dir:    ${RUN_OUTPUT_DIR}"
echo "Run log:           ${RUN_LOGFILE}"
echo "Slurm copy:        ${RUN_SLURM_OUT}"
echo "Project root:      ${PROJECT_ROOT}"
echo "Selected venv:     ${SELECTED_VENV}"
echo "Module Environment"
echo "============================================================"
module list || true
echo "============================================================"

TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train.py"
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: Training script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

srun -l --ntasks-per-node=1 "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}" "${RUNTIME_CONFIG}" "${JOB_TAG}" "$@" 2>&1 | tee "${LIVE_LOGFILE}" "${RUN_LOGFILE}"
