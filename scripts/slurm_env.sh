#!/bin/bash
# =============================================================================
# Shared SLURM environment setup — sourced by run_training.sh and other scripts.
#
# Expects:
#   CONFIG_FILE  — path to a YAML config (must be set before sourcing)
#
# Exports:
#   MODEL_TYPE_CANON  — canonical ml_model name (e.g. allegro_cueq_fast)
#   SELECTED_VENV     — path to the activated virtual environment
#   PYTHON_BIN        — full path to the python interpreter
#   CUDA_HOME, LD_LIBRARY_PATH, XLA_FLAGS, NCCL_*, XLA_PYTHON_CLIENT_*
# =============================================================================

# Disable core dumps — they can be gigabytes and fill small filesystems.
# Override with CORE_DUMP_ENABLED=1 if you need to debug a crash.
if [[ "${CORE_DUMP_ENABLED:-0}" != "1" ]]; then
    ulimit -c 0
fi

if [[ -z "${CONFIG_FILE:-}" ]]; then
    echo "ERROR: CONFIG_FILE must be set before sourcing slurm_env.sh" >&2
    return 1 2>/dev/null || exit 1
fi

if [[ -n "${CAMEO_CG_PROJECT_ROOT:-}" ]]; then
    PROJECT_ROOT="${CAMEO_CG_PROJECT_ROOT}"
elif [[ -n "${PROJECT_ROOT:-}" ]]; then
    PROJECT_ROOT="${PROJECT_ROOT}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
fi
export PROJECT_ROOT

# ---------------------------------------------------------------------------
# Parse ml_model from YAML without Python (awk-based, fast)
# ---------------------------------------------------------------------------
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
    echo "ERROR: Could not determine model.ml_model from config: ${CONFIG_FILE}" >&2
    return 1 2>/dev/null || exit 1
fi

MODEL_TYPE_LOWER="${MODEL_TYPE,,}"
MODEL_TYPE_CANON="${MODEL_TYPE_LOWER//-/_}"
export MODEL_TYPE_CANON

# ---------------------------------------------------------------------------
# Select virtual environment based on model type
# ---------------------------------------------------------------------------
SELECTED_VENV_SOURCE="CAMEO_ACTIVE_VENV"
if [[ -n "${CAMEO_ACTIVE_VENV:-}" ]]; then
    SELECTED_VENV="${CAMEO_ACTIVE_VENV}"
elif [[ "${MODEL_TYPE_CANON}" == allegro_cueq* ]]; then
    SELECTED_VENV_SOURCE="CAMEO_CUEQ_VENV"
    SELECTED_VENV="${CAMEO_CUEQ_VENV:-}"
else
    SELECTED_VENV_SOURCE="CAMEO_STANDARD_VENV"
    SELECTED_VENV="${CAMEO_STANDARD_VENV:-}"
fi

if [[ -z "${SELECTED_VENV:-}" ]]; then
    echo "Python Venv not set at ${SELECTED_VENV_SOURCE}" >&2
    return 1 2>/dev/null || exit 1
fi

if [[ ! -d "${SELECTED_VENV}" || ! -f "${SELECTED_VENV}/bin/activate" ]]; then
    echo "Python Venv not set at ${SELECTED_VENV}" >&2
    return 1 2>/dev/null || exit 1
fi

SELECTED_VENV_NAME="$(basename "${SELECTED_VENV}")"
export SELECTED_VENV

# ---------------------------------------------------------------------------
# Load modules and activate environment
# ---------------------------------------------------------------------------
source "${PROJECT_ROOT}/env_setup/load_modules_2026.sh"
source "${SELECTED_VENV}/bin/activate"
PYTHON_BIN="$(command -v python)"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "ERROR: No python found after activating ${SELECTED_VENV_NAME}" >&2
    return 1 2>/dev/null || exit 1
fi
export PYTHON_BIN
source "${PROJECT_ROOT}/env_setup/set_lammps_paths_2026.sh"

echo "Selected ml_model: ${MODEL_TYPE_CANON} -> activated ${SELECTED_VENV_NAME}"
echo "JAX version: $(${PYTHON_BIN} -c 'import jax; print(jax.__version__)' 2>&1)"
echo "JAX location: $(${PYTHON_BIN} -c 'import jax, os; print(os.path.dirname(jax.__file__))' 2>&1)"

# ---------------------------------------------------------------------------
# Compiler and CUDA setup
# ---------------------------------------------------------------------------
export CC="$(which gcc)"
export CXX="$(which g++)"
export CLANG_CUDA_COMPILER_PATH="$(which gcc)"

CUDA_ROOT="$(${PYTHON_BIN} -c 'import os; from jax_plugins import xla_cuda12; print(os.path.dirname(xla_cuda12.__file__))')"
SITE_PACKAGES="$(${PYTHON_BIN} -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${CUDA_ROOT}:${SITE_PACKAGES}/nvidia/cudnn/lib:${SITE_PACKAGES}/nvidia/cuda_runtime/lib:${SITE_PACKAGES}/nvidia/cublas/lib:${SITE_PACKAGES}/nvidia/cusolver/lib:${LD_LIBRARY_PATH:-}"

export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME} --xla_gpu_autotune_level=0"
export ALLEGRO_TP_METHOD_FALLBACK="${ALLEGRO_TP_METHOD_FALLBACK:-error}"

# ---------------------------------------------------------------------------
# JAX / XLA / NCCL runtime settings
# ---------------------------------------------------------------------------
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"

export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy
