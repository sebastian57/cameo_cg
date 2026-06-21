#!/bin/bash

# Discover LAMMPS / chemtrain-deploy runtime paths from the active environment.
# For use with load_modules_2026.sh + venv_cameocg_juwels2026.
#
# Expected usage:
#   1. activate the desired Python environment first
#   2. source this script
#
# Optional overrides:
#   CAMEO_LAMMPS_BUILD_DIR  - explicit LAMMPS build/bin directory to prepend to PATH
#   LAMMPS_PLUGIN_PATH      - explicit plugin build directory
#   JCN_PJRT_PATH           - explicit PJRT plugin library directory
#
# This script intentionally derives the chemtrain-deploy paths from the
# installed chemtrain package location so editable installs keep working after
# cloning repos into a different filesystem layout.

_python_bin="${PYTHON_BIN:-$(command -v python 2>/dev/null || true)}"
if [[ -z "${_python_bin}" ]]; then
    echo "ERROR: Could not find python while setting LAMMPS paths." >&2
    return 1 2>/dev/null || exit 1
fi

if [[ -n "${CAMEO_LAMMPS_BUILD_DIR:-}" ]]; then
    _lammps_build_dir="${CAMEO_LAMMPS_BUILD_DIR}"
elif command -v lmp >/dev/null 2>&1; then
    _lammps_build_dir=""
else
    _lammps_build_dir=""
fi

if [[ -n "${_lammps_build_dir}" ]]; then
    export PATH="${_lammps_build_dir}:${PATH}"
fi

_chemtrain_repo_root="$(${_python_bin} - <<'PYIN'
from pathlib import Path
import importlib.util

spec = importlib.util.find_spec("chemtrain")
if spec is None or spec.origin is None:
    raise SystemExit(1)
print(Path(spec.origin).resolve().parents[1])
PYIN
)"

if [[ -z "${_chemtrain_repo_root}" ]]; then
    echo "ERROR: Failed to resolve chemtrain install root from Python." >&2
    return 1 2>/dev/null || exit 1
fi

_deploy_root="${_chemtrain_repo_root}/chemtrain-deploy"
_default_plugin_path="${_deploy_root}/build"
_default_pjrt_path="${_deploy_root}/lib"

export CHEMTRAIN_REPO_ROOT="${CHEMTRAIN_REPO_ROOT:-${_chemtrain_repo_root}}"
export CHEMTRAIN_DEPLOY_ROOT="${CHEMTRAIN_DEPLOY_ROOT:-${_deploy_root}}"
export LAMMPS_PLUGIN_PATH="${LAMMPS_PLUGIN_PATH:-${_default_plugin_path}}"
export JCN_PJRT_PATH="${JCN_PJRT_PATH:-${_default_pjrt_path}}"

if [[ -d "${JCN_PJRT_PATH}" ]]; then
    export LD_LIBRARY_PATH="${JCN_PJRT_PATH}:${LD_LIBRARY_PATH:-}"
fi

_nvidia_nvrtc_lib="$(${_python_bin} - <<'PYIN'
from pathlib import Path
import site

for root in site.getsitepackages():
    candidate = Path(root) / "nvidia" / "cuda_nvrtc" / "lib"
    if candidate.is_dir():
        print(candidate)
        break
PYIN
)"

if [[ -n "${_nvidia_nvrtc_lib}" && -d "${_nvidia_nvrtc_lib}" ]]; then
    export LD_LIBRARY_PATH="${_nvidia_nvrtc_lib}:${LD_LIBRARY_PATH:-}"
fi

_nvidia_cublas_lib="$(${_python_bin} - <<'PYIN'
from pathlib import Path
import site

for root in site.getsitepackages():
    candidate = Path(root) / "nvidia" / "cublas" / "lib"
    if candidate.is_dir():
        print(candidate)
        break
PYIN
)"

if [[ -n "${_nvidia_cublas_lib}" && -d "${_nvidia_cublas_lib}" ]]; then
    export LD_LIBRARY_PATH="${_nvidia_cublas_lib}:${LD_LIBRARY_PATH:-}"
fi

_cueq_ops_lib="$(${_python_bin} - <<'PYIN'
from pathlib import Path
import site

for root in site.getsitepackages():
    candidate = Path(root) / "cuequivariance_ops" / "lib"
    if candidate.is_dir():
        print(candidate)
        break
PYIN
)"

if [[ -n "${_cueq_ops_lib}" && -d "${_cueq_ops_lib}" ]]; then
    export LD_LIBRARY_PATH="${_cueq_ops_lib}:${LD_LIBRARY_PATH:-}"
fi

if [[ ! -d "${CHEMTRAIN_DEPLOY_ROOT}" ]]; then
    echo "WARNING: chemtrain-deploy directory not found next to chemtrain install: ${CHEMTRAIN_DEPLOY_ROOT}" >&2
fi
if [[ ! -d "${LAMMPS_PLUGIN_PATH}" ]]; then
    echo "WARNING: LAMMPS_PLUGIN_PATH does not exist: ${LAMMPS_PLUGIN_PATH}" >&2
fi
if [[ ! -d "${JCN_PJRT_PATH}" ]]; then
    echo "WARNING: JCN_PJRT_PATH does not exist: ${JCN_PJRT_PATH}" >&2
fi
