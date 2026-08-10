#!/bin/bash
# Failure-tolerant lifecycle bridge between Slurm launchers and registry.py.

_RUN_REGISTRY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
_RUN_REGISTRY_SCRIPT="${_RUN_REGISTRY_DIR}/registry.py"

run_registry_start() {
    local run_type="${1:-ad-hoc}"
    local config_path="${2:-}"
    shift 2 || true

    local python_bin="${RUN_REGISTRY_PYTHON:-${PYTHON_BIN:-python3}}"
    local args=(start --run-type "${run_type}")
    if [[ -n "${config_path}" ]]; then
        args+=(--config "${config_path}")
    fi
    local output_path
    for output_path in "$@"; do
        [[ -n "${output_path}" ]] && args+=(--output "${output_path}")
    done

    local identity
    if ! identity="$("${python_bin}" "${_RUN_REGISTRY_SCRIPT}" "${args[@]}")"; then
        echo "WARNING: run registry start failed; continuing without registry tracking." >&2
        RUN_REGISTRY_ID=""
        return 0
    fi
    RUN_REGISTRY_ID="${identity}"
    RUN_REGISTRY_FINISHED=0
    return 0
}

run_registry_finish() {
    local exit_code="${1:-0}"
    if [[ "${RUN_REGISTRY_FINISHED:-0}" == "1" ]]; then
        return 0
    fi
    RUN_REGISTRY_FINISHED=1
    if [[ -z "${RUN_REGISTRY_ID:-}" ]]; then
        return 0
    fi

    local python_bin="${RUN_REGISTRY_PYTHON:-${PYTHON_BIN:-python3}}"
    if ! "${python_bin}" "${_RUN_REGISTRY_SCRIPT}" finish \
        --identity "${RUN_REGISTRY_ID}" --exit-code "${exit_code}"; then
        echo "WARNING: run registry finish failed for ${RUN_REGISTRY_ID}; continuing." >&2
    fi
    return 0
}


_run_registry_exit_trap() {
    local exit_code=$?
    trap - EXIT
    run_registry_finish "${exit_code}"
    exit "${exit_code}"
}

run_registry_install_exit_trap() {
    trap _run_registry_exit_trap EXIT
}
