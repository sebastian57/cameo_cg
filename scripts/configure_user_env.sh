#!/bin/bash
# Persist cameo_cg_pkgflow environment variables in ~/.bashrc.
#
# Usage examples:
#   source scripts/configure_user_env.sh \
#       --project-root /p/project1/cameo/schmidt36/cameo_cg_pkgflow \
#       --cueq-venv /p/project1/cameo/schmidt36/test_env_newsetup \
#       --standard-venv /p/project1/cameo/schmidt36/clean_booster_env
#
#   source scripts/configure_user_env.sh \
#       --active-venv /p/project1/cameo/schmidt36/test_env_newsetup
#
#   source scripts/configure_user_env.sh --show

set -euo pipefail

BASHRC_PATH="${HOME}/.bashrc"
START_MARK="# >>> cameo_cg_pkgflow env >>>"
END_MARK="# <<< cameo_cg_pkgflow env <<<"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")"
DEFAULT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

PROJECT_ROOT_VALUE="${DEFAULT_PROJECT_ROOT}"
CUEQ_VENV_VALUE=""
STANDARD_VENV_VALUE=""
ACTIVE_VENV_VALUE=""
LAMMPS_BUILD_DIR_VALUE=""
LMP_BIN_VALUE=""
SHOW_ONLY=0

usage() {
    cat <<USAGE
Usage:
  source scripts/configure_user_env.sh [options]

Options:
  --project-root PATH       Set CAMEO_CG_PROJECT_ROOT (default: this checkout)
  --cueq-venv PATH          Set CAMEO_CUEQ_VENV
  --standard-venv PATH      Set CAMEO_STANDARD_VENV
  --active-venv PATH        Set CAMEO_ACTIVE_VENV
  --lammps-build-dir PATH   Set CAMEO_LAMMPS_BUILD_DIR
  --lmp-bin PATH            Set CAMEO_LMP_BIN
  --bashrc PATH             Write to a different shell rc file
  --show                    Print the currently managed block and exit
  -h, --help                Show this help

Notes:
  - Run this script with 'source' if you want the current shell to pick up
    the updated values immediately.
  - Running it as a normal script still updates ~/.bashrc for future shells.
USAGE
}

quote_shell() {
    printf '%q' "$1"
}

build_block() {
    local block="${START_MARK}"
    block+=$'\n'
    block+="# Managed by ${SCRIPT_PATH}"
    block+=$'\n'
    block+="export CAMEO_CG_PROJECT_ROOT=$(quote_shell "${PROJECT_ROOT_VALUE}")"
    block+=$'\n'
    if [[ -n "${CUEQ_VENV_VALUE}" ]]; then
        block+="export CAMEO_CUEQ_VENV=$(quote_shell "${CUEQ_VENV_VALUE}")"
        block+=$'\n'
    fi
    if [[ -n "${STANDARD_VENV_VALUE}" ]]; then
        block+="export CAMEO_STANDARD_VENV=$(quote_shell "${STANDARD_VENV_VALUE}")"
        block+=$'\n'
    fi
    if [[ -n "${ACTIVE_VENV_VALUE}" ]]; then
        block+="export CAMEO_ACTIVE_VENV=$(quote_shell "${ACTIVE_VENV_VALUE}")"
        block+=$'\n'
    fi
    if [[ -n "${LAMMPS_BUILD_DIR_VALUE}" ]]; then
        block+="export CAMEO_LAMMPS_BUILD_DIR=$(quote_shell "${LAMMPS_BUILD_DIR_VALUE}")"
        block+=$'\n'
    fi
    if [[ -n "${LMP_BIN_VALUE}" ]]; then
        block+="export CAMEO_LMP_BIN=$(quote_shell "${LMP_BIN_VALUE}")"
        block+=$'\n'
    fi
    block+="${END_MARK}"
    printf '%s\n' "${block}"
}

show_block() {
    if [[ ! -f "${BASHRC_PATH}" ]]; then
        echo "No rc file found at ${BASHRC_PATH}"
        return 0
    fi
    awk -v start="${START_MARK}" -v end="${END_MARK}" '
        $0 == start { printing = 1 }
        printing { print }
        $0 == end { printing = 0 }
    ' "${BASHRC_PATH}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-root)
            PROJECT_ROOT_VALUE="$2"
            shift 2
            ;;
        --cueq-venv)
            CUEQ_VENV_VALUE="$2"
            shift 2
            ;;
        --standard-venv)
            STANDARD_VENV_VALUE="$2"
            shift 2
            ;;
        --active-venv)
            ACTIVE_VENV_VALUE="$2"
            shift 2
            ;;
        --lammps-build-dir)
            LAMMPS_BUILD_DIR_VALUE="$2"
            shift 2
            ;;
        --lmp-bin)
            LMP_BIN_VALUE="$2"
            shift 2
            ;;
        --bashrc)
            BASHRC_PATH="$2"
            shift 2
            ;;
        --show)
            SHOW_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            return 0 2>/dev/null || exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            return 1 2>/dev/null || exit 1
            ;;
    esac
done

if [[ "${SHOW_ONLY}" == "1" ]]; then
    show_block
    return 0 2>/dev/null || exit 0
fi

mkdir -p "$(dirname "${BASHRC_PATH}")"
touch "${BASHRC_PATH}"

TMP_FILE="$(mktemp)"
trap 'rm -f "${TMP_FILE}"' EXIT

awk -v start="${START_MARK}" -v end="${END_MARK}" '
    $0 == start { skipping = 1; next }
    $0 == end { skipping = 0; next }
    !skipping { print }
' "${BASHRC_PATH}" > "${TMP_FILE}"

{
    cat "${TMP_FILE}"
    if [[ -s "${TMP_FILE}" ]]; then
        printf '\n'
    fi
    build_block
    printf '\n'
} > "${BASHRC_PATH}"

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    # shellcheck source=/dev/null
    source "${BASHRC_PATH}"
    echo "Updated ${BASHRC_PATH} and reloaded it into the current shell."
else
    # shellcheck source=/dev/null
    source "${BASHRC_PATH}"
    echo "Updated ${BASHRC_PATH}."
    echo "Run 'source ${BASHRC_PATH}' in your shell, or invoke this script with 'source', to refresh the current session."
fi

echo "Managed environment block:"
show_block
