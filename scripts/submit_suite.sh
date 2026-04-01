#!/bin/bash
# =============================================================================
# Submit a suite of training runs as a SLURM array job.
#
# Usage:
#   scripts/submit_suite.sh --input_dir <config_dir> --name <suite_name> \
#       [--max_concurrent 4] [--nodes 1]
#
# Each *.yaml in <config_dir> becomes one array task.
# =============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

submit_suite_err_trap() {
    local rc=$?
    echo "[submit_suite.sh] ERROR rc=${rc} line=${BASH_LINENO[0]} cmd=${BASH_COMMAND}" >&2
    exit "${rc}"
}
trap submit_suite_err_trap ERR

canonical_path() {
    local raw="$1"
    if [[ -z "${raw}" ]]; then return 1; fi
    if [[ "${raw}" == /* ]]; then
        if [[ -e "${raw}" ]]; then
            (cd "$(dirname "${raw}")" && printf '%s/%s\n' "$(pwd -P)" "$(basename "${raw}")")
        else
            printf '%s\n' "${raw}"
        fi
        return 0
    fi
    if [[ -e "${PROJECT_ROOT}/${raw}" ]]; then
        (cd "${PROJECT_ROOT}" && printf '%s/%s\n' "$(pwd -P)" "${raw}")
        return 0
    fi
    printf '%s/%s\n' "$(pwd -P)" "${raw}"
}

INPUT_DIR=""
SUITE_NAME=""
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
NODES=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_dir)  INPUT_DIR="$2";       shift 2 ;;
        --name)       SUITE_NAME="$2";      shift 2 ;;
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --nodes)      NODES="$2";           shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --input_dir <config_dir> --name <suite_name> [--max_concurrent <n>] [--nodes <n>]"
            exit 1 ;;
    esac
done

if [[ -z "${INPUT_DIR}" || -z "${SUITE_NAME}" ]]; then
    echo "Usage: $0 --input_dir <config_dir> --name <suite_name> [--max_concurrent <n>] [--nodes <n>]"
    exit 1
fi

INPUT_DIR="$(canonical_path "${INPUT_DIR}")"
if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "ERROR: Input directory not found: ${INPUT_DIR}"
    exit 1
fi

DATE_TAG="$(date +%Y%m%d)"
GROUP_NAME="training_suite_${SUITE_NAME}"
GROUP_OUTPUT_DIR="${INPUT_DIR}/outputs/${DATE_TAG}_${GROUP_NAME}"
mkdir -p "${GROUP_OUTPUT_DIR}"
GROUP_OUTPUT_DIR="$(cd "${GROUP_OUTPUT_DIR}" && pwd -P)"

mapfile -t configs < <(find "${INPUT_DIR}" -maxdepth 1 -type f -name '*.yaml' | sort)
if [[ ${#configs[@]} -eq 0 ]]; then
    echo "ERROR: No YAML config files found in ${INPUT_DIR}"
    exit 1
fi

MANIFEST_PATH="${GROUP_OUTPUT_DIR}/config_manifest.txt"
: > "${MANIFEST_PATH}"
for config in "${configs[@]}"; do
    canonical_path "${config}" >> "${MANIFEST_PATH}"
done

ARRAY_MAX=$(( ${#configs[@]} - 1 ))
ARRAY_SPEC="0-${ARRAY_MAX}%${MAX_CONCURRENT}"

echo "============================================================"
echo "Submitting suite: ${GROUP_NAME}"
echo "  Project root:     ${PROJECT_ROOT}"
echo "  Input directory:  ${INPUT_DIR}"
echo "  Group output:     ${GROUP_OUTPUT_DIR}"
echo "  Config count:     ${#configs[@]}"
echo "  Array spec:       ${ARRAY_SPEC}"
echo "  Nodes per job:    ${NODES}"
echo "============================================================"

sbatch \
    --chdir "${PROJECT_ROOT}" \
    --nodes "${NODES}" \
    --array "${ARRAY_SPEC}" \
    --output "${GROUP_OUTPUT_DIR}/slurm-bootstrap-%A_%a.out" \
    --export=ALL,PARENT_OUTPUT_DIR="${GROUP_OUTPUT_DIR}",CONFIG_LIST_FILE="${MANIFEST_PATH}",CAMEO_CG_PROJECT_ROOT="${PROJECT_ROOT}" \
    "${SCRIPT_DIR}/run_training.sh"
