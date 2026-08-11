#!/bin/bash

#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --partition=booster
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Usage:
#   sbatch scripts/run_relative_entropy.sh local_work/re_finetuning_1pro_4zohB01_320/config_re_aggforce_from_fm.yaml
set -Eeuo pipefail

re_err_trap() {
    local rc=$?
    echo "[run_relative_entropy.sh] ERROR rc=${rc} line=${BASH_LINENO[0]} cmd=${BASH_COMMAND}" >&2
    exit "${rc}"
}
trap re_err_trap ERR

CONFIG_FILE=""
RESUME_VALUE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            RESUME_VALUE="${2:?--resume requires an argument (auto|<path>)}"; shift 2 ;;
        -h|--help)
            echo "Usage: sbatch scripts/run_relative_entropy.sh <config.yaml> [--resume auto|<path>]"
            exit 0 ;;
        *)
            if [[ -z "${CONFIG_FILE}" ]]; then CONFIG_FILE="$1"; shift
            else echo "ERROR: unexpected argument: $1" >&2; exit 1; fi ;;
    esac
done
if [[ -z "${CONFIG_FILE}" ]]; then
    echo "Usage: sbatch scripts/run_relative_entropy.sh <config.yaml> [--resume auto|<path>]"
    exit 1
fi

PROJECT_ROOT="${CAMEO_CG_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
resolve_repo_root() {
    local hint cand
    for hint in "$@"; do
        [[ -z "${hint}" ]] && continue
        cand="${hint}"
        [[ -f "${cand}" ]] && cand="$(dirname "${cand}")"
        while [[ "${cand}" != "/" ]]; do
            if [[ -f "${cand}/scripts/slurm_env.sh" && -f "${cand}/scripts/train_relative_entropy.py" ]]; then
                printf '%s
' "$(cd "${cand}" && pwd -P)"
                return 0
            fi
            cand="$(dirname "${cand}")"
        done
    done
    return 1
}

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
PROJECT_ROOT="$(resolve_repo_root "${CONFIG_DIR}" "${SLURM_SUBMIT_DIR:-}" "${PROJECT_ROOT}" "$(pwd -P)" "$(dirname "${BASH_SOURCE[0]}")")" || {
    echo "ERROR: Could not locate project root containing scripts/slurm_env.sh and scripts/train_relative_entropy.py"
    exit 1
}
SCRIPT_DIR="${PROJECT_ROOT}/scripts"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "ERROR: Could not locate python/python3 before loading SLURM env."
    exit 1
fi

JOB_TAG="${SLURM_JOB_ID:-local}"
RUN_NAME="${RUN_NAME:-$(basename "${CONFIG_FILE%.*}")}"
DATE_TAG="$(date +%Y%m%d)"
CONFIG_OUTPUT_DIR="$(${PYTHON_BIN} -c "
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
    RUN_OUTPUT_DIR="${PROJECT_ROOT}/local_work/outputs/${DATE_TAG}_${RUN_NAME}"
fi
mkdir -p "${RUN_OUTPUT_DIR}"
RUN_OUTPUT_DIR="$(cd "${RUN_OUTPUT_DIR}" && pwd -P)"
RUN_EXPORT_DIR="${RUN_OUTPUT_DIR}/exports"
RUN_CHECKPOINT_DIR="${RUN_OUTPUT_DIR}/checkpoints"
RUN_PROFILE_DIR="${RUN_OUTPUT_DIR}/profiles"
RUN_RE_DIR="${RUN_OUTPUT_DIR}/relative_entropy"
mkdir -p "${RUN_EXPORT_DIR}" "${RUN_CHECKPOINT_DIR}" "${RUN_PROFILE_DIR}" "${RUN_RE_DIR}"

RUN_SLURM_OUT="${RUN_OUTPUT_DIR}/slurm-re-${JOB_TAG}.out"
RUN_SLURM_ERR="${RUN_OUTPUT_DIR}/slurm-re-${JOB_TAG}.err"
touch "${RUN_SLURM_OUT}" "${RUN_SLURM_ERR}"
exec > >(tee -a "${RUN_SLURM_OUT}") 2> >(tee -a "${RUN_SLURM_ERR}" >&2)

echo "[run_relative_entropy.sh] Project root: ${PROJECT_ROOT}"
echo "[run_relative_entropy.sh] Config file:  ${CONFIG_FILE}"
echo "[run_relative_entropy.sh] Run dir:      ${RUN_OUTPUT_DIR}"

source "${SCRIPT_DIR}/slurm_env.sh"

INPUT_CONFIG_COPY="${RUN_OUTPUT_DIR}/config_input_re_${JOB_TAG}.yaml"
RUNTIME_CONFIG="${RUN_OUTPUT_DIR}/config_runtime_re_${JOB_TAG}.yaml"
cp -f "${CONFIG_FILE}" "${INPUT_CONFIG_COPY}"
ln -sfn "$(basename "${INPUT_CONFIG_COPY}")" "${RUN_OUTPUT_DIR}/config_input_re.yaml"

"${PYTHON_BIN}" - <<'PYCFG' "${CONFIG_FILE}" "${RUNTIME_CONFIG}" "${RUN_OUTPUT_DIR}" "${RUN_EXPORT_DIR}" "${RUN_CHECKPOINT_DIR}" "${RUN_PROFILE_DIR}" "${RUN_RE_DIR}" "${PROJECT_ROOT}"
from pathlib import Path
import sys, yaml
src = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
run_dir = Path(sys.argv[3]).resolve()
export_dir = Path(sys.argv[4]).resolve()
checkpoint_dir = Path(sys.argv[5]).resolve()
profile_dir = Path(sys.argv[6]).resolve()
re_dir = Path(sys.argv[7]).resolve()
project_root = Path(sys.argv[8]).resolve()
data = yaml.safe_load(src.read_text()) or {}
paths = data.setdefault('paths', {})
paths['output_dir'] = str(run_dir)
paths['export_dir'] = str(export_dir)
paths['checkpoint_dir'] = str(checkpoint_dir)
paths['profile_dir'] = str(profile_dir)
paths['slurm_dir'] = str(run_dir)
training = data.setdefault('training', {})
training['export_path'] = str(export_dir)
training['checkpoint_path'] = str(checkpoint_dir)
profiling = training.setdefault('profiling', {})
profiling['trace_dir'] = str(profile_dir)
re_cfg = training.setdefault('relative_entropy', {})
re_cfg['output_dir'] = str(re_dir)
for section, key in [
    (data.get('data') or {}, 'path'),
    (re_cfg, 'reference_data_path'),
    (re_cfg, 'initial_state_data_path'),
]:
    raw = section.get(key)
    if raw:
        p = Path(str(raw))
        if not p.is_absolute():
            for cand in (src.parent / p, project_root / p):
                if cand.exists():
                    section[key] = str(cand.resolve())
                    break
            else:
                section[key] = str((project_root / p).resolve())
init = training.get('init_from_checkpoint') or {}
raw = init.get('path')
if raw:
    p = Path(str(raw))
    if not p.is_absolute():
        for cand in (src.parent / p, project_root / p):
            if cand.exists():
                init['path'] = str(cand.resolve())
                break
        else:
            init['path'] = str((project_root / p).resolve())
    training['init_from_checkpoint'] = init
out.write_text(yaml.safe_dump(data, sort_keys=False))
PYCFG
ln -sfn "$(basename "${RUNTIME_CONFIG}")" "${RUN_OUTPUT_DIR}/config_runtime_re.yaml"

LOGFILE="${RUN_OUTPUT_DIR}/relative_entropy_${JOB_TAG}.log"
cd "${PROJECT_ROOT}"
echo "============================================================"
echo "Relative entropy config: ${RUNTIME_CONFIG}"
echo "Run dir:                 ${RUN_OUTPUT_DIR}"
echo "Job tag:                 ${JOB_TAG}"
echo "============================================================"

source "${PROJECT_ROOT}/runs/registry_hook.sh"
run_registry_start relative-entropy "${CONFIG_FILE}" "${RUN_OUTPUT_DIR}"
run_registry_install_exit_trap

RE_ARGS=("${RUNTIME_CONFIG}")
[[ -n "${RESUME_VALUE}" ]] && RE_ARGS+=("--resume" "${RESUME_VALUE}")
srun -l --ntasks-per-node=1 "${PYTHON_BIN}" -u "${SCRIPT_DIR}/train_relative_entropy.py" "${RE_ARGS[@]}" 2>&1 | tee "${LOGFILE}"
echo "============================================================"
echo "Relative entropy complete. Log: ${LOGFILE}"
echo "============================================================"
