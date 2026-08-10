#!/bin/bash
# SLURM array submission script for parallel CAMEO CG MD replicas.
#
# Each array task runs one replica of the simulation (--replica $SLURM_ARRAY_TASK_ID).
# The number of array tasks is read automatically from n_replicas in the config.
#
# Usage:
#   scripts/submit_md_array.sh [md_config.yaml] [--max_concurrent N] [--time HH:MM:SS]
#
# Examples:
#   sbatch scripts/submit_md_array.sh configs/md_1pro_4zoh.yaml
#   sbatch scripts/submit_md_array.sh configs/md_1pro_4zoh.yaml --max_concurrent 2
#
# To override SLURM options at submit time, pass them before the script path:
#   sbatch --time=04:00:00 scripts/submit_md_array.sh configs/md_1pro_4zoh.yaml
#
#SBATCH --job-name=cameo_md_array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/cameo_md_%A_%a.out
#SBATCH --error=slurm/cameo_md_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${CAMEO_CG_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"

# ── Parse arguments ──────────────────────────────────────────────────────────
CONFIG="${1:-configs/md_1pro_4zoh.yaml}"
MAX_CONCURRENT="${MAX_CONCURRENT:-0}"  # 0 = no limit
TIME_LIMIT="${TIME_LIMIT:-}"

shift 1 || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --time)           TIME_LIMIT="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Resolve config path
if [[ "${CONFIG}" != /* ]]; then
    CONFIG="${PROJECT_ROOT}/${CONFIG}"
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config not found: ${CONFIG}"
    exit 1
fi

# ── Read n_replicas from the YAML ────────────────────────────────────────────
source "${PROJECT_ROOT}/../load_modules_2026.sh"
source "${PROJECT_ROOT}/../venv_cameocg_jupiter2026/bin/activate"
source "${PROJECT_ROOT}/../set_lammps_paths_2026.sh"

N_REPLICAS="$(python3 -c "
import yaml, sys
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('md', {}).get('n_replicas', 1))
")"

if [[ "${N_REPLICAS}" -lt 1 ]]; then
    echo "ERROR: n_replicas=${N_REPLICAS} in config — nothing to submit."
    exit 1
fi

ARRAY_MAX=$(( N_REPLICAS - 1 ))
if [[ "${MAX_CONCURRENT}" -gt 0 ]]; then
    ARRAY_SPEC="0-${ARRAY_MAX}%${MAX_CONCURRENT}"
else
    ARRAY_SPEC="0-${ARRAY_MAX}"
fi

mkdir -p "${PROJECT_ROOT}/slurm"

# ── Print summary ────────────────────────────────────────────────────────────
echo "============================================================"
echo "CAMEO CG MD Replica Array"
echo "Config       : ${CONFIG}"
echo "Project root : ${PROJECT_ROOT}"
echo "n_replicas   : ${N_REPLICAS}"
echo "Array spec   : ${ARRAY_SPEC}"
echo "============================================================"

# ── Submit ───────────────────────────────────────────────────────────────────
SBATCH_ARGS=(
    --chdir "${PROJECT_ROOT}"
    --array "${ARRAY_SPEC}"
    --export "ALL,MD_CONFIG=${CONFIG},CAMEO_CG_PROJECT_ROOT=${PROJECT_ROOT}"
)
if [[ -n "${TIME_LIMIT}" ]]; then
    SBATCH_ARGS+=(--time "${TIME_LIMIT}")
fi

# If called via `sbatch scripts/submit_md_array.sh config.yaml`, SLURM runs
# the whole script as the batch job — the task body below is executed per task.
# If called directly in bash (not under sbatch), the block below is skipped
# because SLURM_ARRAY_TASK_ID is unset.
if [[ -n "${SLURM_ARRAY_TASK_ID+x}" ]]; then
    # ── Per-task execution ───────────────────────────────────────────────────
    source "${PROJECT_ROOT}/../load_modules_2026.sh"
    source "${PROJECT_ROOT}/../venv_cameocg_jupiter2026/bin/activate"
    source "${PROJECT_ROOT}/../set_lammps_paths_2026.sh"

    echo "============================================================"
    echo "CAMEO CG MD Replica"
    echo "Config     : ${MD_CONFIG}"
    echo "Replica    : ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_MAX}"
    echo "Job ID     : ${SLURM_JOB_ID}"
    echo "Node       : ${SLURMD_NODENAME}"
    echo "============================================================"

    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

    source "${PROJECT_ROOT}/runs/registry_hook.sh"
    run_registry_start md "${MD_CONFIG}"
    run_registry_install_exit_trap

    python scripts/run_md.py "${MD_CONFIG}" "${SLURM_JOB_ID}" --replica "${SLURM_ARRAY_TASK_ID}"
else
    # ── First invocation: re-submit self as an array job ────────────────────
    sbatch "${SBATCH_ARGS[@]}" "${BASH_SOURCE[0]}" "${CONFIG}"
fi
