#!/bin/bash
#SBATCH --job-name=teacher_labels
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/teacher_labels_%j.out
#SBATCH --error=slurm/teacher_labels_%j.err

set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: sbatch scripts/submit_teacher_materialization.sh DATASET MANIFEST ENSEMBLE_SPEC OUTPUT [BATCH_SIZE]" >&2
    exit 2
fi

PROJECT_ROOT="${CAMEO_CG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET="$1"
MANIFEST="$2"
ENSEMBLE_SPEC="$3"
OUTPUT="$4"
BATCH_SIZE="${5:-128}"

source "${PROJECT_ROOT}/../load_modules_2026.sh"
source "${PROJECT_ROOT}/../venv_cameocg_jupiter2026/bin/activate"
source "${PROJECT_ROOT}/../set_lammps_paths_2026.sh"

mkdir -p "${PROJECT_ROOT}/slurm"
cd "${PROJECT_ROOT}"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

if [[ "${OUTPUT}" != /* ]]; then
    OUTPUT="${PROJECT_ROOT}/${OUTPUT}"
fi

source "${PROJECT_ROOT}/runs/registry_hook.sh"
run_registry_start teacher-materialization "" "${OUTPUT}"
run_registry_install_exit_trap

python -u scripts/materialize_direct_force_teacher.py \
    "${DATASET}" "${MANIFEST}" "${ENSEMBLE_SPEC}" "${OUTPUT}" \
    --batch-size "${BATCH_SIZE}"

