#!/bin/bash
# SLURM submission script for CAMEO CG JAX-MD simulations.
#
# Usage:
#   sbatch scripts/submit_md.sh [md_config.yaml]
#   sbatch scripts/submit_md.sh configs/md_1pro_4zoh.yaml
#
#SBATCH --job-name=cameo_md
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/cameo_md_%j.out
#SBATCH --error=slurm/cameo_md_%j.err

set -euo pipefail

CONFIG="${1:-configs/md_1pro_4zoh.yaml}"
PROJECT_ROOT="${CAMEO_CG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

echo "============================================================"
echo "CAMEO CG JAX-MD Simulation"
echo "Config      : $CONFIG"
echo "Project root: $PROJECT_ROOT"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "============================================================"

# Load environment (same as training).
source "$PROJECT_ROOT/../load_modules_2026.sh"
source "$PROJECT_ROOT/../venv_cameocg_jupiter2026/bin/activate"
source "$PROJECT_ROOT/../set_lammps_paths_2026.sh"

mkdir -p "$PROJECT_ROOT/slurm"
cd "$PROJECT_ROOT"

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

REGISTRY_PYTHON="${RUN_REGISTRY_PYTHON:-$(command -v python3 || command -v python)}"
MD_OUTPUT_DIR="$("${REGISTRY_PYTHON}" -c '
from pathlib import Path
import sys, yaml
config = Path(sys.argv[1])
root = Path(sys.argv[2])
data = yaml.safe_load(config.read_text()) or {}
raw = (data.get("md") or {}).get("output_dir", "local_work/md_runs")
path = Path(str(raw))
print((path if path.is_absolute() else root / path).resolve())
' "${CONFIG}" "${PROJECT_ROOT}")"

source "${PROJECT_ROOT}/runs/registry_hook.sh"
run_registry_start md "${CONFIG}" "${MD_OUTPUT_DIR}"

submit_md_registry_exit() {
    local rc=$?
    trap - EXIT
    run_registry_finish "${rc}"
    exit "${rc}"
}
trap submit_md_registry_exit EXIT

python scripts/run_md.py "$CONFIG" "$SLURM_JOB_ID"
