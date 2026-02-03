#!/bin/bash
# =============================================================================
# Scaling Sweep Submission Script
# Submits training jobs across a range of device counts for scaling analysis.
#
# Each job reuses run_training.sh (same env setup, same train.py) but with
# --nodes and CUDA_VISIBLE_DEVICES overridden via sbatch command line.
# run_training.sh preserves CUDA_VISIBLE_DEVICES if already set, and
# train.py derives local_device_ids from it automatically.
#
# Strategy (from SCALING_ANALYSIS.md):
#   CUDA_VISIBLE_DEVICES=0       -> 1,2,3,4 nodes -> 1,2,3,4 devices
#   CUDA_VISIBLE_DEVICES=0,1     -> 3,4,5,6 nodes -> 6,8,10,12 devices
#   CUDA_VISIBLE_DEVICES=0,1,2   -> 6 nodes       -> 18 devices
#   CUDA_VISIBLE_DEVICES=0,1,2,3 -> 4,5,6 nodes   -> 16,20,24 devices
#
# Usage (from clean_code_base/):
#   ./scripts/run_scaling_sweep.sh config_timing_test.yaml
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEAN_CODE_BASE_DIR="$(dirname "$SCRIPT_DIR")"
WORKER_SCRIPT="${SCRIPT_DIR}/run_training.sh"

CONFIG_FILE="${1:-config_timing_test.yaml}"

# Verify worker script exists
if [[ ! -f "$WORKER_SCRIPT" ]]; then
    echo "ERROR: Worker script not found: $WORKER_SCRIPT"
    exit 1
fi

# Verify config exists (check both relative to CWD and clean_code_base)
if [[ ! -f "$CONFIG_FILE" ]] && [[ ! -f "${CLEAN_CODE_BASE_DIR}/$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Timestamped log directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${CLEAN_CODE_BASE_DIR}/scaling_results/run_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Scaling Sweep Submission"
echo "============================================================"
echo "Timestamp:      $TIMESTAMP"
echo "Config:         $CONFIG_FILE"
echo "Log dir:        $LOG_DIR"
echo "Worker script:  $WORKER_SCRIPT"
echo ""
echo "Device counts:  1, 2, 3, 4, 6, 8, 10, 12, 16, 18, 20, 24"
echo "============================================================"
echo ""

# All (nodes, gpus_per_node) configurations — 12 jobs total
# Command-line --nodes overrides #SBATCH --nodes=1 in run_training.sh
# --export=ALL,CUDA_VISIBLE_DEVICES=... overrides the default in the script

# declare -a CONFIGS=(
#     "1 1"    # 1 device
#     "2 1"    # 2 devices
#     "3 1"    # 3 devices
#     "4 1"    # 4 devices
#     "3 2"    # 6 devices
#     "4 2"    # 8 devices
#     "5 2"    # 10 devices
#     "6 2"    # 12 devices
#     "4 4"    # 16 devices
#     "6 3"    # 18 devices
#     "5 4"    # 20 devices
#     "6 4"    # 24 devices
# )

declare -a CONFIGS=(
    "2 2"    # 6 devices
)

declare -a JOB_IDS

# Exported once: no commas in value, safe for --export=ALL
export JAX_COORDINATOR_HEARTBEAT_TIMEOUT_SECS=120

for config in "${CONFIGS[@]}"; do
    read -r NODES GPUS <<< "$config"
    DEVICES=$((NODES * GPUS))

    # Build CUDA_VISIBLE_DEVICES: "0" / "0,1" / "0,1,2" / "0,1,2,3"
    # NOTE: cannot pass via --export=ALL,CUDA_VISIBLE_DEVICES=0,1,2 because
    # SLURM parses the commas as field separators, truncating to just "0".
    # Export in the shell instead; --export=ALL propagates it.
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS - 1)))

    echo "Submitting: $NODES nodes x $GPUS GPUs = $DEVICES devices  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)..."

    JOB_ID=$(sbatch \
        --nodes=$NODES \
        --job-name="scale_${DEVICES}dev" \
        --output="${LOG_DIR}/job_${DEVICES}dev_%j.out" \
        --error="${LOG_DIR}/job_${DEVICES}dev_%j.err" \
        --exclusive \
        --export=ALL \
        "$WORKER_SCRIPT" "$CONFIG_FILE" \
        | awk '{print $4}')

    JOB_IDS+=("$JOB_ID")
    echo "  -> Job ID: $JOB_ID"
done

echo ""
echo "============================================================"
echo "All 12 jobs submitted"
echo "============================================================"
echo ""
echo "Job summary:"
idx=0
for config in "${CONFIGS[@]}"; do
    read -r NODES GPUS <<< "$config"
    DEVICES=$((NODES * GPUS))
    printf "  %2d devices  (%d nodes x %d GPUs):  job %s\n" "$DEVICES" "$NODES" "$GPUS" "${JOB_IDS[$idx]}"
    ((idx++))
done
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     $LOG_DIR"
echo ""

# Save job manifest
{
    echo "Scaling Sweep: $TIMESTAMP"
    echo "Config: $CONFIG_FILE"
    echo ""
    echo "Jobs:"
    idx=0
    for config in "${CONFIGS[@]}"; do
        read -r NODES GPUS <<< "$config"
        DEVICES=$((NODES * GPUS))
        echo "  ${DEVICES} devices (${NODES} nodes x ${GPUS} GPUs): job ${JOB_IDS[$idx]}"
        ((idx++))
    done
} > "${LOG_DIR}/job_info.txt"

echo "Job info saved to: ${LOG_DIR}/job_info.txt"
