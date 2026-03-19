#!/bin/bash
# Sweep generation parameter configs for broadcasting (6 configs)
# Skips cases where both rep_penalty AND freq_penalty are enabled

REMOTE_HOST="asdwb@d8545-10s10301.wisc.cloudlab.us"
MODELS="Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B Qwen/Qwen2.5-72B"
NUM_PROMPTS=500
RATE="inf"
DATASET_PATH="data/cara/best-route-extension.jsonl"

# 6 configs: temp={0,0.1} x {rep=1.15, freq=1.2, none} — skip both enabled
CONFIGS=(
    "0.0  1.0   0.0"
    "0.0  1.15  0.0"
    "0.0  1.0   1.2"
    "0.1  1.0   0.0"
    "0.1  1.15  0.0"
    "0.1  1.0   1.2"
)

echo "=========================================="
echo "Broadcasting Parameter Sweep (${#CONFIGS[@]} configs)"
echo "=========================================="

for i in "${!CONFIGS[@]}"; do
    read -r TEMP REP FREQ <<< "${CONFIGS[$i]}"
    SUFFIX="sweep_t${TEMP}_r${REP}_f${FREQ}"

    # Skip if result already exists
    EXISTING=$(ssh ${REMOTE_HOST} "ls Block/experiment_output/cara_broadcast_training_data/broadcast_custom_500prompts_${SUFFIX}_*.json 2>/dev/null | wc -l")
    if [ "${EXISTING}" -gt 0 ]; then
        echo ""
        echo "=== Config $((i+1))/${#CONFIGS[@]}: temp=${TEMP} rep=${REP} freq=${FREQ} — SKIPPING (already exists) ==="
        continue
    fi

    echo ""
    echo "=== Config $((i+1))/${#CONFIGS[@]}: temp=${TEMP} rep=${REP} freq=${FREQ} ==="

    # Kill any existing CARA server first
    ssh ${REMOTE_HOST} "pkill -f cara_serve.py" 2>/dev/null || true
    sleep 3

    TEMPERATURE=${TEMP} REPETITION_PENALTY=${REP} FREQUENCY_PENALTY=${FREQ} \
        bash block/exp/cara/run_broadcasting_remote.sh \
        ${REMOTE_HOST} \
        "${MODELS}" \
        custom ${NUM_PROMPTS} ${RATE} "${SUFFIX}" \
        ${DATASET_PATH}

    if [ $? -ne 0 ]; then
        echo "WARNING: Config $((i+1)) failed, continuing to next..."
    fi

    echo "--- Config $((i+1)) done ---"
done

# Kill server after sweep
ssh ${REMOTE_HOST} "pkill -f cara_serve.py" 2>/dev/null || true

echo ""
echo "=========================================="
echo "Sweep complete!"
echo "=========================================="
