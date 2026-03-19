#!/bin/bash
# ============================================================================
# Per-Instance Latency Training Data Collection for CARA
# ============================================================================
#
# Targets individual vLLM instances (NOT the coordinator) to collect clean
# per-model, per-GPU latency data without broadcasting interference.
#
# Each (model, GPU) gets its own QPS sweep, producing separate JSONL files
# for XGBoost predictor training.
#
# Usage:
#   ./block/exp/end_to_end_exp_scripts/a100_supplementary/latency_sweep_per_instance.sh
#
# Output:
#   data/cara/latency_training_data/{model}_{gpu}/qps_{rate}.jsonl
#
# ============================================================================

set -e

# ============================================================================
# Configuration
# ============================================================================

NUM_PROMPTS="${NUM_PROMPTS:-5000}"
CONCURRENCY="${CONCURRENCY:-32}"

# Instance definitions: "ip:port model_name gpu_type qps_levels"
# Each instance gets its own QPS sweep
INSTANCES=(
    # 72B on A100x4 (TP=4) — lower QPS range due to large model
    "128.105.146.44:8000 Qwen/Qwen2.5-72B a100 3 5 7 9 12 15"
    "128.105.146.72:8000 Qwen/Qwen2.5-72B a100 3 5 7 9 12 15"
    # 14B on V100x4 (TP=4)
    "128.105.146.25:8000 Qwen/Qwen2.5-14B v100 5 9 12 15 18 21 24"
    # 7B on A30x1
    "128.105.146.38:8000 Qwen/Qwen2.5-7B a30 9 12 15 18 21 24 27 30"
    # 3B on A30x1
    "128.105.146.37:8000 Qwen/Qwen2.5-3B a30 12 15 18 21 24 27 30 33 36"
)

# Output
OUTPUT_DIR="${OUTPUT_DIR:-data/cara/latency_training_data}"
LOG_DIR="${LOG_DIR:-experiment_output/logs/latency_sweep}"

# ============================================================================
# Setup
# ============================================================================

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/sweep_${TIMESTAMP}.log"
}

# ============================================================================
# Sweep
# ============================================================================

log "============================================================"
log "CARA Per-Instance Latency Sweep"
log "============================================================"
log "Num prompts per QPS: ${NUM_PROMPTS}"
log "Concurrency: ${CONCURRENCY}"
log "Output: ${OUTPUT_DIR}"
log "============================================================"

TOTAL_INSTANCE=${#INSTANCES[@]}
INSTANCE_NUM=0

for INSTANCE_DEF in "${INSTANCES[@]}"; do
    # Parse instance definition
    read -r ADDR MODEL GPU QPS_LIST <<< "$INSTANCE_DEF"
    HOST=$(echo "$ADDR" | cut -d: -f1)
    PORT=$(echo "$ADDR" | cut -d: -f2)
    INSTANCE_NUM=$((INSTANCE_NUM + 1))

    # Create output dir for this model+gpu combo
    # Sanitize model name for directory: Qwen/Qwen2.5-72B -> qwen2.5-72b
    MODEL_DIR=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
    INSTANCE_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_DIR}_${GPU}"
    mkdir -p "$INSTANCE_OUTPUT_DIR"

    log ""
    log "============================================================"
    log "[Instance ${INSTANCE_NUM}/${TOTAL_INSTANCE}] ${MODEL} on ${GPU} (${ADDR})"
    log "============================================================"

    # Check instance is reachable
    if ! curl -s --connect-timeout 5 "http://${ADDR}/v1/models" >/dev/null 2>&1; then
        log "WARNING: Instance ${ADDR} not reachable, skipping"
        continue
    fi

    QPS_ARRAY=($QPS_LIST)
    TOTAL_QPS=${#QPS_ARRAY[@]}
    CURRENT=0

    for QPS in "${QPS_ARRAY[@]}"; do
        CURRENT=$((CURRENT + 1))
        OUTPUT_FILE="${INSTANCE_OUTPUT_DIR}/qps_${QPS}.jsonl"

        log ""
        log "  [${CURRENT}/${TOTAL_QPS}] QPS=${QPS} -> ${OUTPUT_FILE}"

        CMD="python -m block.predictor.cara.offline_training.generate_latency_benchmark"
        CMD="$CMD --host ${HOST} --port ${PORT}"
        CMD="$CMD --num-prompts ${NUM_PROMPTS}"
        CMD="$CMD --request-rate ${QPS}"
        CMD="$CMD --concurrency ${CONCURRENCY}"
        CMD="$CMD --output ${OUTPUT_FILE}"
        CMD="$CMD --seed $((42 + QPS))"

        # Add schedule_trace polling for this instance
        CMD="$CMD --poll-instances '{\"http://${ADDR}\": \"${MODEL}\"}'"
        CMD="$CMD --poll-interval 0.5"

        log "  Command: $CMD"

        START_TIME=$(date +%s)
        eval "$CMD" 2>&1 | tee -a "$LOG_DIR/${MODEL_DIR}_${GPU}_qps${QPS}_${TIMESTAMP}.log"
        EXIT_CODE=${PIPESTATUS[0]}
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        if [ $EXIT_CODE -eq 0 ]; then
            NUM_RECORDS=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo "0")
            log "  DONE: ${NUM_RECORDS} records, ${DURATION}s elapsed"
        else
            log "  FAILED: exit code ${EXIT_CODE}, ${DURATION}s elapsed"
        fi

        # Cooldown between QPS levels
        if [ $CURRENT -lt $TOTAL_QPS ]; then
            COOLDOWN=15
            log "  Cooling down ${COOLDOWN}s..."
            sleep $COOLDOWN
        fi
    done

    # Longer cooldown between instances
    if [ $INSTANCE_NUM -lt $TOTAL_INSTANCE ]; then
        log "Instance cooldown 30s before next model..."
        sleep 30
    fi
done

# ============================================================================
# Summary
# ============================================================================

log ""
log "============================================================"
log "SWEEP COMPLETE"
log "============================================================"

TOTAL_RECORDS=0
for INSTANCE_DEF in "${INSTANCES[@]}"; do
    read -r ADDR MODEL GPU QPS_LIST <<< "$INSTANCE_DEF"
    MODEL_DIR=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
    INSTANCE_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_DIR}_${GPU}"

    log "  ${MODEL} (${GPU}):"
    for QPS in $QPS_LIST; do
        FILE="${INSTANCE_OUTPUT_DIR}/qps_${QPS}.jsonl"
        if [ -f "$FILE" ]; then
            LINES=$(wc -l < "$FILE")
            TOTAL_RECORDS=$((TOTAL_RECORDS + LINES))
            log "    qps_${QPS}: ${LINES} records"
        else
            log "    qps_${QPS}: MISSING"
        fi
    done
done

log ""
log "Total records: ${TOTAL_RECORDS}"
log "Log: ${LOG_DIR}/sweep_${TIMESTAMP}.log"
