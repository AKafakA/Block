#!/bin/bash
# ============================================================================
# Latency Training Data Collection Sweep for CARA XGBoost Predictor
# ============================================================================
#
# Sends 20k requests at each QPS level (9, 12, 15, 18, 21, 24, 27, 30, 33, 36)
# through the CARA scheduler, collecting latency records per request.
#
# Prerequisites:
#   - CARA system deployed (vLLM instances + coordinator on port 8200)
#   - Preprocessed training data at $REAL_DATA_PATH (for length distributions)
#   - Run from Block repo root
#
# Usage:
#   ./block/exp/end_to_end_exp_scripts/a100_supplementary/latency_sweep.sh
#
# Output:
#   data/cara/latency_training_data/qps_{rate}.jsonl   (latency records)
#   data/cara/latency_training_data/qps_{rate}.states.jsonl  (schedule_state snapshots)
#
# ============================================================================

set -e

# ============================================================================
# Configuration
# ============================================================================

# CARA coordinator
COORDINATOR_HOST="${COORDINATOR_HOST:-127.0.0.1}"
COORDINATOR_PORT="${COORDINATOR_PORT:-8200}"

# Workload
NUM_PROMPTS="${NUM_PROMPTS:-20000}"
CONCURRENCY="${CONCURRENCY:-64}"
QPS_LEVELS="${QPS_LEVELS:-9 12 15 18 21 24 27 30 33 36}"
# Real data: use real prompts, models produce natural outputs (max_tokens=1024)
REAL_DATA_PATH="${REAL_DATA_PATH-data/cara/training_data/cara_v3_all_training.json}"

# Instance polling for schedule_state capture
# Format: JSON dict mapping "http://host:port" -> "model_name"
# Set to empty string to disable polling
POLL_INSTANCES="${POLL_INSTANCES:-}"
POLL_INTERVAL="${POLL_INTERVAL:-0.5}"

# Output
OUTPUT_DIR="${OUTPUT_DIR:-data/cara/latency_training_data}"
LOG_DIR="${LOG_DIR:-experiment_output/logs/latency_sweep}"

# ============================================================================
# Setup
# ============================================================================

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/sweep_${TIMESTAMP}.log"
}

# ============================================================================
# Pre-flight checks
# ============================================================================

log "============================================================"
log "CARA Latency Training Data Sweep"
log "============================================================"
log "Coordinator: http://${COORDINATOR_HOST}:${COORDINATOR_PORT}"
log "Num prompts per QPS: ${NUM_PROMPTS}"
log "QPS levels: ${QPS_LEVELS}"
log "Concurrency: ${CONCURRENCY}"
log "Real data: ${REAL_DATA_PATH:-DISABLED (synthetic)}"
log "Instance polling: ${POLL_INSTANCES:-DISABLED}"
log "Output: ${OUTPUT_DIR}"
log "============================================================"

# Check coordinator is reachable
if ! curl -s --connect-timeout 5 "http://${COORDINATOR_HOST}:${COORDINATOR_PORT}/health" >/dev/null 2>&1; then
    log "WARNING: Coordinator at ${COORDINATOR_HOST}:${COORDINATOR_PORT} not reachable."
    log "Make sure CARA is deployed before running this sweep."
    log "Continuing anyway (requests will fail if coordinator is down)..."
fi

# Check real data exists
if [ -n "$REAL_DATA_PATH" ] && [ ! -f "$REAL_DATA_PATH" ]; then
    log "ERROR: Real data file not found: ${REAL_DATA_PATH}"
    log "Run preprocessing first or set REAL_DATA_PATH= to use synthetic."
    exit 1
fi

# ============================================================================
# Sweep
# ============================================================================

TOTAL_QPS=$(echo "$QPS_LEVELS" | wc -w)
CURRENT=0

for QPS in $QPS_LEVELS; do
    CURRENT=$((CURRENT + 1))
    OUTPUT_FILE="${OUTPUT_DIR}/qps_${QPS}.jsonl"

    log ""
    log "------------------------------------------------------------"
    log "[${CURRENT}/${TOTAL_QPS}] QPS=${QPS} (${NUM_PROMPTS} requests)"
    log "------------------------------------------------------------"

    # Build command
    CMD="python -m block.predictor.cara.offline_training.generate_latency_benchmark"
    CMD="$CMD --host ${COORDINATOR_HOST} --port ${COORDINATOR_PORT}"
    CMD="$CMD --num-prompts ${NUM_PROMPTS}"
    CMD="$CMD --request-rate ${QPS}"
    CMD="$CMD --concurrency ${CONCURRENCY}"
    CMD="$CMD --output ${OUTPUT_FILE}"
    CMD="$CMD --seed $((42 + QPS))"  # Different seed per QPS for variety

    # Use real data if available
    if [ -n "$REAL_DATA_PATH" ]; then
        CMD="$CMD --real-data ${REAL_DATA_PATH}"
    fi

    # Add instance polling if configured
    if [ -n "$POLL_INSTANCES" ]; then
        CMD="$CMD --poll-instances '${POLL_INSTANCES}'"
        CMD="$CMD --poll-interval ${POLL_INTERVAL}"
    fi

    log "Command: $CMD"

    START_TIME=$(date +%s)
    eval "$CMD" 2>&1 | tee -a "$LOG_DIR/qps_${QPS}_${TIMESTAMP}.log"
    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    if [ $EXIT_CODE -eq 0 ]; then
        # Count records
        NUM_RECORDS=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo "0")
        NUM_SUCCESS=$(grep -c '"success": true' "$OUTPUT_FILE" 2>/dev/null || echo "0")
        log "DONE: QPS=${QPS}, ${NUM_RECORDS} records (${NUM_SUCCESS} success), ${DURATION}s elapsed"
    else
        log "FAILED: QPS=${QPS}, exit code ${EXIT_CODE}, ${DURATION}s elapsed"
    fi

    # Brief cooldown between QPS levels to let the system stabilize
    if [ $CURRENT -lt $TOTAL_QPS ]; then
        COOLDOWN=30
        log "Cooling down ${COOLDOWN}s before next QPS level..."
        sleep $COOLDOWN
    fi
done

# ============================================================================
# Summary
# ============================================================================

log ""
log "============================================================"
log "SWEEP COMPLETE"
log "============================================================"
log "Output files:"
for QPS in $QPS_LEVELS; do
    FILE="${OUTPUT_DIR}/qps_${QPS}.jsonl"
    if [ -f "$FILE" ]; then
        LINES=$(wc -l < "$FILE")
        SUCCESS=$(grep -c '"success": true' "$FILE" 2>/dev/null || echo "0")
        log "  qps_${QPS}.jsonl: ${LINES} records (${SUCCESS} successful)"
    else
        log "  qps_${QPS}.jsonl: MISSING"
    fi
done

TOTAL_RECORDS=0
for f in "${OUTPUT_DIR}"/qps_*.jsonl; do
    if [ -f "$f" ]; then
        TOTAL_RECORDS=$((TOTAL_RECORDS + $(wc -l < "$f")))
    fi
done
log ""
log "Total records across all QPS levels: ${TOTAL_RECORDS}"
log "Log: ${LOG_DIR}/sweep_${TIMESTAMP}.log"
