#!/bin/bash
# Predictor watchdog — runs verify_predictors.sh every INTERVAL seconds indefinitely.
# Used during Phase 3 (burstiness_exp.sh / error_heatmap_exp.sh) since those scripts
# do per-cell reset+deploy via experiment.sh and each deploy has a small chance of
# missing predictors due to parallel-ssh race.
#
# Usage: nohup bash predictor_watchdog.sh <metric> <batch_cap> <interval_seconds> > /tmp/a30_pred_watchdog.log 2>&1 &

set -u
METRIC="${1:-min_new_request_latency}"
BATCH="${2:-48}"
INTERVAL="${3:-60}"
CFG="${4:-block/config/llama_config.json}"
TIMEOUT="${5:-1000}"
WORKERS=16

echo "[watchdog] starting $(date -u +%Y-%m-%dT%H:%M:%SZ) metric=$METRIC batch=$BATCH interval=${INTERVAL}s"
while true; do
    bash block/exp/end_to_end_exp_scripts/a30_main/verify_predictors.sh \
        "$CFG" "$METRIC" "$BATCH" "$WORKERS" "$TIMEOUT" 2>&1 | grep -E 'attempt.*missing|confirmed|WARNING' | head -20
    sleep "$INTERVAL"
done
