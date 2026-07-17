#!/bin/bash
# Deploy all 16 predictors SEQUENTIALLY (1 at a time, 2s apart) to avoid concurrent-startup race.
# Used as a fallback when the gate detects <16/16 after experiment.sh's racy parallel deploy.
#
# Args: same as run_exp_predictor_N.sh:
#   $1 CONFIG_PATH    $2 METRIC_TYPE           $3 ENABLE_TIME_ESTIMATION
#   $4 BATCH_CAP      $5 ENABLE_CHUNKED_PREFILL $6 NUM_WORKERS
#   $7 BRANCH_NAME    $8 BATCH_SIZE_THRESHOLD   $9 PREDICTOR_TIMEOUT

CONFIG_PATH=$1
METRIC_TYPE=$2
ENABLE_TIME_ESTIMATION=$3
BATCH_CAP=$4
ENABLE_CHUNKED_PREFILL=$5
NUM_WORKERS=$6
BRANCH_NAME=$7
BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION=$8
PREDICTOR_TIMEOUT=$9

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Killing all predictors cluster-wide before sequential redeploy..."
parallel-ssh -i -t 30 -h block/config/hosts "pkill -9 -f 'predictor/api_server' 2>/dev/null; sleep 1; ps aux | grep -c 'predictor/api_server' | grep -v grep" 2>&1 | tail -3
sleep 10

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Sequentially deploying 16 predictors (2s between each)..."
for n in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
  sh block/exp/run_exp_predictor_$n.sh \
    "$CONFIG_PATH" "$METRIC_TYPE" "$ENABLE_TIME_ESTIMATION" "$BATCH_CAP" \
    "$ENABLE_CHUNKED_PREFILL" "$NUM_WORKERS" "$BRANCH_NAME" \
    "$BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION" "$PREDICTOR_TIMEOUT" > /dev/null 2>&1
  sleep 2
done
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Sequential deploy dispatched, waiting 60s for settlement..."
sleep 60
