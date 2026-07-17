#!/bin/bash
# Gated mirror of main_experiment.sh — ENFORCES 16/16 predictors on all 12 nodes.
# Root cause: experiment.sh does 7+9 parallel predictor startups per node, causing a
# concurrent-startup race (~8% chance one predictor silently dies per deploy).
#
# Per-run flow:
#   1. experiment.sh RESTART_VLLM=true RUN_EXP=false    # deploy only (racy parallel)
#   2. Gate check 16/16 on all 12 nodes
#   3. If fail: sequential redeploy (1 predictor at a time, 2s gap) + re-gate
#   4. If still fail: ABORT ENTIRE SCRIPT
#   5. Start a 30s mid-run monitor in background (kills benchmark if any node drops below 16)
#   6. experiment.sh RESTART_VLLM=false RUN_EXP=true    # benchmark only
#   7. Kill monitor, check result
#
# Env overrides: SCHEDULER_LIST, N_SELECTED, OUTPUT_DIR_PREFIX, QPS_LIST
# Logs: /tmp/a30_gate_checks.log  + /tmp/a30_run_monitor.log

START_INDEX=0
BATCH_CAP=48
PREDICTOR_WORKERS=16
GLOBAL_SCHEDULER_WORKERS=1
BACKEND_WORKERS=1
MAX_MODEL_LENGTH=4096
CHUNK_SIZE=512
TIMEOUT_IN_SECONDS=1800
PREDICTOR_TIMEOUT_IN_SECONDS=1000
BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION="0"
BRANCH_NAME="main"
USE_PROCESS_FOR_FRONTEND=true
UPDATE_BLOCK_CODE=false
UPDATE_VLLM_CODE=false

ENABLE_CHUNKED_PREFILL="true"
MODEL="meta-llama/Llama-2-7b-hf"
MODEL_TYPE="llama"
DATASET_NAME="sharegpt"
DATASET_PATH="~/Block/data/trace_data/sharegpt/generate/llama"

# Paper schedulers + QPS sweep (override via env)
SCHEDULER_LIST="${SCHEDULER_LIST:-min_new_request_latency min_lunmnix_load min_infass_load round_robin min_qpm_load random}"
QPS_LIST="${QPS_LIST:-20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36}"
PROFILING_SAMPLE_RATE=0.000
USE_FOR_PROFILING_ONLY=false
NUM_REQUEST=10000
KEEP_ALL_METRICS=false
N_SELECTED="${N_SELECTED:-12}"  # 12=Fanout (A30 has 12 instances), 2=Po2 (applies to Block only)
OUTPUT_DIR_PREFIX="${OUTPUT_DIR_PREFIX:-main_gated}"
AVAILABLE_INSTANCE="12"
ENABLE_PREEMPTIVE_AUTO_PROVISIONING="false"
MAX_SLO="0"

GATE_LOG="/tmp/a30_gate_checks.log"
MON_LOG="/tmp/a30_run_monitor.log"
GATE_SCRIPT="block/exp/end_to_end_exp_scripts/a30_main/verify_predictors_16.sh"
SEQ_DEPLOY="block/exp/end_to_end_exp_scripts/a30_main/redeploy_predictors_sequential.sh"

run_one_with_gate() {
  local scheduler=$1
  local use_estimation_len=$2
  local qps=$3
  local n=$4
  local attempt=${5:-1}

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] === RUN scheduler=$scheduler est=$use_estimation_len qps=$qps n=$n attempt=$attempt ===" >> $GATE_LOG

  # Step 1: Deploy only
  sh block/exp/experiment.sh \
    $scheduler $NUM_REQUEST true $BATCH_CAP \
    $DATASET_NAME $DATASET_PATH $DATASET_NAME true $KEEP_ALL_METRICS $START_INDEX \
    $MODEL $MODEL_TYPE $MAX_MODEL_LENGTH $ENABLE_CHUNKED_PREFILL \
    $PREDICTOR_WORKERS $GLOBAL_SCHEDULER_WORKERS $BACKEND_WORKERS $CHUNK_SIZE \
    $qps $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $n \
    $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $USE_FOR_PROFILING_ONLY \
    $PREDICTOR_TIMEOUT_IN_SECONDS $USE_PROCESS_FOR_FRONTEND \
    $UPDATE_BLOCK_CODE $UPDATE_VLLM_CODE false \
    $use_estimation_len $OUTPUT_DIR_PREFIX \
    $AVAILABLE_INSTANCE $MAX_SLO $ENABLE_PREEMPTIVE_AUTO_PROVISIONING >> $GATE_LOG 2>&1

  sleep 30  # settle

  # Step 2: Gate check
  if ! sh $GATE_SCRIPT >> $GATE_LOG 2>&1; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GATE FAIL after racy parallel deploy, running sequential redeploy..." >> $GATE_LOG
    # Determine ENABLE_TIME_ESTIMATION arg for predictor script (matches experiment.sh internal logic)
    local enable_time_est="true"
    if [ "$use_estimation_len" = "false" ]; then enable_time_est="false"; fi
    sh $SEQ_DEPLOY block/config/llama_config.json $scheduler $enable_time_est $BATCH_CAP $ENABLE_CHUNKED_PREFILL $PREDICTOR_WORKERS $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $PREDICTOR_TIMEOUT_IN_SECONDS >> $GATE_LOG 2>&1
    if ! sh $GATE_SCRIPT >> $GATE_LOG 2>&1; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GATE FAIL AFTER SEQUENTIAL REDEPLOY — ABORTING" >> $GATE_LOG
      exit 99
    fi
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GATE PASS after sequential redeploy" >> $GATE_LOG
  else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GATE PASS after racy deploy (first try)" >> $GATE_LOG
  fi

  # Step 3: Start mid-run monitor (background)
  (
    while true; do
      sleep 30
      if ! sh $GATE_SCRIPT >> $MON_LOG 2>&1; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] MID-RUN FAIL during scheduler=$scheduler qps=$qps est=$use_estimation_len — killing benchmark" >> $MON_LOG
        pkill -f "benchmark_serving.py" 2>/dev/null
        break
      fi
    done
  ) &
  MONITOR_PID=$!

  # Step 4: Run benchmark only
  sh block/exp/experiment.sh \
    $scheduler $NUM_REQUEST false $BATCH_CAP \
    $DATASET_NAME $DATASET_PATH $DATASET_NAME true $KEEP_ALL_METRICS $START_INDEX \
    $MODEL $MODEL_TYPE $MAX_MODEL_LENGTH $ENABLE_CHUNKED_PREFILL \
    $PREDICTOR_WORKERS $GLOBAL_SCHEDULER_WORKERS $BACKEND_WORKERS $CHUNK_SIZE \
    $qps $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $n \
    $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $USE_FOR_PROFILING_ONLY \
    $PREDICTOR_TIMEOUT_IN_SECONDS $USE_PROCESS_FOR_FRONTEND \
    $UPDATE_BLOCK_CODE $UPDATE_VLLM_CODE true \
    $use_estimation_len $OUTPUT_DIR_PREFIX \
    $AVAILABLE_INSTANCE $MAX_SLO $ENABLE_PREEMPTIVE_AUTO_PROVISIONING >> $GATE_LOG 2>&1

  # Step 5: Stop monitor
  kill $MONITOR_PID 2>/dev/null
  wait $MONITOR_PID 2>/dev/null

  # Check if mid-run monitor killed the benchmark — if so, retry (up to 2 attempts total)
  if grep -q "MID-RUN FAIL during scheduler=$scheduler qps=$qps est=$use_estimation_len" $MON_LOG 2>/dev/null; then
    if [ $attempt -lt 2 ]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Benchmark was killed by monitor. Retrying once..." >> $GATE_LOG
      run_one_with_gate "$scheduler" "$use_estimation_len" "$qps" "$n" 2
    else
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Benchmark killed twice for same QPS. ABORTING." >> $GATE_LOG
      exit 98
    fi
  fi
}

# Iterate all combos (mirrors main_experiment.sh outer structure)
for scheduler in $SCHEDULER_LIST; do
  if [ "$scheduler" = "min_new_request_latency" ]; then
    USE_LEN_LIST="true false"
    N=$N_SELECTED
  else
    USE_LEN_LIST="false"
    N=12  # baselines use N=12
  fi
  for use_estimation_len in $USE_LEN_LIST; do
    for qps in $QPS_LIST; do
      run_one_with_gate $scheduler $use_estimation_len $qps $N
    done
  done
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] main_experiment_gated complete" >> $GATE_LOG
