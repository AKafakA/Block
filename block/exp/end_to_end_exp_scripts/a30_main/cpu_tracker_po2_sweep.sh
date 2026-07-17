#!/bin/bash
# CPU tracker sweep for Po2-est + Po2-oracle.
# Matches revision Fanout cpu_tracker coverage: QPS {20, 24, 28, 32, 36} × {est, oracle}.
# Predictors launched with --enable_cpu_tracking. Output dirs include "cpu_tracking_true" for plotter compat.

set -u

TARGET_HOST="$(head -1 block/config/hosts)"
BATCH_CAP=48
CHUNK_SIZE=512
PREDICTOR_WORKERS=16
MAX_MODEL_LENGTH=4096
TIMEOUT_IN_SECONDS=1800
PREDICTOR_TIMEOUT_IN_SECONDS=1000
BRANCH_NAME="main"
MODEL="meta-llama/Llama-2-7b-hf"
PROFILING_SAMPLE_RATE=0.000
NUM_REQUEST=10000
AVAILABLE_INSTANCE="12"
MAX_SLO="0"
HOST_CONFIG_PATH='block/config/host_configs.json'
PRED_CONFIG_PATH='block/config/llama_config.json'
DATASET_PATH="~/Block/data/trace_data/sharegpt/generate/llama"
METRIC_TYPE=min_new_request_latency
N_SELECTED=2
QPS_LEVELS="20 24 28 32 36"
RESULTS_FILE="/tmp/a30_cpu_tracker_po2.log"
OUTPUT_PREFIX="cpu_tracker_po2"

echo "=== CPU tracker Po2 sweep starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"

deploy_with_cpu_tracking() {
    local metric=$1
    echo "[deploy] metric=$metric (CPU tracking ENABLED for all 16 predictors per node)"
    sh block/exp/reset.sh
    sleep 30
    nohup sh block/exp/run_exp_vllm.sh $BATCH_CAP $MODEL false 0 $MAX_MODEL_LENGTH true 1 $CHUNK_SIZE > /dev/null 2>&1 &
    sleep 90
    # Predictor scripts take 10th arg = ENABLE_CPU_TRACKING. Pass "true" to enable.
    for suffix in $(seq 1 7); do
        nohup sh block/exp/run_exp_predictor_${suffix}.sh $PRED_CONFIG_PATH $metric true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS true > /dev/null 2>&1 &
    done
    sleep 10
    for suffix in $(seq 8 $PREDICTOR_WORKERS); do
        nohup sh block/exp/run_exp_predictor_${suffix}.sh $PRED_CONFIG_PATH $metric true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS true > /dev/null 2>&1 &
    done
    sleep 60
    echo "[deploy] done"
}

launch_scheduler() {
    local metric=$1
    ssh -n -o ConnectTimeout=5 "$TARGET_HOST" "pkill -f global_scheduler/api_server" 2>&1 || true
    sleep 5
    nohup sh block/exp/run_exp_global_scheduler.sh "$TARGET_HOST" $N_SELECTED $N_SELECTED $metric $HOST_CONFIG_PATH 1 $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO false > /dev/null 2>&1 &
    sleep 20
}

run_cell() {
    local qps=$1 use_est=$2 mode_tag=$3
    local est_flag=""
    [ "$use_est" = "true" ] && est_flag="--use_estimated_response_lens"
    local output_dir="${OUTPUT_PREFIX}/sharegpt/${METRIC_TYPE}/qps_${qps}_n_${N_SELECTED}_len_estimated_${use_est}_cpu_tracking_true"
    parallel-ssh -i -t 0 --host "$TARGET_HOST" "cd Block && export PYTHONPATH=. && export HF_TOKEN= && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $NUM_REQUEST --dataset_type sharegpt --dataset_path $DATASET_PATH --qps $qps --backend block --log_filename benchmark.log --output_dir $output_dir --data_start_index 0 --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS $est_flag" > /dev/null 2>&1
    sleep 5
    echo "[cpu_tracker/${mode_tag}] QPS=$qps done at $(date -u +%H:%M:%SZ)" | tee -a "$RESULTS_FILE"
    # Move per-cell predictor logs (with CPU output) to per-cell dir for offline retrieval
    parallel-ssh -i -t 0 -h block/config/hosts "cd Block && mkdir -p experiment_output/$output_dir/predictor_logs && cp experiment_output/logs/predictor_*.log experiment_output/$output_dir/predictor_logs/. 2>/dev/null" > /dev/null 2>&1
}

# Po2-est sweep
deploy_with_cpu_tracking "$METRIC_TYPE"
launch_scheduler "$METRIC_TYPE"
echo "=== [cpu_tracker/po2_est] sweep starting ===" | tee -a "$RESULTS_FILE"
for qps in $QPS_LEVELS; do
    run_cell "$qps" "true" "po2_est"
done

# Po2-oracle sweep (same metric, no redeploy needed — only benchmark flag differs)
echo "=== [cpu_tracker/po2_oracle] sweep starting ===" | tee -a "$RESULTS_FILE"
launch_scheduler "$METRIC_TYPE"  # ensure clean scheduler state
for qps in $QPS_LEVELS; do
    run_cell "$qps" "false" "po2_oracle"
done

echo "=== CPU tracker Po2 sweep complete at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"
