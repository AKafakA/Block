#!/bin/bash
# Po4-est + Po8-est sparse sweep for Section 6.6 N-tunable enhancement.
# Single deploy, scheduler relaunch per N. Dense near QPS=32 for capacity refinement.

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
QPS_LEVELS="20 24 28 30 31 32 33 34 36"
RESULTS_FILE="/tmp/a30_po4po8_sweep.log"
OUTPUT_PREFIX="po4po8_sweep"

echo "=== Po4+Po8 sweep starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"

# --- (100,0) heatmap rerun (third deploy, verify flip) ---
HEATMAP_RESULTS=/tmp/a30_phase4_2_simple_results.txt
echo "[100-0 REDO] fresh deploy + (100,0) rerun (third-time check)" | tee -a "$HEATMAP_RESULTS"

deploy_for_heatmap() {
    sh block/exp/reset.sh
    sleep 30
    nohup sh block/exp/run_exp_vllm.sh $BATCH_CAP $MODEL false 0 $MAX_MODEL_LENGTH true 1 $CHUNK_SIZE > /dev/null 2>&1 &
    sleep 90
    for s in $(seq 1 7); do
        nohup sh block/exp/run_exp_predictor_${s}.sh $PRED_CONFIG_PATH $METRIC_TYPE true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
    done
    sleep 10
    for s in $(seq 8 $PREDICTOR_WORKERS); do
        nohup sh block/exp/run_exp_predictor_${s}.sh $PRED_CONFIG_PATH $METRIC_TYPE true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
    done
    sleep 60
    bash block/exp/end_to_end_exp_scripts/a30_main/verify_predictors.sh "$PRED_CONFIG_PATH" "$METRIC_TYPE" "$BATCH_CAP" "$PREDICTOR_WORKERS" "$PREDICTOR_TIMEOUT_IN_SECONDS" || true
}

deploy_for_heatmap
ssh -n -o ConnectTimeout=5 "$TARGET_HOST" "pkill -f global_scheduler/api_server" 2>&1 || true
sleep 5
# length_error_pct=100, latency_error_pct=0 (args 14 and 15 to scheduler)
nohup sh block/exp/run_exp_global_scheduler.sh "$TARGET_HOST" 2 2 $METRIC_TYPE $HOST_CONFIG_PATH 1 $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO false 100 0 > /dev/null 2>&1 &
sleep 20
HEATMAP_DIR="error_heatmap_po2_rerun/sharegpt/${METRIC_TYPE}/qps_32_len_err_100_lat_err_0_REDO2"
parallel-ssh -i -t 0 --host "$TARGET_HOST" "cd Block && export PYTHONPATH=. && export HF_TOKEN= && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $NUM_REQUEST --dataset_type sharegpt --dataset_path $DATASET_PATH --qps 32 --backend block --log_filename benchmark.log --output_dir $HEATMAP_DIR --data_start_index 0 --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS --use_estimated_response_lens" > /dev/null 2>&1
sleep 5
hnpz=/tmp/heatmap_redo2_100_0.npz
rsync -az "asdwb@d7525-10s10327.wisc.cloudlab.us:~/Block/experiment_output/$HEATMAP_DIR/benchmark_all_metrics.npz" "$hnpz" > /dev/null 2>&1
python3 -c "import numpy as np; d=np.load('$hnpz', allow_pickle=True); req=d['request_latencies']; ttft=d['prefill_token_latencies']; print(f'[(100,0) REDO2] e2e_mean={float(np.mean(req)):.0f}, e2e_P99={float(np.percentile(req,99)):.0f}, TTFT_P99={float(np.percentile(ttft,99)):.0f} vs baseline 14737')" | tee -a "$HEATMAP_RESULTS"
echo "[100-0 REDO] done at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$HEATMAP_RESULTS"

deploy_fresh() {
    echo "[deploy] metric=$METRIC_TYPE batch=$BATCH_CAP chunk=$CHUNK_SIZE"
    sh block/exp/reset.sh
    sleep 30
    nohup sh block/exp/run_exp_vllm.sh $BATCH_CAP $MODEL false 0 $MAX_MODEL_LENGTH true 1 $CHUNK_SIZE > /dev/null 2>&1 &
    sleep 90
    for suffix in $(seq 1 7); do
        nohup sh block/exp/run_exp_predictor_${suffix}.sh $PRED_CONFIG_PATH $METRIC_TYPE true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
    done
    sleep 10
    for suffix in $(seq 8 $PREDICTOR_WORKERS); do
        nohup sh block/exp/run_exp_predictor_${suffix}.sh $PRED_CONFIG_PATH $METRIC_TYPE true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
    done
    sleep 60
    echo "[deploy] done"
}

launch_scheduler() {
    local N=$1
    ssh -n -o ConnectTimeout=5 "$TARGET_HOST" "pkill -f global_scheduler/api_server" 2>&1 || true
    sleep 5
    nohup sh block/exp/run_exp_global_scheduler.sh "$TARGET_HOST" $N $N $METRIC_TYPE $HOST_CONFIG_PATH 1 $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO false > /dev/null 2>&1 &
    sleep 20
}

run_cell() {
    local qps=$1 tag=$2 N=$3
    local output_dir="${OUTPUT_PREFIX}/sharegpt/${METRIC_TYPE}/qps_${qps}_n_${N}_len_estimated_true"
    parallel-ssh -i -t 0 --host "$TARGET_HOST" "cd Block && export PYTHONPATH=. && export HF_TOKEN= && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $NUM_REQUEST --dataset_type sharegpt --dataset_path $DATASET_PATH --qps $qps --backend block --log_filename benchmark.log --output_dir $output_dir --data_start_index 0 --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS --use_estimated_response_lens" > /dev/null 2>&1
    sleep 5
    local npz="/tmp/po4po8_$(echo $output_dir | tr '/' '_').npz"
    rsync -az "asdwb@d7525-10s10327.wisc.cloudlab.us:~/Block/experiment_output/$output_dir/benchmark_all_metrics.npz" "$npz" > /dev/null 2>&1
    local p99=$(python3 -c "import numpy as np; d=np.load('$npz', allow_pickle=True); v=d['prefill_token_latencies']; print(f'{np.percentile(v,99):.0f}')" 2>/dev/null)
    echo "[$tag] QPS=$qps TTFT P99=${p99}ms at $(date -u +%H:%M:%SZ)" | tee -a "$RESULTS_FILE"
}

deploy_fresh

# Po4-est
launch_scheduler 4
echo "=== [po4_est] sweep starting ===" | tee -a "$RESULTS_FILE"
for qps in $QPS_LEVELS; do
    run_cell "$qps" "po4_est" "4"
done

# Po8-est
launch_scheduler 8
echo "=== [po8_est] sweep starting ===" | tee -a "$RESULTS_FILE"
for qps in $QPS_LEVELS; do
    run_cell "$qps" "po8_est" "8"
done

echo "=== Po4+Po8 sweep complete at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"
