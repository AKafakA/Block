#!/bin/bash
# Post-Qwen: sync CPU patch, then TWO deploys to keep CPU tracking isolated.
# Deploy 1 (no tracking, for fair comparison vs other data): (100,0) REDO2 + Po4@32 + Po8@32
# Deploy 2 (with tracking): CPU tracker Po2-est sweep × 5 QPS

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
RESULTS_FILE="/tmp/a30_post_qwen.log"

echo "=== Post-Qwen starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"

# Sync patched scheduler to all 12 nodes
echo "[sync] pushing patched api_server.py to all 12 nodes"
parallel-scp -h block/config/hosts block/global_scheduler/api_server.py Block/block/global_scheduler/api_server.py > /dev/null 2>&1
echo "[sync] done"

deploy_llama() {
    local enable_cpu=$1  # "true" or "false"
    echo "[deploy] Llama + 16 predictors (cpu_tracking=$enable_cpu)"
    sh block/exp/reset.sh
    sleep 30
    nohup sh block/exp/run_exp_vllm.sh $BATCH_CAP $MODEL false 0 $MAX_MODEL_LENGTH true 1 $CHUNK_SIZE > /dev/null 2>&1 &
    sleep 90
    for suffix in $(seq 1 7); do
        nohup sh block/exp/run_exp_predictor_${suffix}.sh $PRED_CONFIG_PATH $METRIC_TYPE true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS "$enable_cpu" > /dev/null 2>&1 &
    done
    sleep 10
    for suffix in $(seq 8 $PREDICTOR_WORKERS); do
        nohup sh block/exp/run_exp_predictor_${suffix}.sh $PRED_CONFIG_PATH $METRIC_TYPE true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS "$enable_cpu" > /dev/null 2>&1 &
    done
    sleep 60
    bash block/exp/end_to_end_exp_scripts/a30_main/verify_predictors.sh "$PRED_CONFIG_PATH" "$METRIC_TYPE" "$BATCH_CAP" "$PREDICTOR_WORKERS" "$PREDICTOR_TIMEOUT_IN_SECONDS" || true
    echo "[deploy] done"
}

launch_scheduler_with() {
    local N=$1 extra="$2"
    ssh -n -o ConnectTimeout=5 "$TARGET_HOST" "pkill -f global_scheduler/api_server" 2>&1 || true
    sleep 5
    nohup sh block/exp/run_exp_global_scheduler.sh "$TARGET_HOST" $N $N $METRIC_TYPE $HOST_CONFIG_PATH 1 $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO false $extra > /dev/null 2>&1 &
    sleep 20
}

run_bench() {
    local qps=$1 output_dir=$2 tag=$3
    parallel-ssh -i -t 0 --host "$TARGET_HOST" "cd Block && export PYTHONPATH=. && export HF_TOKEN= && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $NUM_REQUEST --dataset_type sharegpt --dataset_path $DATASET_PATH --qps $qps --backend block --log_filename benchmark.log --output_dir $output_dir --data_start_index 0 --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS --use_estimated_response_lens" > /dev/null 2>&1
    sleep 5
    local npz=/tmp/postq_$(echo $output_dir | tr '/' '_').npz
    rsync -az "asdwb@d7525-10s10327.wisc.cloudlab.us:~/Block/experiment_output/$output_dir/benchmark_all_metrics.npz" "$npz" > /dev/null 2>&1
    python3 -c "import numpy as np; d=np.load('$npz', allow_pickle=True); req=d['request_latencies']; ttft=d['prefill_token_latencies']; cpu=float(d.get('mean_cpu_percent', np.array(0.0))) if 'mean_cpu_percent' in d.keys() else None; cpu_str=f', mean_cpu={cpu:.1f}%' if cpu is not None else ''; print(f'[$tag] e2e_mean={float(np.mean(req)):.0f}, e2e_P99={float(np.percentile(req,99)):.0f}, TTFT_P99={float(np.percentile(ttft,99)):.0f}{cpu_str}')" | tee -a "$RESULTS_FILE"
}

# === DEPLOY 1: NO CPU tracking (for comparable measurements) ===
deploy_llama false

# (100,0) REDO2
echo "=== [(100,0) REDO2] ===" | tee -a "$RESULTS_FILE"
launch_scheduler_with 2 "100 0"
run_bench 32 "error_heatmap_po2_rerun/sharegpt/${METRIC_TYPE}/qps_32_len_err_100_lat_err_0_REDO2" "(100,0) REDO2"

# === DEPLOY 2: WITH CPU tracking (isolated for cpu_tracker only) ===
deploy_llama true

# CPU tracker Po2-est sweep
launch_scheduler_with 2 ""
echo "=== [cpu_tracker/po2_est v2] sweep starting ===" | tee -a "$RESULTS_FILE"
for qps in $QPS_LEVELS; do
    run_bench "$qps" "cpu_tracker_po2_v2/sharegpt/${METRIC_TYPE}/qps_${qps}_n_2_len_estimated_true_cpu_tracking_true" "cpu_tracker po2_est QPS=$qps"
done

# === DEPLOY 3: NO CPU tracking (for Po4/Po8 measurements) ===
deploy_llama false

# Po4-est: QPS 32, 33, 34
echo "=== [Po4-est sweep] ===" | tee -a "$RESULTS_FILE"
launch_scheduler_with 4 ""
for qps in 32 33 34; do
    run_bench "$qps" "po4po8_capacity/sharegpt/${METRIC_TYPE}/qps_${qps}_n_4_len_estimated_true" "po4_est QPS=$qps"
done

# Po8-est: QPS 32, 33, 34
echo "=== [Po8-est sweep] ===" | tee -a "$RESULTS_FILE"
launch_scheduler_with 8 ""
for qps in 32 33 34; do
    run_bench "$qps" "po4po8_capacity/sharegpt/${METRIC_TYPE}/qps_${qps}_n_8_len_estimated_true" "po8_est QPS=$qps"
done

echo "=== Post-Qwen complete at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"
