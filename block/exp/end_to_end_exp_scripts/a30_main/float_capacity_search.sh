#!/bin/bash
# NOTE (Apr 21 lesson): for absolute-metric experiments (heatmap/burstiness), every cell needs
# FRESH vLLM+predictor deploy. This script uses deploy-per-scheduler (shared across QPS probes)
# — acceptable for CAPACITY SEARCH but NOT for absolute-comparison runs. v2 fix: deploy_fresh
# inside QPS loop. Existing data accepted as-is.

# Phase 4.1 Float capacity refinement — adaptive 0.1-step sweep around each scheduler's integer capacity.
# For each scheduler: run midpoint (int+0.5), then step 0.1 up or down based on SLO crossing.
# Stops each scheduler's sweep at first SLO crossing (going up) or first under-SLO found (going down).
# RESTART_VLLM only between different metric_types; predictors reused within same metric_type group.
#
# Usage: sh block/exp/end_to_end_exp_scripts/a30_main/float_capacity_search.sh [from_tag]
#        from_tag (optional): skip configs before this tag (for resume)

set -u

TARGET_HOST="$(head -1 block/config/hosts)"
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
MODEL="meta-llama/Llama-2-7b-hf"
MODEL_TYPE="llama"
PROFILING_SAMPLE_RATE=0.000
NUM_REQUEST=10000
AVAILABLE_INSTANCE="12"
MAX_SLO="0"
HOST_CONFIG_PATH='block/config/host_configs.json'
PREDICTOR_CONFIG_PATH="block/config/${MODEL_TYPE}_config.json"
DATASET_PATH="~/Block/data/trace_data/sharegpt/generate/llama"
OUTPUT_PREFIX_ROOT="main_float"
SLO_MS=10000
RESULTS_FILE="/tmp/a30_phase4_1_float_results.txt"

# Scheduler configs: tag:metric_type:N_SELECTED:use_est:int_lo:int_hi
# mid = (int_lo + int_hi) / 2
SCHEDULER_CONFIGS=(
    "po2_est:min_new_request_latency:2:true:31:32"
    "po2_oracle:min_new_request_latency:2:false:32:33"
    "fanout_est:min_new_request_latency:12:true:31:32"
    "fanout_oracle:min_new_request_latency:12:false:32:33"
    "llumnix:min_lunmnix_load:12:false:31:32"
    "infaas:min_infass_load:12:false:30:31"
    "rr:round_robin:12:false:26:27"
    "minqpm:request_per_seconds:12:false:25:26"
    "random:random:12:false:26:27"
)

FROM_TAG="${1:-}"

echo "=== Phase 4.1 Float capacity search starting at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"

deploy_fresh() {
    local metric_type=$1
    echo "[deploy_fresh] metric_type=$metric_type"
    sh block/exp/reset.sh
    sleep 30
    nohup sh block/exp/run_exp_vllm.sh $BATCH_CAP $MODEL false 0 $MAX_MODEL_LENGTH true $BACKEND_WORKERS $CHUNK_SIZE > /dev/null 2>&1 &
    sleep 90
    for suffix in $(seq 1 7); do
        nohup sh block/exp/run_exp_predictor_${suffix}.sh $PREDICTOR_CONFIG_PATH $metric_type true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
    done
    sleep 10
    for suffix in $(seq 8 $PREDICTOR_WORKERS); do
        nohup sh block/exp/run_exp_predictor_${suffix}.sh $PREDICTOR_CONFIG_PATH $metric_type true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
    done
    sleep 60
    echo "[deploy_fresh] done"
}

launch_scheduler() {
    local N=$1 metric_type=$2
    ssh -o ConnectTimeout=5 "$TARGET_HOST" "pkill -f global_scheduler/api_server" 2>&1 || true
    sleep 5
    nohup sh block/exp/run_exp_global_scheduler.sh "$TARGET_HOST" $N $N $metric_type $HOST_CONFIG_PATH $GLOBAL_SCHEDULER_WORKERS $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO false > /dev/null 2>&1 &
    sleep 20
}

# Returns TTFT P99 in ms via stdout.
run_benchmark() {
    local qps=$1 use_est=$2 output_dir=$3
    local est_flag=""
    [ "$use_est" = "true" ] && est_flag="--use_estimated_response_lens"
    parallel-ssh -i -t 0 --host "$TARGET_HOST" "cd Block && export PYTHONPATH=. && export HF_TOKEN= && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $NUM_REQUEST --dataset_type sharegpt --dataset_path $DATASET_PATH --qps $qps --backend block --log_filename benchmark.log --output_dir $output_dir --data_start_index 0 --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS $est_flag" > /dev/null 2>&1
    sleep 5
    local npz="/tmp/float_$(echo $output_dir | tr '/' '_').npz"
    rsync -az "asdwb@d7525-10s10327.wisc.cloudlab.us:~/Block/experiment_output/$output_dir/benchmark_all_metrics.npz" "$npz" > /dev/null 2>&1
    python3 -c "import numpy as np; d=np.load('$npz', allow_pickle=True); v=d['prefill_token_latencies']; print(f'{np.percentile(v,99):.0f}')"
}

sweep_one() {
    local tag=$1 N=$2 use_est=$3 lo=$4 hi=$5
    local mid=$(awk "BEGIN{printf \"%.1f\", ($lo+$hi)/2}")
    local output_prefix="$OUTPUT_PREFIX_ROOT/$tag"
    echo "=== [$tag] midpoint QPS=$mid ==="
    local p99=$(run_benchmark "$mid" "$use_est" "$output_prefix/qps_$mid")
    echo "[$tag] QPS=$mid: TTFT P99=${p99}ms" | tee -a "$RESULTS_FILE"
    local capacity="unknown"
    if [ "$(awk "BEGIN{print ($p99 < $SLO_MS)}")" = "1" ]; then
        capacity="$mid"
        for step in 0.1 0.2 0.3 0.4; do
            local qps=$(awk "BEGIN{printf \"%.1f\", $mid+$step}")
            p99=$(run_benchmark "$qps" "$use_est" "$output_prefix/qps_$qps")
            echo "[$tag] QPS=$qps: TTFT P99=${p99}ms (sweep up)" | tee -a "$RESULTS_FILE"
            if [ "$(awk "BEGIN{print ($p99 >= $SLO_MS)}")" = "1" ]; then
                echo "[$tag] SLO crossed at $qps, capacity=$capacity" | tee -a "$RESULTS_FILE"
                return
            fi
            capacity="$qps"
        done
        echo "[$tag] no SLO crossing found in up-sweep, capacity=$capacity" | tee -a "$RESULTS_FILE"
    else
        # mid over SLO — sweep down
        for step in 0.1 0.2 0.3 0.4; do
            local qps=$(awk "BEGIN{printf \"%.1f\", $mid-$step}")
            p99=$(run_benchmark "$qps" "$use_est" "$output_prefix/qps_$qps")
            echo "[$tag] QPS=$qps: TTFT P99=${p99}ms (sweep down)" | tee -a "$RESULTS_FILE"
            if [ "$(awk "BEGIN{print ($p99 < $SLO_MS)}")" = "1" ]; then
                capacity="$qps"
                echo "[$tag] first under-SLO at $qps, capacity=$capacity" | tee -a "$RESULTS_FILE"
                return
            fi
        done
        echo "[$tag] no under-SLO found in down-sweep — capacity below $lo" | tee -a "$RESULTS_FILE"
    fi
}

last_metric=""
skipping=1
[ -z "$FROM_TAG" ] && skipping=0
for cfg in "${SCHEDULER_CONFIGS[@]}"; do
    IFS=':' read -r tag metric N use_est lo hi <<< "$cfg"
    if [ "$skipping" = "1" ]; then
        [ "$tag" = "$FROM_TAG" ] && skipping=0 || continue
    fi
    if [ "$metric" != "$last_metric" ]; then
        deploy_fresh "$metric"
        last_metric="$metric"
    fi
    launch_scheduler "$N" "$metric"
    sweep_one "$tag" "$N" "$use_est" "$lo" "$hi"
done

echo "=== Phase 4.1 complete at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"
