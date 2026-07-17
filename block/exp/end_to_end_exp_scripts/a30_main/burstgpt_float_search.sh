#!/bin/bash
# NOTE (Apr 21 lesson): for absolute-metric experiments (heatmap/burstiness), every cell needs
# FRESH vLLM+predictor deploy. This script uses deploy-per-scheduler (shared across QPS probes)
# — acceptable for CAPACITY SEARCH but NOT for absolute-comparison runs. v2 fix: deploy_fresh
# inside QPS loop. Existing data accepted as-is.

# Phase 4.2 BurstGPT variation — binary search for Po2-oracle + Llumnix--.
# BurstGPT lacks prompt text -> no Po2-est (length predictor can't run on BurstGPT).
# vLLM batch=48, chunk=512 (back to default), predictor uses llama_config.json.
# Same Llama-2-7b model; just dataset swap to burstgpt.

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
DATASET_PATH="~/Block/data/trace_data/burstgpt/generate/llama"
SLO_MS=10000
RESULTS_FILE="/tmp/a30_phase4_2_simple_results.txt"

# tag:metric:N:use_est:seed_int  (no Po2-est since BurstGPT has no prompt text)
# Seeds chosen from paper Table: Po2-oracle=59.0, Llumnix=55.1 at 3s SLO; at 10s SLO slightly higher → +2
SCHED_CONFIGS=(
    "po2_oracle:min_new_request_latency:2:false:61"
    "llumnix:min_lunmnix_load:12:false:57"
)

FROM_SCHED="${FROM_SCHED:-}"

echo "=== Phase 4.2 BurstGPT driver starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"

deploy_fresh() {
    local metric_type=$1
    echo "[deploy] metric=$metric_type batch=$BATCH_CAP chunk=$CHUNK_SIZE"
    sh block/exp/reset.sh
    sleep 30
    nohup sh block/exp/run_exp_vllm.sh $BATCH_CAP $MODEL false 0 $MAX_MODEL_LENGTH true 1 $CHUNK_SIZE > /dev/null 2>&1 &
    sleep 90
    for suffix in $(seq 1 7); do
        nohup sh block/exp/run_exp_predictor_${suffix}.sh $PRED_CONFIG_PATH $metric_type true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
    done
    sleep 10
    for suffix in $(seq 8 $PREDICTOR_WORKERS); do
        nohup sh block/exp/run_exp_predictor_${suffix}.sh $PRED_CONFIG_PATH $metric_type true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
    done
    sleep 60
    echo "[deploy] done"
}

launch_scheduler() {
    local N=$1 metric_type=$2
    ssh -o ConnectTimeout=5 "$TARGET_HOST" "pkill -f global_scheduler/api_server" 2>&1 || true
    sleep 5
    nohup sh block/exp/run_exp_global_scheduler.sh "$TARGET_HOST" $N $N $metric_type $HOST_CONFIG_PATH 1 $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO false > /dev/null 2>&1 &
    sleep 20
}

run_benchmark() {
    local qps=$1 use_est=$2 output_dir=$3
    local est_flag=""
    [ "$use_est" = "true" ] && est_flag="--use_estimated_response_lens"
    parallel-ssh -i -t 0 --host "$TARGET_HOST" "cd Block && export PYTHONPATH=. && export HF_TOKEN= && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $NUM_REQUEST --dataset_type burstgpt --dataset_path $DATASET_PATH --qps $qps --backend block --log_filename benchmark.log --output_dir $output_dir --data_start_index 0 --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS $est_flag" > /dev/null 2>&1
    sleep 5
    local npz="/tmp/p42_$(echo $output_dir | tr '/' '_').npz"
    rsync -az "asdwb@d7525-10s10327.wisc.cloudlab.us:~/Block/experiment_output/$output_dir/benchmark_all_metrics.npz" "$npz" > /dev/null 2>&1
    python3 -c "import numpy as np; d=np.load('$npz', allow_pickle=True); v=d['prefill_token_latencies']; print(f'{np.percentile(v,99):.0f}')"
}

integer_bracket() {
    local tag=$1 use_est=$2 seed=$3 output_prefix=$4
    local qps=$seed
    local p99=$(run_benchmark "$qps" "$use_est" "$output_prefix/qps_$qps")
    echo "[$tag] int probe QPS=$qps (SLO=${SLO_MS}ms): TTFT P99=${p99}ms" | tee -a "$RESULTS_FILE" >&2
    if [ "$(awk "BEGIN{print ($p99 < $SLO_MS)}")" = "1" ]; then
        local lo=$qps
        for step in 1 2 3 4; do
            qps=$((seed + step))
            p99=$(run_benchmark "$qps" "$use_est" "$output_prefix/qps_$qps")
            echo "[$tag] int up QPS=$qps (SLO=${SLO_MS}ms): TTFT P99=${p99}ms" | tee -a "$RESULTS_FILE" >&2
            if [ "$(awk "BEGIN{print ($p99 >= $SLO_MS)}")" = "1" ]; then
                echo "$lo $qps"; return
            fi
            lo=$qps
        done
        echo "$lo $((lo+1))"
    else
        local hi=$qps
        for step in 1 2 3 4; do
            qps=$((seed - step))
            p99=$(run_benchmark "$qps" "$use_est" "$output_prefix/qps_$qps")
            echo "[$tag] int down QPS=$qps (SLO=${SLO_MS}ms): TTFT P99=${p99}ms" | tee -a "$RESULTS_FILE" >&2
            if [ "$(awk "BEGIN{print ($p99 < $SLO_MS)}")" = "1" ]; then
                echo "$qps $hi"; return
            fi
            hi=$qps
        done
        echo "$((hi-1)) $hi"
    fi
}

float_search() {
    local tag=$1 use_est=$2 lo=$3 hi=$4 output_prefix=$5
    local lo_real=$lo hi_real=$hi
    local capacity=""
    local probe
    for probe in 1 2 3 4; do
        if [ "$(awk "BEGIN{print ($hi_real - $lo_real <= 0.15)}")" = "1" ]; then
            capacity=$lo_real
            echo "[$tag] float converged bracket [$lo_real, $hi_real] at probe#$probe, capacity=$capacity" | tee -a "$RESULTS_FILE" >&2
            return
        fi
        local mid=$(awk "BEGIN{l=int($lo_real*10+0.5); h=int($hi_real*10+0.5); m=int((l+h+1)/2); printf \"%.1f\", m/10}")
        local p99=$(run_benchmark "$mid" "$use_est" "$output_prefix/qps_$mid")
        echo "[$tag] float probe#$probe QPS=$mid (bracket [$lo_real,$hi_real], SLO=${SLO_MS}ms): TTFT P99=${p99}ms" | tee -a "$RESULTS_FILE" >&2
        if [ "$(awk "BEGIN{print (($SLO_MS - 1000) <= $p99 && $p99 < $SLO_MS)}")" = "1" ]; then
            capacity=$(awk "BEGIN{printf \"%.1f\", $mid+0.1}")
            echo "[$tag] 9.X band at $mid, capacity=$capacity (mid+0.1)" | tee -a "$RESULTS_FILE" >&2
            return
        fi
        if [ "$(awk "BEGIN{print ($SLO_MS <= $p99 && $p99 < ($SLO_MS + 1000))}")" = "1" ]; then
            capacity=$mid
            echo "[$tag] 10.X band at $mid, capacity=$capacity" | tee -a "$RESULTS_FILE" >&2
            return
        fi
        if [ "$(awk "BEGIN{print ($p99 < $SLO_MS)}")" = "1" ]; then
            lo_real=$mid
        else
            hi_real=$mid
        fi
    done
    capacity=$lo_real
    echo "[$tag] float max probes exhausted, capacity=$capacity (final bracket [$lo_real, $hi_real])" | tee -a "$RESULTS_FILE" >&2
}

last_metric=""
sched_skipping=0
[ -n "$FROM_SCHED" ] && sched_skipping=1

for sc in "${SCHED_CONFIGS[@]}"; do
    IFS=':' read -r tag metric N use_est seed <<< "$sc"
    if [ "$sched_skipping" = "1" ]; then
        [ "$tag" = "$FROM_SCHED" ] && sched_skipping=0 || continue
    fi
    if [ "$metric" != "$last_metric" ]; then
        deploy_fresh "$metric"
        last_metric="$metric"
    fi
    echo "=== [burstgpt/$tag] N=$N est=$use_est seed_int=$seed ===" | tee -a "$RESULTS_FILE"
    launch_scheduler "$N" "$metric"
    output_prefix="generality_float/burstgpt/$tag"
    bracket=$(integer_bracket "burstgpt/$tag" "$use_est" "$seed" "$output_prefix")
    lo=$(echo $bracket | awk '{print $1}')
    hi=$(echo $bracket | awk '{print $2}')
    echo "[burstgpt/$tag] integer bracket: [$lo, $hi]" | tee -a "$RESULTS_FILE"
    float_search "burstgpt/$tag" "$use_est" "$lo" "$hi" "$output_prefix"
done

echo "=== BurstGPT driver complete at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"
