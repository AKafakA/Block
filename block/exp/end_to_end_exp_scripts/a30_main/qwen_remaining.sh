#!/bin/bash
# Remaining Qwen schedulers after po2_est converged at 73.9.
# po2_oracle: seed=76 (po2_est_cap + 2), llumnix N=12: seed=70 (5-7 below po2)

set -u

TARGET_HOST="$(head -1 block/config/hosts)"
PREDICTOR_WORKERS=16
TIMEOUT_IN_SECONDS=1800
PREDICTOR_TIMEOUT_IN_SECONDS=1000
MODEL="Qwen/Qwen2-7B"
PROFILING_SAMPLE_RATE=0.000
NUM_REQUEST=10000
AVAILABLE_INSTANCE="12"
MAX_SLO="0"
MAX_MODEL_LENGTH=4096
HOST_CONFIG_PATH='block/config/host_configs.json'
DATASET_PATH="~/Block/data/trace_data/sharegpt/generate/qwen"
SLO_MS=10000
RESULTS_FILE=/tmp/a30_phase4_2_simple_results.txt
LOG=/tmp/a30_qwen_remaining.log

echo "=== Qwen remaining starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG" "$RESULTS_FILE"

launch_scheduler() {
    local N=$1 metric=$2
    ssh -n -o ConnectTimeout=5 "$TARGET_HOST" "pkill -f global_scheduler/api_server" 2>&1 || true
    sleep 5
    nohup sh block/exp/run_exp_global_scheduler.sh "$TARGET_HOST" $N $N $metric $HOST_CONFIG_PATH 1 $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO false > /dev/null 2>&1 &
    sleep 20
}

run_bench() {
    local qps=$1 use_est=$2 output_dir=$3
    local est_flag=""
    [ "$use_est" = "true" ] && est_flag="--use_estimated_response_lens"
    parallel-ssh -i -t 0 --host "$TARGET_HOST" "cd Block && export PYTHONPATH=. && export HF_TOKEN= && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $NUM_REQUEST --dataset_type sharegpt --dataset_path $DATASET_PATH --qps $qps --backend block --log_filename benchmark.log --output_dir $output_dir --data_start_index 0 --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS $est_flag" > /dev/null 2>&1
    sleep 5
    local npz=/tmp/qwenrem_$(echo $output_dir | tr '/' '_').npz
    rsync -az "asdwb@d7525-10s10327.wisc.cloudlab.us:~/Block/experiment_output/$output_dir/benchmark_all_metrics.npz" "$npz" > /dev/null 2>&1
    python3 -c "import numpy as np; d=np.load('$npz', allow_pickle=True); v=d['prefill_token_latencies']; print(f'{np.percentile(v,99):.0f}')" 2>/dev/null
}

int_then_float() {
    local tag=$1 N=$2 metric=$3 use_est=$4 seed=$5 output_prefix=$6

    launch_scheduler $N $metric

    # Integer search: probe seed, step up if under SLO, down if over
    local qps=$seed lo hi
    local p99=$(run_bench "$qps" "$use_est" "$output_prefix/qps_$qps")
    echo "[$tag] int seed QPS=$qps: TTFT P99=${p99}ms" | tee -a "$LOG" "$RESULTS_FILE"

    if [ "$(awk "BEGIN{print ($p99 < $SLO_MS)}")" = "1" ]; then
        lo=$qps
        for step in 1 2 3 4 5; do
            qps=$((seed + step))
            p99=$(run_bench "$qps" "$use_est" "$output_prefix/qps_$qps")
            echo "[$tag] int up QPS=$qps: TTFT P99=${p99}ms" | tee -a "$LOG" "$RESULTS_FILE"
            if [ "$(awk "BEGIN{print ($p99 >= $SLO_MS)}")" = "1" ]; then
                hi=$qps; break
            fi
            lo=$qps
        done
        [ -z "${hi:-}" ] && hi=$((lo+1))
    else
        hi=$qps
        for step in 1 2 3 4 5; do
            qps=$((seed - step))
            p99=$(run_bench "$qps" "$use_est" "$output_prefix/qps_$qps")
            echo "[$tag] int down QPS=$qps: TTFT P99=${p99}ms" | tee -a "$LOG" "$RESULTS_FILE"
            if [ "$(awk "BEGIN{print ($p99 < $SLO_MS)}")" = "1" ]; then
                lo=$qps; break
            fi
            hi=$qps
        done
        [ -z "${lo:-}" ] && lo=$((hi-1))
    fi
    echo "[$tag] integer bracket: [$lo, $hi]" | tee -a "$LOG" "$RESULTS_FILE"

    # Float search ≤4 probes, 0.1 resolution
    local lo_real=$lo hi_real=$hi capacity=""
    for probe in 1 2 3 4; do
        if [ "$(awk "BEGIN{print ($hi_real - $lo_real <= 0.15)}")" = "1" ]; then
            capacity=$lo_real
            echo "[$tag] float converged at probe#$probe, capacity=$capacity" | tee -a "$LOG" "$RESULTS_FILE"
            return
        fi
        local mid=$(awk "BEGIN{l=int($lo_real*10+0.5); h=int($hi_real*10+0.5); m=int((l+h+1)/2); printf \"%.1f\", m/10}")
        p99=$(run_bench "$mid" "$use_est" "$output_prefix/qps_$mid")
        echo "[$tag] float probe#$probe QPS=$mid: TTFT P99=${p99}ms" | tee -a "$LOG" "$RESULTS_FILE"
        if [ "$(awk "BEGIN{print (($SLO_MS - 1000) <= $p99 && $p99 < $SLO_MS)}")" = "1" ]; then
            capacity=$(awk "BEGIN{printf \"%.1f\", $mid+0.1}")
            echo "[$tag] 9.X band at $mid, capacity=$capacity" | tee -a "$LOG" "$RESULTS_FILE"
            return
        fi
        if [ "$(awk "BEGIN{print ($SLO_MS <= $p99 && $p99 < ($SLO_MS + 1000))}")" = "1" ]; then
            capacity=$mid
            echo "[$tag] 10.X band at $mid, capacity=$capacity" | tee -a "$LOG" "$RESULTS_FILE"
            return
        fi
        if [ "$(awk "BEGIN{print ($p99 < $SLO_MS)}")" = "1" ]; then lo_real=$mid; else hi_real=$mid; fi
    done
    echo "[$tag] float exhausted, capacity=$lo_real (bracket [$lo_real, $hi_real])" | tee -a "$LOG" "$RESULTS_FILE"
}

# po2_oracle (N=2, no est, seed 76)
int_then_float "qwen/po2_oracle" 2 "min_new_request_latency" "false" 76 "generality_float/qwen/po2_oracle"

# llumnix (N=12, seed 70)
int_then_float "qwen/llumnix" 12 "min_lunmnix_load" "false" 70 "generality_float/qwen/llumnix"

echo "=== Qwen remaining complete at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG" "$RESULTS_FILE"
