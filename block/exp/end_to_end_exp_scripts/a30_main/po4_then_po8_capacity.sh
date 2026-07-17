#!/bin/bash
# Post-CPU: Po4 capacity search (seed=32), then Po8 capacity search (seed=Po4_cap).
# No CPU tracking (matches Phase 1 measurement protocol).

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
SLO_MS=10000
RESULTS_FILE=/tmp/a30_phase4_2_simple_results.txt
LOG=/tmp/a30_po4_po8_capacity.log

echo "=== Po4+Po8 capacity starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG" "$RESULTS_FILE"

# Restart predictors WITHOUT --enable_cpu_tracking (clean measurement)
echo "[restart] killing predictors + scheduler, relaunching no-CPU"
parallel-ssh -i -t 0 -h block/config/hosts "pkill -f predictor/api_server" > /dev/null 2>&1
ssh -n -o ConnectTimeout=5 "$TARGET_HOST" "pkill -f global_scheduler/api_server" 2>&1 || true
sleep 5
for suffix in $(seq 1 7); do
    nohup sh block/exp/run_exp_predictor_${suffix}.sh $PRED_CONFIG_PATH $METRIC_TYPE true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS false > /dev/null 2>&1 &
done
sleep 10
for suffix in $(seq 8 $PREDICTOR_WORKERS); do
    nohup sh block/exp/run_exp_predictor_${suffix}.sh $PRED_CONFIG_PATH $METRIC_TYPE true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS false > /dev/null 2>&1 &
done
sleep 60
bash block/exp/end_to_end_exp_scripts/a30_main/verify_predictors.sh "$PRED_CONFIG_PATH" "$METRIC_TYPE" "$BATCH_CAP" "$PREDICTOR_WORKERS" "$PREDICTOR_TIMEOUT_IN_SECONDS" || true
echo "[deploy] done"

launch_scheduler() {
    local N=$1
    ssh -n -o ConnectTimeout=5 "$TARGET_HOST" "pkill -f global_scheduler/api_server" 2>&1 || true
    sleep 5
    nohup sh block/exp/run_exp_global_scheduler.sh "$TARGET_HOST" $N $N $METRIC_TYPE $HOST_CONFIG_PATH 1 $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO false > /dev/null 2>&1 &
    sleep 20
}

run_bench() {
    local qps=$1 output_dir=$2
    parallel-ssh -i -t 0 --host "$TARGET_HOST" "cd Block && export PYTHONPATH=. && export HF_TOKEN= && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $NUM_REQUEST --dataset_type sharegpt --dataset_path $DATASET_PATH --qps $qps --backend block --log_filename benchmark.log --output_dir $output_dir --data_start_index 0 --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS --use_estimated_response_lens" > /dev/null 2>&1
    sleep 5
    local npz=/tmp/popon_$(echo $output_dir | tr '/' '_').npz
    rsync -az "asdwb@d7525-10s10327.wisc.cloudlab.us:~/Block/experiment_output/$output_dir/benchmark_all_metrics.npz" "$npz" > /dev/null 2>&1
    python3 -c "import numpy as np; d=np.load('$npz', allow_pickle=True); v=d['prefill_token_latencies']; print(f'{np.percentile(v,99):.0f}')" 2>/dev/null
}

int_then_float() {
    local tag=$1 N=$2 seed=$3 output_prefix=$4 max_minutes=${5:-60}
    local start_ts=$(date +%s)
    launch_scheduler $N

    local qps=$seed lo hi
    local p99=$(run_bench "$qps" "$output_prefix/qps_$qps")
    echo "[$tag] int seed QPS=$qps: TTFT P99=${p99}ms" | tee -a "$LOG" "$RESULTS_FILE"

    if [ "$(awk "BEGIN{print ($p99 < $SLO_MS)}")" = "1" ]; then
        lo=$qps
        for step in 1 2 3 4 5; do
            qps=$((seed + step))
            p99=$(run_bench "$qps" "$output_prefix/qps_$qps")
            echo "[$tag] int up QPS=$qps: TTFT P99=${p99}ms" | tee -a "$LOG" "$RESULTS_FILE"
            if [ "$(awk "BEGIN{print ($p99 >= $SLO_MS)}")" = "1" ]; then
                hi=$qps; break
            fi
            lo=$qps
            # Time check: stop early if running long
            if [ $(($(date +%s) - start_ts)) -gt $((max_minutes * 60 - 240)) ]; then
                echo "[$tag] time budget exceeded, stopping at lo=$lo" | tee -a "$LOG" "$RESULTS_FILE"
                CAPACITY_OUT=$lo
                return
            fi
        done
        [ -z "${hi:-}" ] && hi=$((lo+1))
    else
        hi=$qps
        for step in 1 2 3 4 5; do
            qps=$((seed - step))
            p99=$(run_bench "$qps" "$output_prefix/qps_$qps")
            echo "[$tag] int down QPS=$qps: TTFT P99=${p99}ms" | tee -a "$LOG" "$RESULTS_FILE"
            if [ "$(awk "BEGIN{print ($p99 < $SLO_MS)}")" = "1" ]; then
                lo=$qps; break
            fi
            hi=$qps
        done
        [ -z "${lo:-}" ] && lo=$((hi-1))
    fi
    echo "[$tag] integer bracket: [$lo, $hi]" | tee -a "$LOG" "$RESULTS_FILE"

    # Float refine ≤3 probes (limited budget)
    local lo_real=$lo hi_real=$hi
    for probe in 1 2 3; do
        if [ "$(awk "BEGIN{print ($hi_real - $lo_real <= 0.15)}")" = "1" ]; then
            CAPACITY_OUT=$lo_real
            echo "[$tag] float converged probe#$probe, capacity=$CAPACITY_OUT" | tee -a "$LOG" "$RESULTS_FILE"
            return
        fi
        if [ $(($(date +%s) - start_ts)) -gt $((max_minutes * 60 - 240)) ]; then
            CAPACITY_OUT=$lo_real
            echo "[$tag] time budget reached at float probe#$probe, capacity=$CAPACITY_OUT" | tee -a "$LOG" "$RESULTS_FILE"
            return
        fi
        local mid=$(awk "BEGIN{l=int($lo_real*10+0.5); h=int($hi_real*10+0.5); m=int((l+h+1)/2); printf \"%.1f\", m/10}")
        p99=$(run_bench "$mid" "$output_prefix/qps_$mid")
        echo "[$tag] float probe#$probe QPS=$mid: TTFT P99=${p99}ms" | tee -a "$LOG" "$RESULTS_FILE"
        if [ "$(awk "BEGIN{print (($SLO_MS - 1000) <= $p99 && $p99 < $SLO_MS)}")" = "1" ]; then
            CAPACITY_OUT=$(awk "BEGIN{printf \"%.1f\", $mid+0.1}"); echo "[$tag] 9.X at $mid, capacity=$CAPACITY_OUT" | tee -a "$LOG" "$RESULTS_FILE"; return
        fi
        if [ "$(awk "BEGIN{print ($SLO_MS <= $p99 && $p99 < ($SLO_MS + 1000))}")" = "1" ]; then
            CAPACITY_OUT=$mid; echo "[$tag] 10.X at $mid, capacity=$CAPACITY_OUT" | tee -a "$LOG" "$RESULTS_FILE"; return
        fi
        if [ "$(awk "BEGIN{print ($p99 < $SLO_MS)}")" = "1" ]; then lo_real=$mid; else hi_real=$mid; fi
    done
    CAPACITY_OUT=$lo_real
    echo "[$tag] float exhausted, capacity=$CAPACITY_OUT" | tee -a "$LOG" "$RESULTS_FILE"
}

# Po4 capacity, max 30 min
CAPACITY_OUT=""
int_then_float "po4_est" 4 32 "po4_capacity" 30
PO4_CAP="$CAPACITY_OUT"
echo "=== [po4_est FINAL] capacity=$PO4_CAP ===" | tee -a "$LOG" "$RESULTS_FILE"

# Po8: ALWAYS probe QPS=32 first (paper snapshot), then capacity search from Po4 cap
echo "=== [po8_est] step 1: QPS=32 snapshot point (paper reference) ===" | tee -a "$LOG" "$RESULTS_FILE"
launch_scheduler 8
po8_32_p99=$(run_bench 32 "po8_capacity/qps_32_snapshot")
echo "[po8_est] QPS=32 (snapshot): TTFT P99=${po8_32_p99}ms" | tee -a "$LOG" "$RESULTS_FILE"

# Po8 capacity search from Po4 cap (already have QPS=32 above; no need to re-probe if seed=32)
PO8_SEED=$(awk "BEGIN{printf \"%d\", $PO4_CAP+0.5}")
if [ "$PO8_SEED" = "32" ]; then
    # Already probed 32, walk up from 33
    PO8_SEED=33
fi
echo "=== [po8_est] step 2: capacity search seed=$PO8_SEED (Po4 cap=$PO4_CAP) ===" | tee -a "$LOG" "$RESULTS_FILE"
CAPACITY_OUT=""
int_then_float "po8_est" 8 "$PO8_SEED" "po8_capacity" 22
PO8_CAP="$CAPACITY_OUT"
echo "=== [po8_est FINAL] QPS=32 TTFT=${po8_32_p99}ms, capacity=$PO8_CAP ===" | tee -a "$LOG" "$RESULTS_FILE"

echo "=== Po4+Po8 capacity complete at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG" "$RESULTS_FILE"
echo "SUMMARY: Po4-est cap=$PO4_CAP, Po8-est cap=$PO8_CAP" | tee -a "$LOG" "$RESULTS_FILE"
