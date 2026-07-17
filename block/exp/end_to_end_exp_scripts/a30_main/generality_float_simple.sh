#!/bin/bash
# Phase 4.2 simple case — batch24 + chunk2048 variations × (Po2-est, Po2-oracle, Llumnix--).
# Same model (Llama-7b) + dataset (sharegpt), so no data/model changes; just batch_cap/chunk_size differ.
# Per scheduler: integer bracketing (step by 1) → float binary search within bracket (0.5, 0.3/0.7 style).
# Early-stop float search when TTFT P99 within ±1s of SLO (9000-11000 ms).

set -u

TARGET_HOST="$(head -1 block/config/hosts)"
PREDICTOR_WORKERS=16
GLOBAL_SCHEDULER_WORKERS=1
BACKEND_WORKERS=1
MAX_MODEL_LENGTH=4096
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
SLO_MS=10000
SLO_TOLERANCE_MS=1000  # stop float search when TTFT within ±1000ms of SLO
RESULTS_FILE="/tmp/a30_phase4_2_simple_results.txt"

# Variations: tag:batch_cap:chunk_size:seed_block_ora:seed_block_est:seed_llumnix
# Seeds are PROXIES for 10s-SLO integer starting point (will search outward).
VARIATIONS=(
    "batch24:24:512:30:30:27"
    "chunk2048:48:2048:33:33:31"
)

# Scheduler configs per variation: tag:metric:N:use_est
SCHED_CONFIGS=(
    "po2_est:min_new_request_latency:2:true"
    "po2_oracle:min_new_request_latency:2:false"
    "llumnix:min_lunmnix_load:12:false"
)

echo "=== Phase 4.2 simple (batch24 + chunk2048) starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE" >&2

deploy_fresh() {
    local metric_type=$1 batch_cap=$2 chunk_size=$3
    # Pick predictor config to match chunk_size (avoid silent predictor/vllm mismatch for chunk2048)
    local pred_cfg="$PREDICTOR_CONFIG_PATH"
    if [ "$chunk_size" = "2048" ]; then
        pred_cfg="block/config/llama_config_chunk2048.json"
    fi
    echo "[deploy_fresh] metric=$metric_type batch=$batch_cap chunk=$chunk_size pred_cfg=$pred_cfg"
    sh block/exp/reset.sh
    sleep 30
    nohup sh block/exp/run_exp_vllm.sh $batch_cap $MODEL false 0 $MAX_MODEL_LENGTH true $BACKEND_WORKERS $chunk_size > /dev/null 2>&1 &
    sleep 90
    for suffix in $(seq 1 7); do
        nohup sh block/exp/run_exp_predictor_${suffix}.sh $pred_cfg $metric_type true $batch_cap true $PREDICTOR_WORKERS $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
    done
    sleep 10
    for suffix in $(seq 8 $PREDICTOR_WORKERS); do
        nohup sh block/exp/run_exp_predictor_${suffix}.sh $pred_cfg $metric_type true $batch_cap true $PREDICTOR_WORKERS $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
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

# Returns TTFT P99 ms via stdout
run_benchmark() {
    local qps=$1 use_est=$2 output_dir=$3
    local est_flag=""
    [ "$use_est" = "true" ] && est_flag="--use_estimated_response_lens"
    parallel-ssh -i -t 0 --host "$TARGET_HOST" "cd Block && export PYTHONPATH=. && export HF_TOKEN= && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $NUM_REQUEST --dataset_type sharegpt --dataset_path $DATASET_PATH --qps $qps --backend block --log_filename benchmark.log --output_dir $output_dir --data_start_index 0 --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS $est_flag" > /dev/null 2>&1
    sleep 5
    local npz="/tmp/p42_$(echo $output_dir | tr '/' '_').npz"
    rsync -az "asdwb@d7525-10s10327.wisc.cloudlab.us:~/Block/experiment_output/$output_dir/benchmark_all_metrics.npz" "$npz" > /dev/null 2>&1
    python3 -c "import numpy as np; d=np.load('$npz', allow_pickle=True); v=d['prefill_token_latencies']; print(f'{np.percentile(v,99):.0f}')"
}

# Integer bracketing: return "lo hi" where lo under SLO and hi >= SLO
integer_bracket() {
    local tag=$1 N=$2 use_est=$3 seed=$4 output_prefix=$5
    local qps=$seed
    local p99=$(run_benchmark "$qps" "$use_est" "$output_prefix/qps_$qps")
    echo "[$tag] int probe QPS=$qps: TTFT P99=${p99}ms" | tee -a "$RESULTS_FILE" >&2
    if [ "$(awk "BEGIN{print ($p99 < $SLO_MS)}")" = "1" ]; then
        # under SLO, search upward
        local lo=$qps
        for step in 1 2 3; do
            qps=$((seed + step))
            p99=$(run_benchmark "$qps" "$use_est" "$output_prefix/qps_$qps")
            echo "[$tag] int up QPS=$qps: TTFT P99=${p99}ms" | tee -a "$RESULTS_FILE" >&2
            if [ "$(awk "BEGIN{print ($p99 >= $SLO_MS)}")" = "1" ]; then
                echo "$lo $qps"; return
            fi
            lo=$qps
        done
        echo "$lo $((lo+1))"  # fallback
    else
        # over SLO, search downward
        local hi=$qps
        for step in 1 2 3; do
            qps=$((seed - step))
            p99=$(run_benchmark "$qps" "$use_est" "$output_prefix/qps_$qps")
            echo "[$tag] int down QPS=$qps: TTFT P99=${p99}ms" | tee -a "$RESULTS_FILE" >&2
            if [ "$(awk "BEGIN{print ($p99 < $SLO_MS)}")" = "1" ]; then
                echo "$qps $hi"; return
            fi
            hi=$qps
        done
        echo "$((hi-1)) $hi"  # fallback
    fi
}

# Float binary search within [lo, hi]; up to MAX_PROBES probes; stop when TTFT P99 within ±1000ms of SLO.
# Maintains shrinking bracket: probe midpoint (rounded up to nearest 0.1), narrow based on result.
# Uses per-scheduler SLO_MS_LOCAL if set, else falls back to global SLO_MS.
float_search() {
    local tag=$1 use_est=$2 lo=$3 hi=$4 output_prefix=$5
    local slo_ms="${SLO_MS_LOCAL:-$SLO_MS}"
    local lo_real=$lo hi_real=$hi
    local capacity=""
    local probe
    for probe in 1 2 3 4; do
        # Stop if bracket width <= 0.15 (converged to 0.1 resolution)
        if [ "$(awk "BEGIN{print ($hi_real - $lo_real <= 0.15)}")" = "1" ]; then
            capacity=$lo_real
            echo "[$tag] float converged bracket [$lo_real, $hi_real] at probe#$probe, capacity=$capacity" | tee -a "$RESULTS_FILE" >&2
            return
        fi
        # Midpoint rounded up to nearest 0.1 (e.g., 28.25 → 28.3)
        local mid=$(awk "BEGIN{l=int($lo_real*10+0.5); h=int($hi_real*10+0.5); m=int((l+h+1)/2); printf \"%.1f\", m/10}")
        local p99=$(run_benchmark "$mid" "$use_est" "$output_prefix/qps_$mid")
        echo "[$tag] float probe#$probe QPS=$mid (bracket [$lo_real,$hi_real], SLO=${slo_ms}ms): TTFT P99=${p99}ms" | tee -a "$RESULTS_FILE" >&2
        # Within -1s of SLO (9.X for 10s SLO): one more 0.1 step likely crosses, capacity = mid + 0.1
        if [ "$(awk "BEGIN{print (($slo_ms - 1000) <= $p99 && $p99 < $slo_ms)}")" = "1" ]; then
            capacity=$(awk "BEGIN{printf \"%.1f\", $mid+0.1}")
            echo "[$tag] 9.X ms at $mid, capacity=$capacity (mid+0.1)" | tee -a "$RESULTS_FILE" >&2
            return
        fi
        # Within +1s of SLO (10.X for 10s SLO): just crossed, capacity = mid
        if [ "$(awk "BEGIN{print ($slo_ms <= $p99 && $p99 < ($slo_ms + 1000))}")" = "1" ]; then
            capacity=$mid
            echo "[$tag] 10.X ms at $mid, capacity=$capacity" | tee -a "$RESULTS_FILE" >&2
            return
        fi
        # Not close enough: narrow bracket and continue
        if [ "$(awk "BEGIN{print ($p99 < $slo_ms)}")" = "1" ]; then
            lo_real=$mid  # mid under, capacity > mid
        else
            hi_real=$mid  # mid over, capacity < mid
        fi
    done
    # Max probes reached
    capacity=$lo_real
    echo "[$tag] float max probes exhausted, capacity=$capacity (final bracket [$lo_real, $hi_real])" | tee -a "$RESULTS_FILE" >&2
}

sweep_scheduler() {
    local var_tag=$1 batch_cap=$2 chunk_size=$3 sched_tag=$4 metric=$5 N=$6 use_est=$7 seed=$8
    local output_prefix="generality_float/${var_tag}/${sched_tag}"
    echo "=== [$var_tag/$sched_tag] N=$N est=$use_est seed_int=$seed ===" | tee -a "$RESULTS_FILE" >&2
    launch_scheduler "$N" "$metric"
    # integer bracketing
    local bracket=$(integer_bracket "$var_tag/$sched_tag" "$N" "$use_est" "$seed" "$output_prefix")
    local lo=$(echo $bracket | awk '{print $1}')
    local hi=$(echo $bracket | awk '{print $2}')
    echo "[$var_tag/$sched_tag] integer bracket: [$lo, $hi]" | tee -a "$RESULTS_FILE" >&2
    # float binary search
    float_search "$var_tag/$sched_tag" "$use_est" "$lo" "$hi" "$output_prefix"
}

last_metric=""
last_batch=""
last_chunk=""
FROM_VAR="${FROM_VAR:-}"     # env var: skip variations before this tag
FROM_SCHED="${FROM_SCHED:-}" # env var: within FROM_VAR, skip scheds before this tag
var_skipping=1; [ -z "$FROM_VAR" ] && var_skipping=0
for var in "${VARIATIONS[@]}"; do
    IFS=':' read -r var_tag batch chunk seed_ora seed_est seed_llu <<< "$var"
    if [ "$var_skipping" = "1" ]; then
        [ "$var_tag" = "$FROM_VAR" ] && var_skipping=0 || continue
    fi
    sched_skipping=0
    if [ "$var_tag" = "$FROM_VAR" ] && [ -n "$FROM_SCHED" ]; then
        sched_skipping=1
    fi
    for sc in "${SCHED_CONFIGS[@]}"; do
        IFS=':' read -r sched_tag metric N use_est <<< "$sc"
        if [ "$sched_skipping" = "1" ]; then
            [ "$sched_tag" = "$FROM_SCHED" ] && sched_skipping=0 || continue
        fi
        # seed per scheduler
        if [ "$sched_tag" = "po2_est" ]; then seed="$seed_est"; fi
        if [ "$sched_tag" = "po2_oracle" ]; then seed="$seed_ora"; fi
        if [ "$sched_tag" = "llumnix" ]; then seed="$seed_llu"; fi
        # redeploy if batch/chunk/metric changed
        if [ "$batch" != "$last_batch" ] || [ "$chunk" != "$last_chunk" ] || [ "$metric" != "$last_metric" ]; then
            deploy_fresh "$metric" "$batch" "$chunk"
            last_metric="$metric"; last_batch="$batch"; last_chunk="$chunk"
        fi
        sweep_scheduler "$var_tag" "$batch" "$chunk" "$sched_tag" "$metric" "$N" "$use_est" "$seed"
    done
done

echo "=== Phase 4.2 simple complete at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE" >&2
