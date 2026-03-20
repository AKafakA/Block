#!/bin/bash
# Generality Capacity Search — Binary search for each variation
# Run from local machine. Assumes vLLM + predictors already deployed for the given config.
# For variations needing redeploy, set RESTART_VLLM=true.
#
# Usage: bash generality_capacity_search.sh [burstgpt|batch24|chunk2048|qwen|all]

TARGET_HOST="TARGET_HOST_PLACEHOLDER"
PREDICTOR_WORKERS=16
GLOBAL_SCHEDULER_WORKERS=1
MAX_MODEL_LENGTH=4096
TIMEOUT_IN_SECONDS=1800
PREDICTOR_TIMEOUT_IN_SECONDS=1000
BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION="0"
BRANCH_NAME="main"
USE_PROCESS_FOR_FRONTEND=true
UPDATE_BLOCK_CODE=false
UPDATE_VLLM_CODE=false
RUN_EXP=true
ENABLE_CHUNKED_PREFILL="true"
N_SELECTED="2"
SCHEDULER_NAME="min_new_request_latency"
PROFILING_SAMPLE_RATE=0.000
USE_FOR_PROFILING_ONLY=false
NUM_REQUEST=10000
KEEP_ALL_METRICS=false
AVAILABLE_INSTANCE="12"
ENABLE_PREEMPTIVE_AUTO_PROVISIONING="false"
MAX_SLO="0"
START_INDEX=0
BACKEND_WORKERS=1

run_single() {
    local model=$1
    local model_type=$2
    local dataset_name=$3
    local qps=$4
    local batch_cap=$5
    local chunk_size=$6
    local use_estimation_len=$7
    local output_dir_prefix=$8

    local dataset_path="~/Block/data/trace_data/$dataset_name/generate/$model_type"

    echo "  Running: QPS=$qps batch=$batch_cap chunk=$chunk_size est_len=$use_estimation_len"

    sh block/exp/experiment.sh \
        $SCHEDULER_NAME $NUM_REQUEST false $batch_cap \
        $dataset_name $dataset_path $dataset_name true $KEEP_ALL_METRICS $START_INDEX \
        $model $model_type $MAX_MODEL_LENGTH $ENABLE_CHUNKED_PREFILL \
        $PREDICTOR_WORKERS $GLOBAL_SCHEDULER_WORKERS $BACKEND_WORKERS $chunk_size \
        $qps $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $N_SELECTED \
        $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $USE_FOR_PROFILING_ONLY \
        $PREDICTOR_TIMEOUT_IN_SECONDS $USE_PROCESS_FOR_FRONTEND \
        $UPDATE_BLOCK_CODE $UPDATE_VLLM_CODE $RUN_EXP \
        $use_estimation_len $output_dir_prefix \
        $AVAILABLE_INSTANCE $MAX_SLO $ENABLE_PREEMPTIVE_AUTO_PROVISIONING
}

get_ttft_p99() {
    local output_prefix=$1
    local qps=$2
    local dataset=$3
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $TARGET_HOST \
        "grep 'p99 prefill' ~/Block/experiment_output/${output_prefix}/${dataset}/min_new_request_latency/qps_${qps}*/benchmark_logs.txt 2>/dev/null | grep -oP '[\d.]+(?= ms)'"
}

binary_search() {
    local model=$1 model_type=$2 dataset_name=$3 batch_cap=$4 chunk_size=$5
    local use_estimation_len=$6 output_prefix=$7 label=$8

    echo ""
    echo "=========================================="
    echo "Binary Search: $label"
    echo "  model=$model dataset=$dataset_name batch=$batch_cap chunk=$chunk_size est_len=$use_estimation_len"
    echo "=========================================="

    # Binary search with configurable QPS range
    # Phase 1: Coarse
    echo "--- Phase 1: Coarse ($COARSE_QPS) ---"
    for qps in $COARSE_QPS; do
        run_single "$model" "$model_type" "$dataset_name" "$qps" "$batch_cap" "$chunk_size" "$use_estimation_len" "$output_prefix"
        p99=$(get_ttft_p99 "$output_prefix" "$qps" "$dataset_name")
        echo "  QPS=$qps → TTFT P99 = ${p99}ms"
    done

    # Phase 2: Narrow
    echo "--- Phase 2: Narrow ($NARROW_QPS) ---"
    for qps in $NARROW_QPS; do
        run_single "$model" "$model_type" "$dataset_name" "$qps" "$batch_cap" "$chunk_size" "$use_estimation_len" "$output_prefix"
        p99=$(get_ttft_p99 "$output_prefix" "$qps" "$dataset_name")
        echo "  QPS=$qps → TTFT P99 = ${p99}ms"
    done

    # Phase 3: Fine
    echo "--- Phase 3: Fine ($FINE_QPS) ---"
    for qps in $FINE_QPS; do
        run_single "$model" "$model_type" "$dataset_name" "$qps" "$batch_cap" "$chunk_size" "$use_estimation_len" "$output_prefix"
        p99=$(get_ttft_p99 "$output_prefix" "$qps" "$dataset_name")
        echo "  QPS=$qps → TTFT P99 = ${p99}ms"
    done

    echo "=== Binary search complete for $label ==="
}

deploy_fresh() {
    local model=$1 batch_cap=$2 chunk_size=$3 metric_type=$4
    echo "=== Deploying fresh: model=$model batch=$batch_cap chunk=$chunk_size ==="
    sh block/exp/reset.sh
    sleep 30
    nohup sh block/exp/run_exp_vllm.sh $batch_cap $model false 0 $MAX_MODEL_LENGTH true $BACKEND_WORKERS $chunk_size > /dev/null 2>&1 &
    sleep 90
    for suffix in $(seq 1 7); do
        nohup sh block/exp/run_exp_predictor_${suffix}.sh block/config/llama_config.json $metric_type true $batch_cap true $PREDICTOR_WORKERS $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
    done
    sleep 10
    for suffix in $(seq 8 $PREDICTOR_WORKERS); do
        nohup sh block/exp/run_exp_predictor_${suffix}.sh block/config/llama_config.json $metric_type true $batch_cap true $PREDICTOR_WORKERS $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
    done
    sleep 60
    echo "=== Deploy complete ==="
}

# === VARIATION: BurstGPT (no redeploy needed) ===
run_burstgpt() {
    echo ""
    echo "######################################################################"
    echo "# BurstGPT — Llama-7B / BurstGPT dataset (no redeploy)"
    echo "######################################################################"
    # BurstGPT has much higher capacity (~59 QPS for broadcasting)
    COARSE_QPS="50 58 66" NARROW_QPS="54 62" FINE_QPS="56 60 64"
    binary_search "meta-llama/Llama-2-7b-hf" "llama" "burstgpt" 48 512 "false" "generality/burstgpt_oracle" "BurstGPT Oracle"
    binary_search "meta-llama/Llama-2-7b-hf" "llama" "burstgpt" 48 512 "true" "generality/burstgpt_estimated" "BurstGPT Estimated"
}

# === VARIATION: batch_cap=24 (needs redeploy) ===
run_batch24() {
    echo ""
    echo "######################################################################"
    echo "# batch_cap=24 — Llama-7B / ShareGPT (redeploy needed)"
    echo "######################################################################"
    deploy_fresh "meta-llama/Llama-2-7b-hf" 24 512 "min_new_request_latency"
    # batch24: Block*=27.9, Block=27.2, Llumnix--=23.9
    COARSE_QPS="24 28 32" NARROW_QPS="26 30" FINE_QPS="25 27 29"
    binary_search "meta-llama/Llama-2-7b-hf" "llama" "sharegpt" 24 512 "true" "generality/batch24_estimated" "batch24 Estimated"
}

# === VARIATION: chunk_size=2048 (needs redeploy) ===
run_chunk2048() {
    echo ""
    echo "######################################################################"
    echo "# chunk_size=2048 — Llama-7B / ShareGPT (redeploy needed)"
    echo "######################################################################"
    deploy_fresh "meta-llama/Llama-2-7b-hf" 48 2048 "min_new_request_latency"
    # chunk2048: Block*=31.5, Block=30.8, Llumnix--=29.8
    COARSE_QPS="28 32 36" NARROW_QPS="30 34" FINE_QPS="29 31 33"
    binary_search "meta-llama/Llama-2-7b-hf" "llama" "sharegpt" 48 2048 "true" "generality/chunk2048_estimated" "chunk2048 Estimated"
}

# === VARIATION: Qwen2-7B (needs redeploy) ===
run_qwen() {
    echo ""
    echo "######################################################################"
    echo "# Qwen2-7B / ShareGPT (redeploy needed)"
    echo "######################################################################"
    deploy_fresh "Qwen/Qwen2-7B" 48 512 "min_new_request_latency"
    # Qwen7B: Block*=68.3, Block=67.9, Llumnix--=62
    COARSE_QPS="58 66 74" NARROW_QPS="62 70" FINE_QPS="64 68 72"
    binary_search "Qwen/Qwen2-7B" "qwen" "sharegpt" 48 512 "true" "generality/qwen7b_estimated" "Qwen7B Estimated"
}

# === MAIN ===
variation=${1:-all}
echo "========================================================"
echo "Generality Capacity Search — $variation"
echo "========================================================"

case $variation in
    burstgpt) run_burstgpt ;;
    batch24) run_batch24 ;;
    chunk2048) run_chunk2048 ;;
    qwen) run_qwen ;;
    all) run_burstgpt; run_batch24; run_chunk2048; run_qwen ;;
    *) echo "Usage: $0 [burstgpt|batch24|chunk2048|qwen|all]"; exit 1 ;;
esac

echo ""
echo "========================================================"
echo "Generality capacity search complete!"
echo "Results in: experiment_output/generality/"
echo "========================================================"
