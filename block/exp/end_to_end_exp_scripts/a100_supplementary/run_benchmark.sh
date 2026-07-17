#!/bin/bash
# ============================================================================
# Benchmark Script for Block and Llumnix
# ============================================================================
#
# This script runs standardized benchmarks for comparing Block and Llumnix.
# It uses the same parameters for both systems to ensure fair comparison.
#
# IMPORTANT - Backend Clarification:
#   - "block"   : Block system (deploy with deploy_block.sh)
#                 Uses predictive scheduling with Vidur-based latency simulation
#   - "llumnix" : Actual Llumnix 0.1.1 (deploy with deploy_llumnix.sh)
#                 Uses migration + FlashInfer, NO chunked prefill
#                 WARNING: Degrades severely at QPS 32+
#
# Note: To run Block with Llumnix-style load dispatching (min_llumnix_load),
# deploy Block with deploy_block.sh and change the global scheduler's
# --metrics_type to "min_llumnix_load". Results go to llumnix_scheduling_only/.
#
# Prerequisites:
#   - Target system (Block or Llumnix) must be deployed and running
#   - ShareGPT dataset at ~/Block/data/trace_data/sharegpt/generate/llama
#
# Usage:
#   ./run_benchmark.sh <backend> [qps] [num_requests] [output_dir]
#
# Arguments:
#   backend       - "block" or "llumnix"
#   qps           - Queries per second (default: 28)
#   num_requests  - Number of requests (default: 10000)
#   output_dir    - Output directory (default: auto-generated)
#
# Examples:
#   ./run_benchmark.sh block 28 10000
#   ./run_benchmark.sh llumnix 24
#   ./run_benchmark.sh block 16 10000 my_experiment
#
# ============================================================================

# set -e removed: can cause premature exit on non-fatal errors

# ============================================================================
# CONFIGURATION
# ============================================================================
# SSH host for running benchmark (node0)
BENCHMARK_HOST="asdwb@d8545-10s10301.wisc.cloudlab.us"

# Internal IP for API endpoints
NODE0_INTERNAL_IP="10.10.1.1"

# HuggingFace configuration
HF_HOME="/mydata/huggingface"

# Model and dataset
MODEL="meta-llama/Llama-2-7b-hf"
DATASET_PATH="~/Block/data/trace_data/sharegpt/generate/llama"
DATASET_TYPE="sharegpt"

# Fixed benchmark parameters (for comparability)
MAX_REQUEST_LEN=4096
TIMEOUT_IN_SECONDS=3600

# Default experiment settings
DEFAULT_QPS=28
DEFAULT_NUM_REQUESTS=10000

# Port configuration
BLOCK_PORT=8200
LLUMNIX_PORT=8200

# ============================================================================
# Helper Functions
# ============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

ssh_cmd() {
    ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no "$BENCHMARK_HOST" "$@"
}

# ============================================================================
# Run Benchmark
# ============================================================================
run_benchmark() {
    local backend=$1
    local qps=$2
    local num_requests=$3
    local output_dir=$4

    # Determine port and estimated length flag based on backend
    local port
    local est_flag=""
    if [ "$backend" = "block" ]; then
        port=$BLOCK_PORT
        est_flag="--use_estimated_response_lens"
    else
        port=$LLUMNIX_PORT
    fi

    # Create output directory name if not specified
    if [ -z "$output_dir" ]; then
        output_dir="experiment_output/benchmark_output/${backend}_qps${qps}_$(date +%Y%m%d_%H%M%S)"
    fi

    log "Running benchmark with configuration:"
    log "  Backend: $backend"
    log "  QPS: $qps"
    log "  Requests: $num_requests"
    log "  Endpoint: ${NODE0_INTERNAL_IP}:${port}"
    log "  Output: $output_dir"
    log ""

    # Run the benchmark
    ssh_cmd "cd ~/Block && \
        export PYTHONPATH=. && \
        export HF_HOME=$HF_HOME && \
        mkdir -p $output_dir && \
        python -m block.benchmark.benchmark_serving \
            --backend $backend \
            --ip_ports ${NODE0_INTERNAL_IP}:${port} \
            --tokenizer $MODEL \
            --trust_remote_code \
            --dataset_type $DATASET_TYPE \
            --dataset_path $DATASET_PATH \
            --qps $qps \
            --num_sampled_requests $num_requests \
            --max_request_len $MAX_REQUEST_LEN \
            --timeout_in_seconds $TIMEOUT_IN_SECONDS \
            --output_dir $output_dir \
            --log_filename ${backend}_qps${qps}_logs.txt \
            $est_flag \
            2>&1 | tee /tmp/benchmark_${backend}_qps${qps}.log"

    log ""
    log "Benchmark complete!"
    log "Results saved to: $output_dir"
}

# ============================================================================
# Run Full QPS Sweep
# ============================================================================
run_qps_sweep() {
    local backend=$1
    local num_requests=${2:-$DEFAULT_NUM_REQUESTS}
    local qps_values=${3:-"16 20 24 28 32 36"}
    local use_estimated=${4:-true}

    log "Running QPS sweep for $backend (estimated_lens=$use_estimated)"
    log "QPS values: $qps_values"
    log "Requests per QPS: $num_requests"
    log ""

    for qps in $qps_values; do
        log "============================================"
        log "Starting benchmark at QPS=$qps"
        log "============================================"

        local output_dir="experiment_output/benchmark_output/${backend}_sweep/${backend}_qps${qps}"
        run_benchmark "$backend" "$qps" "$num_requests" "$output_dir" "$use_estimated"

        log ""
        log "Completed QPS=$qps, sleeping 30s before next run..."
        sleep 30
    done

    log ""
    log "QPS sweep complete!"
    log "Results in: experiment_output/benchmark_output/${backend}_sweep/"
}

# ============================================================================
# Sync Results to Local Machine
# ============================================================================
sync_results() {
    local remote_dir=$1
    local local_dir=$2

    if [ -z "$remote_dir" ] || [ -z "$local_dir" ]; then
        echo "Usage: $0 sync <remote_dir> <local_dir>"
        echo "Example: $0 sync experiment_output/benchmark_output/block_sweep ./results"
        exit 1
    fi

    log "Syncing results from $BENCHMARK_HOST:~/Block/$remote_dir to $local_dir"
    mkdir -p "$local_dir"
    scp -r "$BENCHMARK_HOST:~/Block/$remote_dir/*" "$local_dir/"
    log "Sync complete!"
}

# ============================================================================
# Main
# ============================================================================
main() {
    local command=${1:-help}

    case $command in
        block|llumnix)
            local backend=$1
            local qps=${2:-$DEFAULT_QPS}
            local num_requests=${3:-$DEFAULT_NUM_REQUESTS}
            local output_dir=$4

            run_benchmark "$backend" "$qps" "$num_requests" "$output_dir"
            ;;
        sweep)
            local backend=$2
            local num_requests=$3
            local qps_values=$4
            local use_estimated=${5:-true}

            if [ -z "$backend" ]; then
                echo "Usage: $0 sweep <block|llumnix> [num_requests] [qps_values] [use_estimated]"
                echo "Example: $0 sweep block 10000 \"16 20 24 28 32 36\" true"
                exit 1
            fi

            run_qps_sweep "$backend" "$num_requests" "$qps_values" "$use_estimated"
            ;;
        sync)
            sync_results "$2" "$3"
            ;;
        help|*)
            echo "Usage: $0 <command> [args]"
            echo ""
            echo "Commands:"
            echo "  block [qps] [requests] [output]   - Run Block benchmark"
            echo "  llumnix [qps] [requests] [output] - Run Llumnix benchmark"
            echo "  sweep <backend> [requests] [qps]  - Run full QPS sweep"
            echo "  sync <remote_dir> <local_dir>     - Sync results to local"
            echo ""
            echo "Default Parameters:"
            echo "  QPS: $DEFAULT_QPS"
            echo "  Requests: $DEFAULT_NUM_REQUESTS"
            echo "  Dataset: ShareGPT"
            echo "  Max request length: $MAX_REQUEST_LEN"
            echo ""
            echo "Examples:"
            echo "  $0 block 28 10000"
            echo "  $0 llumnix 24"
            echo "  $0 sweep block 10000 \"16 20 24 28 32 36\""
            echo "  $0 sync experiment_output/benchmark_output/block_sweep ./local_results"
            exit 1
            ;;
    esac
}

main "$@"
