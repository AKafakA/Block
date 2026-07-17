#!/bin/bash
# ============================================================================
# Block Deployment Script for A100 Cluster
# ============================================================================
#
# This script deploys Block (vLLM + Predictors + Global Scheduler) on a
# multi-node A100 cluster for serving Llama-2-7B with 8 instances.
#
# Default: Predictive scheduling with Vidur-based latency simulation
#   --metrics_type min_new_request_latency
#
# To use Llumnix-style load dispatching (for llumnix_scheduling_only comparison):
#   Change METRICS_TYPE below to "min_llumnix_load"
#   This emulates Llumnix's load-based dispatching without migration
#   Results should go to llumnix_scheduling_only/ directory
#
# Prerequisites:
#   - SSH access to all nodes configured
#   - vLLM installed on all nodes
#   - Block repository at ~/Block on all nodes
#   - HuggingFace token with access to Llama model
#
# Usage:
#   ./deploy_block.sh [start|stop|status]
#
# Configuration:
#   Edit the CLUSTER CONFIGURATION section below for your setup
#
# ============================================================================

# set -e removed: pkill returns non-zero when no processes found, which kills the script

# ============================================================================
# CLUSTER CONFIGURATION - Edit these for your cluster
# ============================================================================
# Node hostnames (SSH-accessible)
NODE0_HOST="asdwb@d8545-10s10301.wisc.cloudlab.us"
NODE1_HOST="asdwb@d8545-10s10305.wisc.cloudlab.us"

# Internal IPs (for inter-node communication)
NODE0_INTERNAL_IP="10.10.1.1"
NODE1_INTERNAL_IP="10.10.1.2"

# HuggingFace configuration
HF_TOKEN="${HF_TOKEN:-$(cat ~/.hf_token 2>/dev/null)}"
if [ -z "$HF_TOKEN" ]; then echo "ERROR: Set HF_TOKEN or create ~/.hf_token" && exit 1; fi
HF_HOME="/mydata/huggingface"

# Model configuration
MODEL="meta-llama/Llama-2-7b-hf"
MAX_NUM_SEQS=96           # Batch size per instance (96 for A100, paper §6.5)
MAX_MODEL_LEN=4096        # Maximum sequence length
CHUNK_SIZE=512            # Chunked prefill token budget

# Block configuration paths
BLOCK_HOST_CONFIG="block/config/a100_8x7b_host_configs.json"
PREDICTOR_CONFIG="block/config/llama7b_a100_config.json"

# Global scheduler settings
# NUM_QUERY_PREDICTOR controls scheduler N (8=Fanout, 2=Po2). Override via env: NUM_QUERY_PREDICTOR=2 sh deploy_block.sh ...
NUM_INSTANCES=8
NUM_QUERY_PREDICTOR="${NUM_QUERY_PREDICTOR:-8}"
NUM_PREDICTORS_PER_INSTANCE=4
# ENABLE_CP controls --enable-chunked-prefill flag for both vLLM and predictors. Override via env: ENABLE_CP=false sh deploy_block.sh ...
ENABLE_CP="${ENABLE_CP:-true}"
PROFILING_SAMPLE_RATE=0.0
PREDICTOR_TIMEOUT=2000
BACKEND_TIMEOUT=3600

# Scheduling policy:
#   "min_new_request_latency" = Block predictive scheduling (default)
#   "min_llumnix_load"        = Llumnix-style load dispatching (for comparison)
METRICS_TYPE="min_new_request_latency"

# ============================================================================
# Helper Functions
# ============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

ssh_cmd() {
    local host=$1
    shift
    ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no "$host" "$@"
}

wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_wait=${4:-300}
    local waited=0

    log "Waiting for $service_name on $host:$port..."
    while [ $waited -lt $max_wait ]; do
        if ssh_cmd "$host" "curl -s --max-time 3 http://127.0.0.1:$port/v1/models 2>/dev/null || curl -s --max-time 3 http://127.0.0.1:$port/health 2>/dev/null" | grep -qE "detail|OK|healthy|model"; then
            log "$service_name is ready on port $port"
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
        echo "  Waiting... ($waited/$max_wait s)"
    done
    log "ERROR: $service_name on port $port did not become ready in $max_wait seconds"
    return 1
}

# ============================================================================
# Stop all Block services
# ============================================================================
stop_block() {
    log "Stopping all Block services..."

    for host in "$NODE0_HOST" "$NODE1_HOST"; do
        log "Stopping services on $host..."
        ssh_cmd "$host" "pkill -f 'vllm.entrypoints.api_server' 2>/dev/null || true"
        ssh_cmd "$host" "pkill -f 'predictor/api_server' 2>/dev/null || true"
        ssh_cmd "$host" "pkill -f 'global_scheduler/api_server' 2>/dev/null || true"
    done

    sleep 5
    log "All Block services stopped"
}

# ============================================================================
# Deploy vLLM instances (8 total, 4 per node)
# ============================================================================
deploy_vllm() {
    log "Deploying vLLM instances (8 total, CP=$ENABLE_CP)..."

    # Build chunked prefill flag for vLLM (matches A30 experiment.sh logic exactly):
    #   CP on:  --enable-chunked-prefill --max-num-batched-tokens 512  (CHUNK_SIZE)
    #   CP off: (no flag)                --max-num-batched-tokens 4096 (MAX_MODEL_LEN)
    # Predictor JSON config has both chunk_size=512 and max_tokens_in_batch=4096; predictor's
    # --enable_chunked_prefill flag picks which one to use, so vLLM/predictor stay consistent.
    local vllm_cp_flag=""
    local vllm_max_tokens="--max-num-batched-tokens $MAX_MODEL_LEN"
    if [ "$ENABLE_CP" = "true" ]; then
        vllm_cp_flag="--enable-chunked-prefill"
        vllm_max_tokens="--max-num-batched-tokens $CHUNK_SIZE"
    fi

    for host in "$NODE0_HOST" "$NODE1_HOST"; do
        ssh_cmd "$host" "cd ~/Block && mkdir -p experiment_output/logs"

        log "Starting 4 vLLM instances on $host (CP=$ENABLE_CP)..."
        ssh_cmd "$host" "cd ~/Block && \
            export HF_HOME=$HF_HOME && \
            export HF_TOKEN=$HF_TOKEN && \
            export VLLM_USE_V1=0 && \
            export PYTHONPATH=. && \
            for gpu in 0 1 2 3; do \
                port=\$((8000 + gpu)); \
                CUDA_VISIBLE_DEVICES=\$gpu nohup python -m vllm.entrypoints.api_server \
                    --model $MODEL \
                    --port \$port \
                    --max-num-seqs $MAX_NUM_SEQS \
                    --max-model-len $MAX_MODEL_LEN \
                    $vllm_cp_flag \
                    $vllm_max_tokens \
                    > experiment_output/logs/vllm_gpu\${gpu}.log 2>&1 & \
            done"
    done

    log "Waiting 360s for all vLLM instances to load models (cudagraphs only when CP enabled)..."
    sleep 360

    # Verify all instances loaded
    for host in "$NODE0_HOST" "$NODE1_HOST"; do
        for port in 8000 8001 8002 8003; do
            if ssh_cmd "$host" "curl -s --max-time 5 http://127.0.0.1:$port/v1/models 2>/dev/null" | grep -q "detail"; then
                log "vLLM $host:$port - OK"
            else
                log "WARNING: vLLM $host:$port - NOT READY, waiting 60s more..."
                sleep 60
            fi
        done
    done

    log "All 8 vLLM instances deployed"
}

# ============================================================================
# Deploy Predictors (4 per instance = 32 total)
# ============================================================================
deploy_predictors() {
    log "Deploying Block predictors (32 total, CP=$ENABLE_CP)..."

    # Build chunked prefill flag for predictor (must match vLLM's CP setting)
    local pred_cp_flag=""
    if [ "$ENABLE_CP" = "true" ]; then
        pred_cp_flag="--enable_chunked_prefill"
    fi

    # Step 1: Each node builds its own cache (parallel, no cross-node scp)
    log "Training predictor cache on each node (parallel)..."
    for host in "$NODE0_HOST" "$NODE1_HOST"; do
        ssh_cmd "$host" "cd ~/Block && pkill -f 'predictor/api_server' 2>/dev/null; rm -f experiment_output/logs/predictor_cache_build.log; export PYTHONPATH=. && \
            nohup python block/predictor/api_server.py \
                --config_path $PREDICTOR_CONFIG \
                --metric_type min_new_request_latency \
                --enable_time_estimation true \
                --batch_size_cap $MAX_NUM_SEQS \
                --workers 1 \
                $pred_cp_flag \
                --threshold_batch_size_for_time_estimation 0 \
                --predictor_timeout $PREDICTOR_TIMEOUT \
                --predictor_index 0 \
                --port 8100 \
                > experiment_output/logs/predictor_cache_build.log 2>&1 &" &
    done
    wait

    # Wait for cache to be built on both nodes (check for "Uvicorn running")
    log "Waiting for cache build on both nodes..."
    sleep 30
    for host in "$NODE0_HOST" "$NODE1_HOST"; do
        for i in {1..24}; do
            if ssh_cmd "$host" "grep -q 'Uvicorn running' ~/Block/experiment_output/logs/predictor_cache_build.log 2>/dev/null"; then
                log "Cache built on $host"
                break
            fi
            sleep 5
        done
    done

    # Step 2: Kill cache-building predictors
    log "Stopping cache-building predictors..."
    for host in "$NODE0_HOST" "$NODE1_HOST"; do
        ssh_cmd "$host" "pkill -f 'predictor/api_server' 2>/dev/null || true"
    done
    sleep 3

    # Step 3: Start all predictors on both nodes (parallel)
    # Port mapping: skip 8200 (global scheduler)
    # GPU0->8100-8103, GPU1->8300-8303, GPU2->8400-8403, GPU3->8500-8503
    PREDICTOR_BASE_PORTS="8100 8300 8400 8500"
    for host in "$NODE0_HOST" "$NODE1_HOST"; do
        log "Starting 16 predictors on $host..."
        ssh_cmd "$host" "cd ~/Block && export PYTHONPATH=. && \
            gpu=0; \
            for base_port in $PREDICTOR_BASE_PORTS; do \
                backend_port=\$((8000 + gpu)); \
                for p in 0 1 2 3; do \
                    pred_port=\$((base_port + p)); \
                    nohup python block/predictor/api_server.py \
                        --config_path $PREDICTOR_CONFIG \
                        --metric_type min_new_request_latency \
                        --enable_time_estimation true \
                        --batch_size_cap $MAX_NUM_SEQS \
                        --workers 1 \
                        $pred_cp_flag \
                        --threshold_batch_size_for_time_estimation 0 \
                        --predictor_timeout $PREDICTOR_TIMEOUT \
                        --predictor_index \$((gpu * 4 + p)) \
                        --port \$pred_port \
                        > experiment_output/logs/predictor_gpu\${gpu}_\${p}.log 2>&1 & \
                done; \
                gpu=\$((gpu + 1)); \
            done" &
    done
    wait

    sleep 20
    log "All 32 predictors deployed"
}

# ============================================================================
# Stop just predictors (for restart between scheduler types)
# ============================================================================
stop_predictors() {
    log "Stopping predictors only..."
    for host in "$NODE0_HOST" "$NODE1_HOST"; do
        ssh_cmd "$host" "pkill -f 'predictor/api_server' 2>/dev/null || true"
    done
    sleep 3
}

# ============================================================================
# Stop just scheduler (for restart with different metric type)
# ============================================================================
stop_scheduler() {
    log "Stopping scheduler only..."
    ssh_cmd "$NODE0_HOST" "pkill -f 'global_scheduler/api_server' 2>/dev/null || true"
    sleep 3
}

# ============================================================================
# Deploy Global Scheduler
# ============================================================================
deploy_scheduler() {
    log "Deploying Block global scheduler on node0..."

    ssh_cmd "$NODE0_HOST" "cd ~/Block && export PYTHONPATH=. && \
        nohup python block/global_scheduler/api_server.py \
            --config_path $BLOCK_HOST_CONFIG \
            --metrics_type $METRICS_TYPE \
            --num_query_predictor $NUM_QUERY_PREDICTOR \
            --num_required_predictor 1 \
            --workers 1 \
            --num_predictor_ports $NUM_PREDICTORS_PER_INSTANCE \
            --profiling_sampling_rate $PROFILING_SAMPLE_RATE \
            --predictor_timeout $PREDICTOR_TIMEOUT \
            --backend_timeout $BACKEND_TIMEOUT \
            --initial_available_instance $NUM_INSTANCES \
            --max_slo_in_seconds 0 \
            > experiment_output/logs/global_scheduler.log 2>&1 &"

    sleep 10
    log "Global scheduler deployed on port 8200"
}

# ============================================================================
# Check status of all services
# ============================================================================
check_status() {
    log "Checking Block service status..."

    echo ""
    echo "=== vLLM Instances ==="
    for host in "$NODE0_HOST" "$NODE1_HOST"; do
        for port in 8000 8001 8002 8003; do
            if ssh_cmd "$host" "curl -s http://127.0.0.1:$port/health 2>/dev/null" | grep -qE "OK|healthy"; then
                echo "  $host:$port - RUNNING"
            else
                echo "  $host:$port - NOT RUNNING"
            fi
        done
    done

    echo ""
    echo "=== Predictors ==="
    for host in "$NODE0_HOST" "$NODE1_HOST"; do
        count=$(ssh_cmd "$host" "pgrep -f 'predictor/api_server' | wc -l" 2>/dev/null || echo "0")
        echo "  $host - $count predictors running"
    done

    echo ""
    echo "=== Global Scheduler ==="
    if ssh_cmd "$NODE0_HOST" "pgrep -f 'global_scheduler/api_server'" &>/dev/null; then
        echo "  $NODE0_HOST:8200 - RUNNING"
    else
        echo "  $NODE0_HOST:8200 - NOT RUNNING"
    fi
    echo ""
}

# ============================================================================
# Main
# ============================================================================
main() {
    local action=${1:-start}

    case $action in
        start)
            log "Starting Block deployment (full)..."
            stop_block
            deploy_vllm
            deploy_predictors
            deploy_scheduler
            log "Block deployment complete!"
            log "Global Scheduler endpoint: http://${NODE0_INTERNAL_IP}:8200"
            log "Scheduling policy: $METRICS_TYPE"
            check_status
            ;;
        start_vllm)
            log "Deploying vLLM only..."
            deploy_vllm
            ;;
        start_predictors)
            log "Deploying predictors only (assumes vLLM already running)..."
            stop_predictors
            stop_scheduler
            deploy_predictors
            ;;
        start_scheduler)
            log "Deploying scheduler only..."
            stop_scheduler
            deploy_scheduler
            log "Scheduler deployed with metric: $METRICS_TYPE"
            ;;
        stop)
            stop_block
            ;;
        stop_predictors)
            stop_predictors
            ;;
        stop_scheduler)
            stop_scheduler
            ;;
        status)
            check_status
            ;;
        *)
            echo "Usage: $0 [start|stop|status]"
            echo ""
            echo "Commands:"
            echo "  start   - Deploy all Block services (default)"
            echo "  stop    - Stop all Block services"
            echo "  status  - Check status of all services"
            exit 1
            ;;
    esac
}

main "$@"
