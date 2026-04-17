#!/bin/bash
# ============================================================================
# Llumnix Deployment Script for A100 Cluster
# ============================================================================
#
# This script deploys Llumnix 0.1.1 with vLLM 0.6.3.post1 on a multi-node
# A100 cluster for serving Llama-2-7B with 8 instances.
#
# Configuration:
#   - FlashInfer attention backend
#   - Migration ENABLED (rayrpc)
#   - NO chunked prefill (incompatible with migration in Llumnix 0.1.1)
#
# Performance Notes:
#   - Works well at QPS 16-28
#   - WARNING: Degrades severely at QPS 32+ (latency explodes)
#   - For high-load comparison, use Block with min_llumnix_load policy instead
#
# Prerequisites:
#   - SSH access to all nodes configured
#   - Llumnix virtualenv at /mydata/llumnix_venv on all nodes
#   - Block repository at ~/Block on all nodes
#   - HuggingFace token with access to Llama model
#
# Usage:
#   ./deploy_llumnix.sh [start|stop|status]
#
# ============================================================================

# set -e removed: pkill returns non-zero when no processes found

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
MAX_NUM_SEQS=96           # Batch size
MAX_MODEL_LEN=4096        # Maximum sequence length

# Llumnix configuration
LLUMNIX_VENV="/mydata/llumnix_venv"
NUM_INSTANCES=8           # Total instances across cluster
RAY_PORT=6380             # Ray cluster port
LLUMNIX_PORT=8200         # Llumnix API server port

# Dispatch policy: load, balanced, queue, random
DISPATCH_POLICY="load"

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

wait_for_llumnix() {
    local max_wait=${1:-600}
    local log_file=${2:-"llumnix_migration_flashinfer.log"}
    local waited=0

    log "Waiting for Llumnix to initialize all $NUM_INSTANCES instances..."
    while [ $waited -lt $max_wait ]; do
        # Check if Llumnix is polling all instances
        if ssh_cmd "$NODE0_HOST" "grep -q 'Polling instance infos of $NUM_INSTANCES instances' ~/Block/experiment_output/logs/$log_file 2>/dev/null"; then
            log "Llumnix is ready with $NUM_INSTANCES instances!"
            return 0
        fi

        # Also check for health endpoint
        if ssh_cmd "$NODE0_HOST" "curl -s http://127.0.0.1:$LLUMNIX_PORT/health 2>/dev/null" | grep -qE "OK|healthy"; then
            # Double-check instance count in logs
            instance_count=$(ssh_cmd "$NODE0_HOST" "grep -o 'Polling instance infos of [0-9]* instances' ~/Block/experiment_output/logs/$log_file 2>/dev/null | tail -1 | grep -o '[0-9]*'" || echo "0")
            if [ "$instance_count" -ge "$NUM_INSTANCES" ]; then
                log "Llumnix is ready with $instance_count instances!"
                return 0
            fi
        fi

        sleep 15
        waited=$((waited + 15))

        # Show current status
        current=$(ssh_cmd "$NODE0_HOST" "grep -c 'Created engine instance' ~/Block/experiment_output/logs/$log_file 2>/dev/null" || echo "0")
        echo "  Waiting... ($waited/$max_wait s) - $current instances created"
    done

    log "WARNING: Llumnix may not have fully initialized. Check logs."
    return 1
}

# ============================================================================
# Stop all Llumnix services
# ============================================================================
stop_llumnix() {
    log "Stopping all Llumnix services..."

    for host in "$NODE0_HOST" "$NODE1_HOST"; do
        log "Stopping services on $host..."
        ssh_cmd "$host" "pkill -9 -f 'llumnix.entrypoints' 2>/dev/null || true"
        ssh_cmd "$host" "$LLUMNIX_VENV/bin/ray stop --force 2>/dev/null || true"
    done

    # Clean up Ray session files
    for host in "$NODE0_HOST" "$NODE1_HOST"; do
        ssh_cmd "$host" "rm -rf /tmp/ray/session_* 2>/dev/null || true"
    done

    sleep 5
    log "All Llumnix services stopped"
}

# ============================================================================
# Start Ray cluster
# ============================================================================
start_ray_cluster() {
    log "Starting Ray cluster..."

    # Start Ray head on node0
    log "Starting Ray head node on $NODE0_HOST..."
    ssh_cmd "$NODE0_HOST" "$LLUMNIX_VENV/bin/ray start --head --port=$RAY_PORT"
    sleep 10

    # Join from node1
    log "Joining Ray cluster from $NODE1_HOST..."
    ssh_cmd "$NODE1_HOST" "export RAY_raylet_start_wait_time_s=120 && \
        $LLUMNIX_VENV/bin/ray start --address=$NODE0_INTERNAL_IP:$RAY_PORT"
    sleep 10

    # Verify cluster
    log "Verifying Ray cluster..."
    ssh_cmd "$NODE0_HOST" "$LLUMNIX_VENV/bin/ray status"
}

# ============================================================================
# Deploy Llumnix with Migration + FlashInfer
# ============================================================================
deploy_llumnix() {
    log "=========================================="
    log "Deploying Llumnix: Migration + FlashInfer"
    log "  Backend: FlashInfer"
    log "  Chunked Prefill: DISABLED"
    log "  Migration: ENABLED (rayrpc)"
    log ""
    log "  NOTE: Degrades at QPS 32+"
    log "=========================================="

    # Create log directory
    ssh_cmd "$NODE0_HOST" "mkdir -p ~/Block/experiment_output/logs"

    local log_file="llumnix_migration_flashinfer.log"

    # Start Llumnix (FlashInfer backend, no chunked prefill, migration enabled)
    log "Starting Llumnix API server on $NODE0_HOST..."
    ssh_cmd "$NODE0_HOST" "cd ~/Block && \
        export HF_HOME=$HF_HOME && \
        export HF_TOKEN=$HF_TOKEN && \
        export PYTHONUNBUFFERED=1 && \
        export VLLM_ATTENTION_BACKEND=FLASHINFER && \
        nohup $LLUMNIX_VENV/bin/python -u -m llumnix.entrypoints.vllm.api_server \
            --host 0.0.0.0 \
            --port $LLUMNIX_PORT \
            --initial-instances $NUM_INSTANCES \
            --ray-cluster-port $RAY_PORT \
            --model $MODEL \
            --worker-use-ray \
            --trust-remote-code \
            --max-model-len $MAX_MODEL_LEN \
            --max-num-seqs $MAX_NUM_SEQS \
            --dispatch-policy $DISPATCH_POLICY \
            --migration-backend rayrpc \
            --enable-migration \
            > experiment_output/logs/$log_file 2>&1 &"

    # Wait for Llumnix to be ready
    wait_for_llumnix 600 "$log_file"
}

# ============================================================================
# Check status of all services
# ============================================================================
check_status() {
    log "Checking Llumnix service status..."

    echo ""
    echo "=== Ray Cluster ==="
    ssh_cmd "$NODE0_HOST" "$LLUMNIX_VENV/bin/ray status 2>/dev/null" | grep -E "CPU|GPU|node" || echo "Ray not running"

    echo ""
    echo "=== Llumnix API Server ==="
    if ssh_cmd "$NODE0_HOST" "pgrep -f 'llumnix.entrypoints'" &>/dev/null; then
        echo "  $NODE0_HOST:$LLUMNIX_PORT - RUNNING"

        local log_file="llumnix_migration_flashinfer.log"
        if ssh_cmd "$NODE0_HOST" "test -f ~/Block/experiment_output/logs/$log_file" 2>/dev/null; then
            instance_count=$(ssh_cmd "$NODE0_HOST" "grep -o 'Polling instance infos of [0-9]* instances' ~/Block/experiment_output/logs/$log_file 2>/dev/null | tail -1 | grep -o '[0-9]*'" || echo "unknown")
            echo "  Active instances: $instance_count"
            echo "  Log file: $log_file"
        fi

        # Check health
        health=$(ssh_cmd "$NODE0_HOST" "curl -s http://127.0.0.1:$LLUMNIX_PORT/health 2>/dev/null" || echo "unreachable")
        echo "  Health check: $health"
    else
        echo "  $NODE0_HOST:$LLUMNIX_PORT - NOT RUNNING"
    fi

    echo ""
    echo "=== GPU Memory Usage ==="
    for host in "$NODE0_HOST" "$NODE1_HOST"; do
        echo "  $host:"
        ssh_cmd "$host" "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader" 2>/dev/null | sed 's/^/    /'
    done
    echo ""
}

# ============================================================================
# Main
# ============================================================================
main() {
    local action=${1:-start}

    case $action in
        start)
            log "Starting Llumnix deployment (Migration + FlashInfer)..."
            stop_llumnix
            sleep 3
            start_ray_cluster
            deploy_llumnix
            log ""
            log "Llumnix deployment complete!"
            log "API endpoint: http://${NODE0_INTERNAL_IP}:${LLUMNIX_PORT}"
            log ""
            check_status
            ;;
        stop)
            stop_llumnix
            ;;
        status)
            check_status
            ;;
        *)
            echo "Usage: $0 [start|stop|status]"
            echo ""
            echo "Commands:"
            echo "  start   - Deploy Llumnix with Migration + FlashInfer"
            echo "  stop    - Stop Llumnix and Ray cluster"
            echo "  status  - Check status of all services"
            echo ""
            echo "Configuration:"
            echo "  - FlashInfer attention backend"
            echo "  - Migration enabled (rayrpc)"
            echo "  - No chunked prefill (incompatible with migration)"
            echo ""
            echo "Performance at QPS=28:"
            echo "  Throughput: 15,047 tok/s"
            echo "  E2E Latency: 7,635 ms"
            echo "  Token Latency: 30.15 ms"
            echo ""
            echo "WARNING: Degrades at QPS 32+"
            echo "  QPS=32: E2E Latency 47,305 ms"
            echo "  QPS=36: E2E Latency 89,027 ms"
            echo ""
            echo "For high-load comparison, use Block with min_llumnix_load policy"
            echo "(results in llumnix_scheduling_only/)"
            exit 1
            ;;
    esac
}

main "$@"
