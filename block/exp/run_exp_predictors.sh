#!/bin/bash
# A30 Unified Predictor deployment script
# Usage: sh block/exp/run_exp_predictors.sh <config_path> <metric_type> <batch_cap> <num_workers> <predictor_timeout> [enable_chunked_prefill] [num_predictors]
#
# NOTE: To avoid CPU overload from concurrent training, this script:
#   1. Checks if model cache exists
#   2. If no cache: starts 1 predictor first for training, then remaining predictors
#   3. If cache exists: starts all predictors in parallel
#
# See DEPLOYMENT_GUIDE.md for known issues and fixes

CONFIG_PATH=$1
METRIC_TYPE=$2
BATCH_CAP=$3
NUM_WORKERS=$4
PREDICTOR_TIMEOUT=$5
ENABLE_CHUNKED_PREFILL=${6:-true}
NUM_PREDICTORS=${7:-16}

HOSTS_FILE="block/config/hosts"

# Build chunked prefill argument
CHUNKED_PREFILL_ARG=""
if [ "$ENABLE_CHUNKED_PREFILL" = "true" ]; then
    CHUNKED_PREFILL_ARG="--enable_chunked_prefill"
fi

echo "Starting predictors on A30 cluster..."
echo "  Config: $CONFIG_PATH"
echo "  Metric Type: $METRIC_TYPE"
echo "  Batch Cap: $BATCH_CAP"
echo "  Chunked Prefill: $ENABLE_CHUNKED_PREFILL"
echo "  Num Predictors per node: $NUM_PREDICTORS"

# Kill existing predictors
parallel-ssh -t 0 -h $HOSTS_FILE "pkill -f 'predictor/api_server' || true"
sleep 5

# Get first host
FIRST_HOST=$(head -1 $HOSTS_FILE)

# Check if cache exists (to avoid redundant training)
CACHE_COUNT=$(ssh $FIRST_HOST "ls ~/Block/cache/*.pkl 2>/dev/null | wc -l" 2>/dev/null || echo "0")
echo "Found $CACHE_COUNT cached model files"

start_predictor() {
    local INDEX=$1
    local PORT=$((8100 + (INDEX - 1) * 100))

    # For indices > 4, use different port scheme to avoid conflicts
    if [ $INDEX -gt 4 ]; then
        PORT=$((8100 + INDEX))
    fi

    echo "Starting predictor $INDEX on port $PORT..."
    parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/Block && mkdir -p experiment_output/logs && export PYTHONPATH=. && nohup python block/predictor/api_server.py \
        --config_path $CONFIG_PATH \
        --metric_type $METRIC_TYPE \
        --enable_time_estimation true \
        --batch_size_cap $BATCH_CAP \
        --workers $NUM_WORKERS \
        $CHUNKED_PREFILL_ARG \
        --threshold_batch_size_for_time_estimation 0 \
        --predictor_timeout $PREDICTOR_TIMEOUT \
        --predictor_index $INDEX \
        --port $PORT \
        > experiment_output/logs/predictor_${INDEX}.log 2>&1 &"
}

if [ "$CACHE_COUNT" -lt "10" ]; then
    # Phase 1: Start single predictor for training (to avoid CPU overload)
    echo ""
    echo "=== Phase 1: Starting single predictor for model training ==="
    start_predictor 1

    # Wait for training to complete (check health endpoint)
    echo "Waiting for predictor 1 training to complete..."
    MAX_WAIT=300  # 5 minutes max
    WAIT_TIME=0
    while [ $WAIT_TIME -lt $MAX_WAIT ]; do
        if ssh $FIRST_HOST "curl -s http://127.0.0.1:8100/health" 2>/dev/null; then
            echo ""
            echo "Predictor 1 training complete and ready!"
            break
        fi
        echo -n "."
        sleep 10
        WAIT_TIME=$((WAIT_TIME + 10))
    done

    if [ $WAIT_TIME -ge $MAX_WAIT ]; then
        echo ""
        echo "WARNING: Predictor 1 training timeout. Check logs at ~/Block/experiment_output/logs/predictor_1.log"
    fi

    # Phase 2: Start remaining predictors (they will use cached models)
    echo ""
    echo "=== Phase 2: Starting remaining predictors (using cached models) ==="
    for INDEX in $(seq 2 $NUM_PREDICTORS); do
        start_predictor $INDEX
    done
else
    # Cache exists - start all predictors in parallel (they will load from cache)
    echo ""
    echo "=== Cache exists - Starting all $NUM_PREDICTORS predictors in parallel ==="
    for INDEX in $(seq 1 $NUM_PREDICTORS); do
        start_predictor $INDEX
    done
fi

echo ""
echo "All predictors starting..."
echo "Check status with: parallel-ssh -h $HOSTS_FILE 'curl -s http://127.0.0.1:8100/health'"
