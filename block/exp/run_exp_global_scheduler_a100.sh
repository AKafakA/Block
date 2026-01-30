#!/bin/bash
# A100 Global Scheduler deployment script
# Usage: sh block/exp/run_exp_global_scheduler_a100.sh <host> <config_path> <metric_type> <num_predictors> <profiling_rate> <backend_timeout> <predictor_timeout>

HOST=$1
CONFIG_PATH=$2
METRIC_TYPE=$3
NUM_PREDICTORS=$4
PROFILING_SAMPLE_RATE=$5
BACKEND_TIMEOUT=$6
PREDICTOR_TIMEOUT=$7

echo "Starting global scheduler on $HOST..."
echo "  Config: $CONFIG_PATH"
echo "  Metric Type: $METRIC_TYPE"
echo "  Num Predictors: $NUM_PREDICTORS"
echo "  Profiling Sample Rate: $PROFILING_SAMPLE_RATE"

# Kill existing global scheduler
ssh $HOST "pkill -f 'global_scheduler' || true"
sleep 5

# Start global scheduler
ssh $HOST "cd ~/Block && mkdir -p experiment_output/logs && export PYTHONPATH=. && nohup python block/global_scheduler/api_server.py \
    --config_path $CONFIG_PATH \
    --metrics_type $METRIC_TYPE \
    --num_query_predictor 2 \
    --num_required_predictor 1 \
    --workers 1 \
    --num_predictor_ports $NUM_PREDICTORS \
    --profiling_sampling_rate $PROFILING_SAMPLE_RATE \
    --predictor_timeout $PREDICTOR_TIMEOUT \
    --backend_timeout $BACKEND_TIMEOUT \
    --initial_available_instance 1 \
    --max_slo_in_seconds 0 \
    > experiment_output/logs/global_scheduler.log 2>&1 &"

echo "Global scheduler starting on $HOST..."
