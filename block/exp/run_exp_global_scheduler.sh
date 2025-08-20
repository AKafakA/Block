HOST=$1
M=$2
N=$3
METRICS_TYPE=$4
CONFIG_PATH=$5
NUM_WORKERS=$6
NUM_PREDICTORS=$7
PROFILING_SAMPLE_RATE=$8
BACKEND_TIMEOUT=$9
PREDICTOR_TIMEOUT=${10}
AVAILABLE_INSTANCE=${11}
MAX_SLO=${12}
ENABLE_PREEMPTIVE_AUTO_PROVISIONING=${13}

parallel-ssh -t 0 --host $HOST "pkill -f global_scheduler"
sleep 10
if [ "$ENABLE_PREEMPTIVE_AUTO_PROVISIONING" = "true" ]; then
  parallel-ssh -t 0 --host $HOST "cd Block && export PYTHONPATH=. && nohup python block/global_scheduler/api_server.py --config_path $CONFIG_PATH --metrics_type $METRICS_TYPE --num_query_predictor $M --num_required_predictor $N --workers $NUM_WORKERS --num_predictor_ports $NUM_PREDICTORS --profiling_sampling_rate $PROFILING_SAMPLE_RATE --predictor_timeout $PREDICTOR_TIMEOUT --backend_timeout $BACKEND_TIMEOUT --initial_available_instance $AVAILABLE_INSTANCE --max_slo_in_seconds $MAX_SLO --use_preemptive_provisioning > experiment_output/logs/global_scheduler.log 2>&1 &"
else
  parallel-ssh -t 0 --host $HOST "cd Block && export PYTHONPATH=. && nohup python block/global_scheduler/api_server.py --config_path $CONFIG_PATH --metrics_type $METRICS_TYPE --num_query_predictor $M --num_required_predictor $N --workers $NUM_WORKERS --num_predictor_ports $NUM_PREDICTORS --profiling_sampling_rate $PROFILING_SAMPLE_RATE --predictor_timeout $PREDICTOR_TIMEOUT --backend_timeout $BACKEND_TIMEOUT --initial_available_instance $AVAILABLE_INSTANCE --max_slo_in_seconds $MAX_SLO > experiment_output/logs/global_scheduler.log 2>&1 &"
fi