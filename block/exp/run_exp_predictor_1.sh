CONFIG_PATH=$1
METRIC_TYPE=$2
ENABLE_TIME_ESTIMATION=$3
BATCH_CAP=$4
ENABLE_CHUNKED_PREFILL=$5
NUM_WORKERS=$6
BRANCH_NAME=$7
BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION=$8
PREDICTOR_TIMEOUT=$9


APPEND_CHUNKED_PREFILL=""
if [ "$ENABLE_CHUNKED_PREFILL" = "true" ]; then
    APPEND_CHUNKED_PREFILL="--enable_chunked_prefill"
fi

parallel-ssh -i -t 0 -h block/config/hosts "cd Block && export PYTHONPATH=. && nohup python block/predictor/api_server.py --config_path $CONFIG_PATH --metric_type $METRIC_TYPE --enable_time_estimation $ENABLE_TIME_ESTIMATION --batch_size_cap $BATCH_CAP --workers $NUM_WORKERS $APPEND_CHUNKED_PREFILL --threshold_batch_size_for_time_estimation $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION --predictor_timeout $PREDICTOR_TIMEOUT --predictor_index 1 > experiment_output/logs/predictor.log 2>&1 &"