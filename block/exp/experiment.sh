# This script is used to run benchmark and global scheduler with the Block framework.
TARGET_HOST=""

SCHEDULER_METRIC_TYPE=$1
ENABLE_TIME_ESTIMATION=true

NUM_DATA=$2
RESTART_VLLM=$3
BATCH_CAP=$4


DATASET_NAME=$5
DATASET_PATH=$6
DATASET_TYPE=$7
GENERATE_NEW_DATA=$8
KEEP_ALL_METRICS=$9
START_INDEX=${10}
MODEL=${11}
MODEL_TYPE=${12}
MAX_MODEL_LENGTH=${13}
HOST_CONFIG_PATH='block/config/host_configs.json'
PREDICTOR_CONFIG_PATH="block/config/${MODEL_TYPE}_config.json"
ENABLE_CHUNKED_PREFILL=${14}

PREDICTOR_WORKERS=${15}
GLOBAL_SCHEDULER_WORKERS=${16}
BACKEND_WORKERS=${17}
CHUNK_SIZE=${18}
QPS=${19}
BRANCH_NAME=${20}
BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION=${21}
N_SELECTED=${22}
PROFILING_SAMPLE_RATE=${23}
TIMEOUT_IN_SECONDS=${24}
USE_FOR_PROFILING_ONLY=${25}
PREDICTOR_TIMEOUT_IN_SECONDS=${26}
USE_PROCESS_FOR_FRONTEND=${27}

UPDATE_BLOCK_CODE=${28}
UPDATE_VLLM_CODE=${29}
RUN_EXP=${30}
USE_ESTIMATION_LEN=${31}
OUTPUT_DIR_PREFIX=${32}

# Setting for auto provisioning
AVAILABLE_INSTANCE=${33}
MAX_SLO=${34}
ENABLE_PREEMPTIVE_AUTO_PROVISIONING=${35}


if [ "$ENABLE_CHUNKED_PREFILL" = "true" ]; then
  MAX_NUM_BATCHED_TOKEN=$CHUNK_SIZE
else
  MAX_NUM_BATCHED_TOKEN=$MAX_MODEL_LENGTH
fi

# Current the v1 version of vllm is supported yet
VLLM_VERSION=0
case "$1" in
    -d|--daemon)
        $0 < /dev/null &> /dev/null & disown
        exit 0
        ;;
    *)
        ;;
esac


#rm -rf experiment_output
#mkdir -p experiment_output/logs

if [ "$RESTART_VLLM" = "true" ]; then
  parallel-ssh --host $TARGET_HOST "cd Block && rm experiment_output/logs/*"
  sh block/exp/reset.sh
  if [ "$UPDATE_BLOCK_CODE" = "true" ]; then
    parallel-ssh -t 0 -h block/config/hosts "cd Block && git checkout $BRANCH_NAME"
    parallel-ssh -t 0 -h block/config/hosts "cd Block && git add -u . && git stash && git reset --hard HEAD~1 && git pull"
  fi
  sleep 60
  nohup sh block/exp/run_exp_vllm.sh $BATCH_CAP $MODEL $UPDATE_VLLM_CODE $VLLM_VERSION $MAX_MODEL_LENGTH $ENABLE_CHUNKED_PREFILL $BACKEND_WORKERS $MAX_NUM_BATCHED_TOKEN > /dev/null 2>&1 &
  sleep 60
  script_base="block/exp/run_exp_predictor"
  if [ "$PREDICTOR_WORKERS" -eq 1 ]; then
    nohup sh "${script_base}_1.sh" $PREDICTOR_CONFIG_PATH $SCHEDULER_METRIC_TYPE $ENABLE_TIME_ESTIMATION $BATCH_CAP $ENABLE_CHUNKED_PREFILL $PREDICTOR_WORKERS $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
  else
    suffix_range=$(seq 1 7)
    for suffix in $suffix_range; do
      nohup sh "${script_base}_${suffix}.sh" $PREDICTOR_CONFIG_PATH $SCHEDULER_METRIC_TYPE $ENABLE_TIME_ESTIMATION $BATCH_CAP $ENABLE_CHUNKED_PREFILL $PREDICTOR_WORKERS $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
    done
    sleep 10
    suffix_range=$(seq 8 $PREDICTOR_WORKERS)
    for suffix in $suffix_range; do
      nohup sh "${script_base}_${suffix}.sh" $PREDICTOR_CONFIG_PATH $SCHEDULER_METRIC_TYPE $ENABLE_TIME_ESTIMATION $BATCH_CAP $ENABLE_CHUNKED_PREFILL $PREDICTOR_WORKERS $BRANCH_NAME $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
    done
    sleep 60
  fi
fi

if [ "$RUN_EXP" = "true" ]; then
  NUM_QUERIES=$NUM_DATA
  # Still use random for global scheduler but use min_latency for predictor
  METRIC_TYPES=$SCHEDULER_METRIC_TYPE
  if [ "$USE_FOR_PROFILING_ONLY" = "true" ]; then
    METRIC_TYPES="random"
  else
    METRIC_TYPES=$SCHEDULER_METRIC_TYPE
  fi
  for qps in $QPS; do
      for num_queries in $NUM_QUERIES; do
        for metric_type in $METRIC_TYPES; do
          if [ "$metric_type" = "min_new_request_latency" ]; then
            N=$N_SELECTED
            USE_ESTIMATION_LEN=$USE_ESTIMATION_LEN
          else
            N="12"
            USE_ESTIMATION_LEN=$USE_ESTIMATION_LEN
          fi
          for n in $N; do
              for use_estimation_len in $USE_ESTIMATION_LEN; do
                  echo "Running experiment with scheduler: $metric_type, model: $MODEL, dataset: $DATASET_NAME, qps: $qps, batch_size_cut: $BATCH_SIZE_THRESHOLD_FOR_TIME_ESTIMATION enable_chunked_prefill: $ENABLE_CHUNKED_PREFILL use_for_profiling_only: $USE_FOR_PROFILING_ONLY predictor timeout: $PREDICTOR_TIMEOUT_IN_SECONDS Profiling sample rate: $PROFILING_SAMPLE_RATE enable preemptive auto provisioning: $ENABLE_PREEMPTIVE_AUTO_PROVISIONING waiting time SLO: $MAX_SLO"
                  nohup sh block/exp/run_exp_global_scheduler.sh $TARGET_HOST $n $n $metric_type $HOST_CONFIG_PATH $GLOBAL_SCHEDULER_WORKERS $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO $ENABLE_PREEMPTIVE_AUTO_PROVISIONING > /dev/null 2>&1 &
                  LOG_FILENAME="benchmark.log"
                  OUTPUT_DIR="${OUTPUT_DIR_PREFIX}/${DATASET_TYPE}/${metric_type}/qps_${qps}_num_queries_${num_queries}_n_${n}_chunked_${ENABLE_CHUNKED_PREFILL}_predictor_${PREDICTOR_WORKERS}_global_${GLOBAL_SCHEDULER_WORKERS}_len_estimated_${use_estimation_len}_max_slo_${MAX_SLO}_enable_preemptive_auto_provisioning_${ENABLE_PREEMPTIVE_AUTO_PROVISIONING}_batch_${BATCH_CAP}_chunk_${CHUNK_SIZE}"
                  sleep 10
                  if [ "$use_estimation_len" = "true" ]; then
                    parallel-ssh -i -t 0 --host $TARGET_HOST "cd Block && export PYTHONPATH=. && export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.10/dist-packages/cusparselt/lib && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $num_queries --dataset_type $DATASET_TYPE --dataset_path $DATASET_PATH --qps $qps --backend block --log_filename $LOG_FILENAME --output_dir $OUTPUT_DIR  --data_start_index $START_INDEX --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS --use_estimated_response_lens"
                  else
                    parallel-ssh -i -t 0 --host $TARGET_HOST "cd Block && export PYTHONPATH=. && export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.10/dist-packages/cusparselt/lib && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $num_queries --dataset_type $DATASET_TYPE --dataset_path $DATASET_PATH --qps $qps --backend block --log_filename $LOG_FILENAME --output_dir $OUTPUT_DIR  --data_start_index $START_INDEX --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS"
                  fi

                  sleep 10
                  parallel-ssh --host $TARGET_HOST "cd Block && mkdir experiment_output/$OUTPUT_DIR/running_logs"
                  parallel-ssh --host $TARGET_HOST "cd Block && mv experiment_output/logs/* experiment_output/$OUTPUT_DIR/running_logs/."
                  done
              done
          done
      done
  done
fi



