HOST=$1
QPS=$2
NUM_QUERIES=$3
DATASET=$4
DATASET_TYPE=$5
MODEL=$6
EXPERIMENT_NAME=$7
GENERATE_NEW_DATA=$8
LOG_FILENAME=$9


parallel-ssh -t 0 --host $HOST "cd Block && export PYTHONPATH=. && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $NUM_QUERIES --dataset_type $DATASET_TYPE --dataset_path $DATASET --qps $QPS --backend block --log_filename $LOG_FILENAME --output_dir $EXPERIMENT_NAME --tag_dataset_with_real_response $GENERATE_NEW_DATA --enable_csv_files $GENERATE_NEW_DATA"