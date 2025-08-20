parallel-ssh -i -t 0 -h block/config/hosts "cd block && rm -rf experiment_output/logs/* && mkdir -p experiment_output/logs"

parallel-ssh -h block/config/hosts "pkill -f vllm.entrypoints.api_server"
parallel-ssh -h block/config/hosts "pkill -f predictor"
parallel-ssh -h block/config/hosts "pkill -f multiprocessing.spawn"