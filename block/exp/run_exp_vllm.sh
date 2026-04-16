BATCH_CAP=$1
MODEL=$2
UPDATE_CODE=$3
VLLM_VERSION=$4
MAX_MODEL_LENGTH=$5
ENABLE_CHUNKED_PREFILL=$6
NUM_WORKERS=$7
MAX_NUM_BATCHED_TOKEN=$8
USE_PROCESS_FOR_FRONTEND=$9
# HF_TOKEN sourced from remote node's ~/.bashrc


if [ "$UPDATE_CODE" = "true" ]; then
    parallel-ssh -t 0 -h block/config/hosts "cd vllm && sudo chown -R $(whoami) .git/ && git reset --hard HEAD~10 && git pull"
fi
if [ "$USE_PROCESS_FOR_FRONTEND" = "false" ]; then
    parallel-ssh -t 0 -h block/config/hosts "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.10/dist-packages/cusparselt/lib && source ~/.bashrc && cd vllm && mkdir -p ~/Block/experiment_output/logs && export VLLM_USE_V1=${VLLM_VERSION} && (nohup python -m vllm.entrypoints.api_server --workers $NUM_WORKERS --model=$MODEL --max-num-seqs $BATCH_CAP --trust-remote-code --max_model_len $MAX_MODEL_LENGTH --enable_chunked_prefill $ENABLE_CHUNKED_PREFILL --max-num-batched-tokens $MAX_NUM_BATCHED_TOKEN --swap-space 0 --disable-frontend-multiprocessing > ~/Block/experiment_output/logs/vllm.log 2>&1 &)"
else
    parallel-ssh -t 0 -h block/config/hosts "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.10/dist-packages/cusparselt/lib && source ~/.bashrc && cd vllm && mkdir -p ~/Block/experiment_output/logs && export VLLM_USE_V1=${VLLM_VERSION} && (nohup python -m vllm.entrypoints.api_server --workers $NUM_WORKERS --model=$MODEL --max-num-seqs $BATCH_CAP --trust-remote-code --max_model_len $MAX_MODEL_LENGTH --enable_chunked_prefill $ENABLE_CHUNKED_PREFILL --max-num-batched-tokens $MAX_NUM_BATCHED_TOKEN --swap-space 0 > ~/Block/experiment_output/logs/vllm.log 2>&1 &)"
fi
