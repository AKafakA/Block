# The host with GPU to run the length estimation script
GPU_HOST = ""
BLOCK_GITHUB_LINK="https://github.com/AKafakA/vllm.git"

parallel-ssh -t 0 -h GPU_HOST "pip install -U pip==25.0.1"
parallel-ssh -t 0 -h GPU_HOST "pip install accelerate deepspeed einops fschat peft simpletransformers fsspec==2025.3.2"
parallel-ssh -t 0 -h GPU_HOST "git clone ${BLOCK_GITHUB_LINK} "
parallel-ssh -t 0 -h GPU_HOST "cd Block && python block/length_estimation/train_roberta.py"
parallel-ssh -t 0 -h GPU_HOST "cd Block && python block/length_estimation/eval_roberta.py"