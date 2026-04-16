# Block Deployment Guide

## Prerequisites

### Installation Order (CRITICAL)

```bash
# 1. vLLM FIRST (precompiled wheels)
cd ~/vllm && git checkout block
VLLM_USE_PRECOMPILED=1 pip install --editable .

# 2. PyTorch AFTER vLLM (ABI must match)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# 3. transformers pinned (5.x breaks vLLM block branch)
pip install transformers==4.50.3

# 4. Environment variables (set BEFORE any vLLM process)
export VLLM_USE_V1=0          # Required: V1 engine lacks schedule_trace API
export HF_TOKEN=your_token     # Required: Llama-2 model access
```

### CloudLab Setup

```bash
# Generate config from CloudLab manifest XML
python block/exp/generate_config.py \
    --user_name YOUR_USERNAME \
    --manifest_path block/cl_manifest.xml \
    --cluster_type a30 \
    --tensor_parallel_size 1 \
    --num_predictors 16

# Run automated setup on all nodes
sh block/exp/setup.sh
```

## A30 Cluster (12 nodes, Llama-2-7B)

### Step 1: Start vLLM on all nodes
```bash
sh block/exp/run_exp_vllm.sh 48 "meta-llama/Llama-2-7b-hf" false 0 4096 true 1 512 true
# Args: batch_size model enable_prefix_cache swap_space max_model_len chunked_prefill tp_size chunk_size enable_chunked_prefill
```

### Step 2: Start predictors (16 per node, deploy in batches)
```bash
for i in $(seq 1 16); do
    sh block/exp/run_exp_predictor_${i}.sh \
        block/config/llama_config.json \
        min_new_request_latency true 48 true 1 main 0 2000 &
    if [ $((i % 8)) -eq 0 ]; then sleep 10; fi  # Avoid OOM from concurrent model loading
done
```

### Step 3: Start global scheduler
```bash
python block/global_scheduler/api_server.py \
    --config_path block/config/host_configs.json \
    --metrics_type min_new_request_latency \
    --num_query_predictor 12 \
    --num_required_predictor 12 \
    --workers 1 \
    --num_predictor_ports 16 \
    --profiling_sampling_rate 0.0 \
    --predictor_timeout 2000 \
    --backend_timeout 1800 \
    --initial_available_instance 12
```

### Step 4: Run benchmark
```bash
python -m block.benchmark.benchmark_serving \
    --ip_ports 127.0.0.1:8200 \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --num_sampled_requests 10000 \
    --dataset_type sharegpt \
    --dataset_path ~/Block/data/trace_data/sharegpt/generate/llama \
    --qps 30 \
    --backend block \
    --trust_remote_code \
    --max_request_len 4096 \
    --timeout_in_seconds 3600
```

## A100 Cluster (2 nodes, Llama-2-70B, TP=4)

### Profiling First (required for new GPU SKU)

```bash
# Mount NVMe for model storage
ssh node "sudo mkfs.ext4 -F /dev/nvme0n1 && \
    sudo mkdir -p /mydata && sudo mount /dev/nvme0n1 /mydata && \
    sudo chown $(whoami) /mydata"
export HF_HOME=/mydata/huggingface

# MLP profiling (~5 min)
python vidur/profiling/mlp/main.py \
    --models "meta-llama/Llama-2-70b-hf" \
    --num_gpus 4 --num_tensor_parallel_workers 4 --max_tokens 4096

# Attention profiling (~2 min)
python vidur/profiling/attention/main.py \
    --models "meta-llama/Llama-2-70b-hf" \
    --num_gpus 4 --num_tensor_parallel_workers 4 --max_model_len 4096

# Collectives profiling (~10 min)
python vidur/profiling/collectives/main.py \
    --num_workers_per_node_combinations 4 --collective all_reduce
python vidur/profiling/collectives/main.py \
    --num_workers_per_node_combinations 2 --collective send_recv
```

### Deploy vLLM with TP=4
```bash
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --port 8000 --tensor-parallel-size 4 \
    --max-num-seqs 48 --max-model-len 4096 \
    --enable-chunked-prefill
```

### Get num_blocks (after vLLM starts)
```bash
curl -s http://127.0.0.1:8000/simple_schedule_trace | jq '.free_gpu_blocks'
# Update block/config/llama70b_a100_40gb_config.json with this value
```

## Common Issues

| Issue | Error | Fix |
|-------|-------|-----|
| V1 engine | `Can't instantiate abstract class AsyncLLM` | `export VLLM_USE_V1=0` |
| transformers 5.x | `AttributeError: all_special_tokens_extended` | `pip install transformers==4.50.3` |
| PyTorch before vLLM | `ImportError: undefined symbol` | Reinstall vLLM with VLLM_USE_PRECOMPILED=1 |
| A30 MIG | vLLM init fails | `sudo nvidia-smi -mig 0` |
| Port 8200 conflict | Predictor can't bind | Reserve 8200 for scheduler only |
| OOM on predictor deploy | Concurrent model loading | Deploy in batches of 8, sleep 10s between |
| Predictor slow warmup | First prediction ~5s | Send warmup request before benchmark |
