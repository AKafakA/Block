# Block System Architecture

## Overview

Block is a prediction-guided global scheduler for multi-instance LLM serving on commodity clusters (no high-speed interconnects). It combines response-length estimation with simulation-based latency prediction to route requests optimally.

## Components

### 1. Length Tagger (`block/length_estimation/`)
- RoBERTa-base fine-tuned regressor (125M params)
- Input: prompt text -> Output: predicted response length (tokens)
- ~10-15% MAE; effective because ranking preservation matters more than absolute accuracy
- Pre-computes on datasets, tags ShareGPT/BurstGPT/ArXiv-Summ

### 2. Predictor Service (`block/predictor/`)
- Co-located sidecar on each inference node (4-16 per node, 192 total on 12-node A30 cluster)
- Takes snapshot of local vLLM queue state via `/schedule_trace`
- Runs Vidur simulation to predict latency for a hypothetical new request
- Scheduling policies: `min_new_request_latency` (default), `min_lunmnix_load`, `round_robin`, `random`
- Overhead: 62-80ms per prediction, ~21-32% CPU per predictor, ~1.8GB memory each

### 3. Global Scheduler (`block/global_scheduler/api_server.py`)
- Centralized router on one node (port 8200)
- Receives incoming request, broadcasts to N predictors (N=12 full, N=2 for Power-of-Two)
- Selects instance with lowest predicted latency
- Supports error injection (`length_error_pct`, `latency_error_pct`) for robustness testing
- Key params: `num_query_predictor`, `num_required_predictor`, `predictor_timeout` (2000ms)

### 4. vLLM Backend (AKafakA/vllm@block branch)
- Modified vLLM 0.5.4-0.7.2 with Block integration
- Exposes `/schedule_trace` and `/simple_schedule_trace` endpoints for queue state
- Critical: `VLLM_USE_V1=0` required (V1 engine lacks trace API)
- Settings: `--enable-chunked-prefill --max-num-batched-tokens 512 --max-num-seqs 48 --max-model-len 4096`

### 5. Vidur Simulator (`vidur/`)
- Fork of Vidur (OSDI'24) with Block modifications
- Simulates vLLM scheduling at per-iteration granularity
- Uses hardware-specific profiling data (MLP, attention, collective kernels)
- Per-GPU-SKU profiles: A30 (PCIe), A100-40GB (NVLink)

## Data Flow

```
Client Request
    |
    v
Global Scheduler (port 8200)
    |
    |--- broadcast to N predictor sidecars ---
    |         |              |              |
    |    Predictor_1    Predictor_2   Predictor_N
    |    (Vidur sim)    (Vidur sim)   (Vidur sim)
    |         |              |              |
    |    predicted_lat  predicted_lat  predicted_lat
    |         |              |              |
    |--- select min(predicted_latency) ---
    |
    v
Route to best vLLM instance (port 8000)
    |
    v
Response -> Client
```

## Configuration

### A30 Cluster (12 nodes, Llama-2-7B)
- 12x vLLM instances (1 per node, batch_size=48, chunk=512)
- 192 predictors (16 per node)
- 1 global scheduler (on any node)
- Config: `block/config/llama_config.json` + `block/config/host_configs.json`

### A100 Cluster (2 nodes, Llama-2-70B, TP=4)
- 2x vLLM instances (TP=4 each across 4 GPUs)
- 8 predictors (4 per node)
- 1 global scheduler
- Config: `block/config/llama70b_a100_40gb_config.json` + `block/config/a100_host_configs.json`
- Requires fresh profiling: MLP + attention + collectives for A100-40GB
