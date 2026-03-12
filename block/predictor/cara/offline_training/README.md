# CARA Offline Training Data Preparation

Unified preprocessing tool for preparing training data from CARA benchmark results.

## Overview

This pipeline processes benchmark data (single-model or multi-model) to create clean training datasets for:
1. **Length prediction**: prompt → output_length per model
2. **Model quality estimation**: prompt → quality_score per model

**Key Features:**
- Auto-detects data format (single-model vs multi-model with `broadcast_results`)
- Auto-detects all models from data (scans all requests for robustness)
- Handles incomplete requests (exports to JSONL for re-running benchmark)
- Multiple quality scoring methods: `llm_judge`, `similarity`, `compression`
- Judge comparison analysis with correlation metrics
- Divergent sample detection and export

## Pipeline Components

```
Raw Benchmark Data (JSON)
         ↓
   Format Detection (single-model or multi-model)
         ↓
   Model Collection (scan all requests)
         ↓
   Response Filtering (errors, empty, truncated, repetition)
         ↓
   Quality Scoring (llm_judge / similarity / compression)
         ↓
   Incomplete Request Export (JSONL for re-running)
         ↓
Training Data (JSON)
```

### Data Formats Supported

**Single-model format** (direct response in response_details):
```json
{
  "response_details": [
    {
      "request_id": "1",
      "prompt": "...",
      "model": "Qwen/Qwen2.5-3B",
      "response": "...",
      "output_len": 256
    }
  ]
}
```

**Multi-model format** (multiple models via `broadcast_results`):
```json
{
  "response_details": [
    {
      "request_id": "1",
      "prompt": "...",
      "broadcast_results": [
        {"model": "Qwen/Qwen2.5-3B", "generated_text": "...", "output_tokens": 200},
        {"model": "Qwen/Qwen2.5-72B", "generated_text": "...", "output_tokens": 250}
      ]
    }
  ]
}
```

### Response Filtering

Filters out low-quality responses:
- **Error checking**: Removes failed requests
- **Empty detection**: Removes responses with ≤1 token
- **Length filtering**: Configurable min/max output tokens
- **Truncation detection**: Flags (optionally filters) responses hitting max_length
- **Repetition detection**: Uses zlib compression ratio (< 0.2 = repetitive, optional filter)

### Quality Scorers

#### `compression` (Fast, no dependencies)
- Uses zlib compression ratio as quality proxy
- Higher compression ratio = more diverse/less repetitive = better quality
- **Best for**: Quick processing, no GPU needed

#### `similarity` (Requires multi-model data)
- Uses sentence-transformers embeddings
- Computes cosine similarity to reference model
- Requires `--reference-model` argument
- **Best for**: Comparing model outputs semantically

#### `llm_judge` (Most accurate, GPU recommended)
- Uses LLM to evaluate quality (correctness, helpfulness, coherence)
- Supports multiple judges for comparison analysis
- Supports batched inference for faster processing
- Default: `Qwen/Qwen2.5-0.5B`
- **Best for**: Nuanced quality assessment, safety-mixed datasets

## Usage

### Basic Usage (Single-Model Data)

```bash
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input cara_result/cara-benchmark.json \
  --scoring-method compression \
  --dataset-name sharegpt-3b
```

### Multi-Model with Similarity Scoring

```bash
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input data/cara/cara-best-route-training.json \
  --scoring-method similarity \
  --reference-model "Qwen/Qwen2.5-72B" \
  --device cuda
```

### LLM Judge Scoring

```bash
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input cara_result/cara-benchmark.json \
  --scoring-method llm_judge \
  --judge-models Qwen/Qwen2.5-0.5B \
  --device cuda \
  --batch-size 8
```

### Multi-Judge Comparison

Compare multiple LLM judges to analyze scoring agreement:

```bash
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input cara_result/cara-benchmark.json \
  --judge-models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-3B \
  --compare-judges \
  --divergence-threshold 0.3
```

**Output files:**
- `{output}.json`: Training data with scores from all judges
- `{output}_divergent.json`: Samples where judges disagree (diff > threshold)

**Analysis includes:**
- Per-judge statistics (mean, std, range, percentiles)
- Global Pearson and Spearman correlation between judges
- Per-request model ranking correlation (for multi-model data)
- Score difference distribution

### Handling Incomplete Requests

For incomplete requests (missing some models), the tool exports them in JSONL format for re-running:

```bash
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input data/cara/broadcast_results.json \
  --expected-models "Qwen/Qwen2.5-3B" "Qwen/Qwen2.5-7B" "Qwen/Qwen2.5-72B"
```

**Output:**
- `{output}_incomplete.jsonl`: Incomplete requests in benchmark format:
  ```json
  {"id": 0, "source": "incomplete/request_id", "prompt": "..."}
  ```

Re-run benchmark with the incomplete requests:
```bash
python -m block.benchmark.cara.benchmark_serving \
  --dataset-name custom \
  --dataset-path output_incomplete.jsonl \
  ...
```

### Allow Incomplete Requests

By default, only requests with all expected models are kept. To allow partial coverage:

```bash
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input data/cara/broadcast_results.json \
  --allow-incomplete
```

### Filtering Options

```bash
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input data/cara/benchmark.json \
  --min-output-tokens 3 \
  --max-output-tokens 1024 \
  --filter-truncated \
  --filter-high-repetition \
  --min-compression-ratio 0.2
```

## Command Line Arguments

### Input/Output
- `-i, --input`: Input benchmark JSON file (required)
- `-o, --output`: Output JSON file (default: auto-generated)
- `--dataset-name`: Dataset name for output metadata (default: `benchmark`)

### Model Configuration
- `--expected-models`: Override auto-detected models (space-separated)
- `--allow-incomplete`: Allow requests missing some models

### Filtering
- `--min-output-tokens`: Minimum valid output length (default: 3)
- `--max-output-tokens`: Max output for truncation detection (default: 1024)
- `--filter-truncated`: Filter out truncated responses
- `--filter-high-repetition`: Filter out highly repetitive responses
- `--min-compression-ratio`: Threshold for repetition (default: 0.2)

### Quality Scoring
- `--scoring-method`: `llm_judge`, `similarity`, `compression`, `none` (default: `llm_judge`)
- `--reference-model`: Reference model for similarity scoring (required for similarity)
- `--judge-models`: Judge model(s) for llm_judge (default: `Qwen/Qwen2.5-0.5B`)
- `--compare-judges`: Enable judge comparison analysis
- `--divergence-threshold`: Score diff threshold for divergent samples (default: 0.3)
- `--device`: Device for scoring models (default: `cpu`)
- `--batch-size`: Batch size for LLM judge (default: 8)

### Output Options
- `--include-response`: Include full response text in output
- `--debug`: Enable debug logging

## Output Format

```json
{
  "dataset_name": "benchmark",
  "scoring_method": "llm_judge",
  "num_requests": 13500,
  "models": [
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-3B"
  ],
  "requests": [
    {
      "request_id": "bench-599cf8ad-0",
      "prompt": "<|im_start|>system\nYou are...",
      "input_len": 28,
      "models": {
        "Qwen/Qwen2.5-72B": {
          "output_length": 271,
          "quality_score": 0.85,
          "compression_ratio": 0.45,
          "is_truncated": false,
          "ttft": 0.0503,
          "server_latency": 1.6928,
          "instance_id": "Qwen-2.5-72B_0",
          "host": "d8545-10s10305.wisc.cloudlab.us"
        },
        "Qwen/Qwen2.5-32B": {
          "output_length": 245,
          "quality_score": 0.82,
          "compression_ratio": 0.42,
          "is_truncated": false,
          "ttft": 0.0421,
          "server_latency": 1.4521,
          "instance_id": "Qwen-2.5-32B_0",
          "host": "d8545-10s10301.wisc.cloudlab.us"
        }
      }
    }
  ]
}
```

## Embedding Models (for similarity scoring)

### Fast (Default)
- `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~80MB)
  - Best for: Quick processing
  - Speed: ~3000 sent/sec on CPU

### Better Quality
- `sentence-transformers/all-mpnet-base-v2` (768-dim, ~420MB)
  - Best for: Better similarity accuracy
  - Speed: ~1000 sent/sec on CPU

### Lightweight
- `sentence-transformers/all-MiniLM-L12-v2` (384-dim, ~120MB)
  - Best for: Balance between speed and quality
  - Speed: ~2000 sent/sec on CPU

## Judge Models (for LLM judge scoring)

### Fast (Default)
- `Qwen/Qwen2.5-0.5B` (0.5B, very fast)

### Recommended
- `Unbabel/M-Prometheus-7B` (7B, specialized for evaluation)
- `prometheus-eval/prometheus-7b-v2.0` (7B, alternative)

### Lightweight
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B, faster but less accurate)

## Dependencies

```bash
# Core dependencies
pip install transformers torch numpy scipy

# For similarity scoring
pip install sentence-transformers

# Optional: for faster processing
pip install accelerate
```

## Example Workflow

```bash
# 1. Preprocess single-model benchmark data
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input cara_result/cara-3b-benchmark.json \
  --scoring-method compression \
  --dataset-name sharegpt-3b

# 2. Or preprocess multi-model broadcast data with LLM judge
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input data/cara/cara-best-route-training.json \
  --scoring-method llm_judge \
  --judge-models Qwen/Qwen2.5-0.5B \
  --device cuda

# 3. Use the processed data for training
python -m block.predictor.cara.offline_training.train_length_predictor \
  --input data/cara/benchmark_llm_judge_training.json \
  --model-name Qwen/Qwen2.5-72B
```

## Statistics Example

```
============================================================
PROCESSING STATISTICS
============================================================
Total requests: 20000
Total model responses: 80000
Valid responses: 75234
Valid requests (with all models): 18500
Incomplete requests: 1500
Filtered responses (empty): 712
Filtered responses (too short): 234
Filtered responses (truncated): 1620
Filtered responses (error): 0
Filtered responses (high repetition): 2200
============================================================
```

## Troubleshooting

### Out of Memory (OOM)

For large datasets or LLM judge scoring:
```bash
# Use CPU device
--device cpu

# Or use smaller batch size
--batch-size 4

# Or use smaller embedding model
--embedding-model sentence-transformers/all-MiniLM-L6-v2
```

### LLM Judge Fails to Parse Ratings

The LLM judge may output non-numeric text. Failed scores are marked as `None` and requests with failures are filtered out. Check the log for:
```
WARNING - Failed to parse rating for model: 'some text'. Marking as invalid (None)
```

### Similarity Scoring Requires Reference Model

```bash
# Error: --scoring-method=similarity requires --reference-model
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input data.json \
  --scoring-method similarity \
  --reference-model "Qwen/Qwen2.5-72B"  # Add this
```

### Slow Processing

```bash
# Use GPU for scoring
--device cuda

# Increase batch size for LLM judge
--batch-size 16

# Use smaller/faster embedding model for similarity
--embedding-model sentence-transformers/all-MiniLM-L6-v2
```

## Comparing Scripts

| Script | Purpose | Data Format |
|--------|---------|-------------|
| `prepare_benchmark_data.py` | Unified preprocessing (recommended) | Single-model or multi-model |
| `prepare_training_data.py` | Legacy multi-model preprocessing | Multi-model (broadcast_results) only |

**Note:** `prepare_benchmark_data.py` is the recommended unified script that handles both data formats and includes all features (auto-detection, incomplete request handling, multi-judge comparison).