# CARA Offline Training Data Preparation

Tools for preparing training data for CARA model estimation from benchmark results.

## Overview

This pipeline processes broadcast benchmark data to create clean training datasets for:
1. **Length prediction**: prompt → output_length per model
2. **Model quality estimation**: prompt → quality_score per model

## Pipeline Components

```
Raw Benchmark Data (JSON)
         ↓
   ResponseFilter (filter bad responses)
         ↓
   Tokenizer (recount tokens)
         ↓
   ModelScorer (compute quality scores)
         ↓
Training Data (JSON)
```

### 1. Response Filtering (`response_filter.py`)

Filters out low-quality responses:
- **Error checking**: Removes failed requests
- **Truncation detection**: Filters responses hitting max_length (incomplete)
- **Repetition detection**: Uses zlib compression ratio (< 0.2 = repetitive)

### 2. Model Scorers

#### `similarity_scorer.py` (Default)
- Uses sentence-transformers embeddings
- Computes cosine similarity to reference model (largest by default)
- Fast, no LLM needed
- **Best for**: Quick processing, semantic similarity

#### `llm_judge_scorer.py`
- Uses separate LLM to judge quality
- Evaluates correctness, helpfulness, harmlessness, coherence
- Default: `Unbabel/M-Prometheus-7B`
- **Best for**: Nuanced quality assessment

## Usage

### Basic Usage (Similarity Scoring)

```bash
python -m block.predictor.cara.offline_training.prepare_training_data \
  --input data/cara/cara-best-route-training.json \
  --tokenizer Qwen/Qwen2.5-72B \
  --scoring-method similarity \
  --device cpu
```

**Output**: `data/cara/best-route_similarity_model_estimation_training.json`

### Advanced Options

```bash
python -m block.predictor.cara.offline_training.prepare_training_data \
  --input data/cara/cara-best-route-training.json \
  --output data/cara/custom_output.json \
  --dataset-name best-route \
  --tokenizer Qwen/Qwen2.5-72B \
  --scoring-method similarity \
  --embedding-model sentence-transformers/all-mpnet-base-v2 \
  --reference-model "Qwen/Qwen2.5-72B" \
  --min-output-tokens 3 \
  --max-output-tokens 1024 \
  --min-compression-ratio 0.2 \
  --device cuda \
  --debug
```

### LLM Judge Scoring

```bash
python -m block.predictor.cara.offline_training.prepare_training_data \
  --input data/cara/cara-best-route-training.json \
  --tokenizer Qwen/Qwen2.5-72B \
  --scoring-method llm_judge \
  --judge-model Unbabel/M-Prometheus-7B \
  --device cuda
```

### Custom Judge Prompt

```bash
# Create custom_judge_prompt.txt with {prompt} and {response} placeholders
python -m block.predictor.cara.offline_training.prepare_training_data \
  --input data/cara/cara-best-route-training.json \
  --tokenizer Qwen/Qwen2.5-72B \
  --scoring-method llm_judge \
  --judge-prompt custom_judge_prompt.txt \
  --device cuda
```

### Excluding Models

When some models are unavailable or have deployment issues:

```bash
# Exclude a single model
python -m block.predictor.cara.offline_training.prepare_training_data \
  --input data/cara/cara-best-route-training.json \
  --exclude-models "Qwen/Qwen2.5-3B"

# Exclude multiple models
python -m block.predictor.cara.offline_training.prepare_training_data \
  --input data/cara/cara-best-route-training.json \
  --exclude-models "Qwen/Qwen2.5-3B" "Qwen/Qwen2.5-32B"
```

### Requiring All Models

When you want only requests with complete model coverage (all non-excluded models responded):

```bash
# Exclude 3B and require all remaining models (72B, 32B, 14B, 7B)
python -m block.predictor.cara.offline_training.prepare_training_data \
  --input data/cara/cara-best-route-training.json \
  --exclude-models "Qwen/Qwen2.5-3B" \
  --require-all-models

# This ensures every request has responses from 72B, 32B, 14B, and 7B
# Requests missing any of these models will be filtered out
```

**Benefits:**
- Ensures consistent model coverage across all training examples
- Prevents bias from partial model coverage
- Better for training quality estimators that compare all models

### Custom Reference Model

The reference model is used for similarity scoring (always gets score 1.0):

```bash
# Use 32B as reference instead of default 72B
python -m block.predictor.cara.offline_training.prepare_training_data \
  --input data/cara/cara-best-route-training.json \
  --reference-model "Qwen/Qwen2.5-32B"

# Important: Requests where the reference model's response is filtered
# will be completely excluded to ensure consistent scoring
```

## Command Line Arguments

### Required
- `--tokenizer`: HuggingFace tokenizer name or path

### Input/Output
- `--input`: Input JSON file (default: `data/cara/cara-best-route-training.json`)
- `--output`: Output JSON file (default: auto-generated)
- `--dataset-name`: Dataset name for output filename (default: `best-route`)

### Scoring
- `--scoring-method`: `similarity` or `llm_judge` (default: `similarity`)
- `--embedding-model`: Embedding model for similarity (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `--judge-model`: Judge LLM model (default: `Unbabel/M-Prometheus-7B`)
- `--judge-prompt`: Custom judge prompt template file
- `--reference-model`: Reference model for similarity (default: auto-detect largest)

### Filtering
- `--min-output-tokens`: Minimum valid output length (default: 3)
- `--max-output-tokens`: Max output before truncation (default: 1024)
- `--min-compression-ratio`: Min compression ratio (default: 0.2)
- `--exclude-models`: Models to exclude from processing (space-separated list)
- `--require-all-models`: Only keep requests where all non-excluded models have valid responses
- `--reference-model`: Reference model for similarity scoring (default: Qwen/Qwen2.5-72B)

### Other
- `--device`: Device for models (`cpu`, `cuda`)
- `--debug`: Enable debug logging

## Output Format

```json
{
  "dataset_name": "best-route",
  "scoring_method": "similarity",
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
          "quality_score": 1.0,
          "ttft": 0.0503,
          "server_latency": 1.6928,
          "instance_id": "Qwen-2.5-72B_0",
          "host": "d8545-10s10305.wisc.cloudlab.us"
        },
        "Qwen/Qwen2.5-32B": {
          "output_length": 245,
          "quality_score": 0.8723,
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

## Embedding Models

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

## Judge Models

### Recommended
- `Unbabel/M-Prometheus-7B` (7B, specialized for evaluation)
- `prometheus-eval/prometheus-7b-v2.0` (7B, alternative)

### Lightweight
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B, faster but less accurate)

## Dependencies

```bash
# Core dependencies
pip install transformers torch

# For similarity scoring
pip install sentence-transformers

# Optional: for faster processing
pip install accelerate
```

## Example Workflow

```bash
# 1. Prepare training data with similarity scoring
python -m block.predictor.cara.offline_training.prepare_training_data \
  --input data/cara/cara-best-route-training.json \
  --tokenizer Qwen/Qwen2.5-72B \
  --scoring-method similarity \
  --device cuda

# Output: data/cara/best-route_similarity_model_estimation_training.json

# 2. Use the processed data for training
python -m block.predictor.cara.offline_training.train_length_predictor \
  --input data/cara/best-route_similarity_model_estimation_training.json \
  --model-name Qwen/Qwen2.5-72B
```

## Statistics Example

```
================================================================================
PROCESSING STATISTICS
================================================================================
Total requests: 13500
Valid requests: 12845
Filtered requests: 655
Filtered responses: 2301
Models processed: 5
  - Qwen/Qwen2.5-14B
  - Qwen/Qwen2.5-32B
  - Qwen/Qwen2.5-3B
  - Qwen/Qwen2.5-72B
  - Qwen/Qwen2.5-7B

Top filter reasons:
  - Truncated: output_tokens=1024 (hit max_length=1024): 1523
  - High repetition detected: compression_ratio=0.156 (threshold=0.2): 534
  - Too short: 3 tokens: 178
  - Response marked as failed: timeout: 66
================================================================================

✅ Saved training data to: data/cara/best-route_similarity_model_estimation_training.json
   Requests: 12845
   Models: 5
     - Qwen/Qwen2.5-14B
     - Qwen/Qwen2.5-32B
     - Qwen/Qwen2.5-3B
     - Qwen/Qwen2.5-72B
     - Qwen/Qwen2.5-7B
   File size: 127.34 MB
```

## Troubleshooting

### Out of Memory (OOM)

For large datasets or LLM judge scoring:
```bash
# Use CPU device
--device cpu

# Or use smaller embedding model
--embedding-model sentence-transformers/all-MiniLM-L6-v2

# Or process in batches (for custom implementation)
```

### Tokenizer Issues

```bash
# Make sure to use the same tokenizer as the model
--tokenizer Qwen/Qwen2.5-72B  # If benchmarked with Qwen

# Add trust_remote_code if needed (handled automatically)
```

### Slow Processing

```bash
# Use GPU for scoring
--device cuda

# Use smaller/faster embedding model
--embedding-model sentence-transformers/all-MiniLM-L6-v2

# Reduce debug logging
# (remove --debug flag)
```