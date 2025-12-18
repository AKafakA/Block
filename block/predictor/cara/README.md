# CARA Predictor Implementation

## Overview

This directory contains the CARA (Co-optimizing Super Heterogeneous LLM Serving) predictor implementation. Unlike Block's simulation-based predictor (which uses Vidur), CARA predictors use learned sequence models for forward-compatible, adaptive performance prediction.

## Implementation Status

### Phase 1: Data Collection (COMPLETED)
- **Dummy CARA Predictor**: Collects training data while using simple heuristics
- **Training Data Collector**: Automatic logging of prediction contexts and actual metrics
- **Predictor API Server**: Independent from Block's predictor infrastructure

### Phase 2: Offline Training (TODO)
- LSTM sequence-to-value regression model
- Training pipeline for collected data

### Phase 3: Production LSTM Predictor (TODO)
- Load pre-trained checkpoints
- Real-time inference

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│               CARA Scheduler (cara_serve.py)            │
└────────────┬────────────────────────────┬───────────────┘
             │                            │
    ┌────────▼────────┐          ┌───────▼────────┐
    │  vLLM Instance  │          │  vLLM Instance │
    │  Port: 8000     │          │  Port: 8000    │
    │                 │          │                │
    │  Predictor      │          │  Predictor     │
    │  Port: 8100 ◄───┼──────────┼────► Port: 8101│
    └─────────────────┘          └────────────────┘
           │                              │
           │   /schedule_trace            │
           │   /log_actual (feedback)     │
           └──────────────────────────────┘
```

## File Structure

```
block/predictor/cara/
├── README.md                       # This file
├── data_structures.py              # PredictRequest, ScheduleState, TrainingExample
├── cara_predictor_config.py        # Config hierarchy (Base, Dummy, LSTM)
├── base_predictor.py               # CARA predictor base class
├── dummy_cara_predictor.py         # Dummy predictor with data collection
├── schedule_trace_client.py        # Client for vLLM /schedule_trace API
├── training_data_collector.py      # Automatic training data logging
├── cara_predictor_api_server.py    # Predictor API server (independent from Block)
└── __init__.py

block/config/cara/
├── predictor_config_dummy.json     # Example dummy predictor config

block/global_scheduler/cara/cara_instance/
├── Instance.py                     # Updated with predictor feedback mechanism
├── vllm_instance.py                # Updated constructors
└── ollama_instance.py              # Updated constructors
```

## Key Design Decisions

### 1. Independent from Block/Vidur
- CARA predictors use `PredictRequest` dataclass (not Vidur `Request`)
- Separate API server (`cara_predictor_api_server.py` vs `api_server.py`)
- No dependency on Vidur simulation

### 2. Type-Based Config Hierarchy
```python
CARABasePredictorConfig
├── DummyPredictorConfig    # For data collection phase
└── LSTMPredictorConfig     # For production (TODO)
```

### 3. Automatic Data Collection Pipeline
1. **Prediction**: Predictor logs (schedule_state, target_request)
2. **Execution**: Instance executes request
3. **Feedback**: Instance sends actual metrics to `/log_actual`
4. **Matching**: Collector matches prediction with ground truth
5. **Saving**: Batched writes to JSONL files

## Usage Guide

### Step 1: Start CARA Predictor Server

```bash
# Start dummy predictor for instance on port 8000
python -m block.predictor.cara.cara_predictor_api_server \
  --host 0.0.0.0 \
  --port 8100 \
  --instance-port 8000 \
  --config-path block/config/cara/predictor_config_dummy.json
```

### Step 2: Enable Predictor Feedback in Instances

When creating instances in `cara_serve.py`, enable feedback:

```python
instance = VllmInstance(
    instance_id="model_0",
    hostname="node1",
    ip_address="192.168.1.1",
    predictor_ports=[8100],  # Predictor API port
    model_name="Qwen/Qwen2.5-7B",
    backend_port=8000,
    enable_predictor_feedback=True,  # Enable data collection
    feedback_sample_rate=1.0          # Collect 100% of requests
)
```

### Step 3: Run Experiments

```bash
# Run CARA scheduler (will call predictors and send feedback)
python -m block.global_scheduler.cara.cara_serve \
  --host 0.0.0.0 \
  --port 8200 \
  --model_config_path block/config/cara/model_deployment.json \
  --host_config block/config/host_configs.json
```

### Step 4: Monitor Data Collection

```bash
# Check collection stats
curl http://localhost:8100/stats

# View collected data
ls -lh training_data/cara/
cat training_data/cara/training_data_instance_8000_*.jsonl | jq .
```

## Configuration

### Dummy Predictor Config

`block/config/cara/predictor_config_dummy.json`:
```json
{
  "predictor_type": "dummy",
  "backend_port": 8000,
  "schedule_trace_timeout": 5,
  "enable_data_collection": true,
  "data_collection_sample_rate": 1.0,
  "data_output_dir": "./training_data/cara",
  "save_batch_size": 100,
  "heuristic_mode": "min_requests"
}
```

**Heuristic Modes**:
- `min_requests`: Prefer instances with fewer total requests
- `max_gpu_blocks`: Prefer instances with more free GPU blocks
- `combined`: Weighted combination of both
- `random`: Random assignment (baseline)

### Training Data Format

Each JSONL file contains training examples:
```json
{
  "request_id": "cmpl-exp1-123-0",
  "num_prompt_tokens": 128,
  "num_predicted_output_tokens": 64,
  "schedule_state": {
    "num_running": 3,
    "num_waiting": 2,
    "free_gpu_blocks": 4500,
    "num_preempted": 0,
    "running_requests": [...],
    "waiting_requests": [...]
  },
  "instance_id": "instance_8000",
  "prediction_timestamp": 1702345678.123,
  "actual_e2e_latency": 2.456,
  "actual_ttft": 0.123,
  "actual_tpot": 0.023,
  "completion_timestamp": 1702345680.579
}
```

## API Endpoints

### Predictor API Server (Port 8100)

#### POST /predict
**Request**:
```json
{
  "request_id": "123",
  "num_prompt_tokens": 128,
  "num_predicted_output_tokens": 64
}
```

**Response**:
```json
{
  "target_metric": 5.0,
  "gpu_blocks": 4500,
  "num_requests": 5,
  "num_preempted": 0,
  "predictor_type": "dummy_cara",
  "time_to_predict": 12.34
}
```

#### POST /log_actual
**Request**:
```json
{
  "request_id": "123",
  "e2e_latency": 2.456,
  "ttft": 0.123,
  "tpot": 0.023
}
```

#### GET /stats
**Response**:
```json
{
  "total_predictions": 1000,
  "total_completed": 950,
  "total_saved": 900,
  "pending": 50,
  "buffered": 50
}
```

#### GET /health
Health check endpoint.

### vLLM Backend API (Port 8000)

#### GET /schedule_trace
**Response** (new 4-field format):
```json
{
  "running": [
    "cmpl-exp1-1-0", 128, 45, 64,
    "cmpl-exp1-2-0", 256, 120, 128
  ],
  "waiting": [
    "cmpl-exp1-3-0", 64, 0, 32
  ],
  "free_gpu_blocks": 4500,
  "num_preempted": 0
}
```

**Format**: Each request is 4 consecutive fields:
1. `request_id` (str)
2. `num_prompt_tokens` (int)
3. `num_computed_tokens` (int)
4. `num_predicted_output_tokens` (int)

## Differences from Block Predictor

| Aspect | Block Predictor | CARA Predictor |
|--------|----------------|----------------|
| **Prediction Method** | Vidur simulation | Learned sequence model (future) |
| **Interface** | Vidur `Request` | CARA `PredictRequest` |
| **Config** | `PredictorConfig` (heavy) | `CARABasePredictorConfig` (minimal) |
| **Data Collection** | None | Automatic with feedback loop |
| **Backend API** | 7-field schedule trace | 4-field schedule trace |
| **Adaptability** | Static (needs reconfig) | Forward-compatible (learns online) |

## Next Steps

### Phase 2: Offline Training
1. **Data Preprocessing**
   - Load collected JSONL files
   - Split train/val/test sets
   - Normalize features

2. **Model Implementation**
   - LSTM sequence encoder for request queue
   - Multi-output head (E2E latency, TTFT, TPOT)
   - Training script with checkpointing

3. **Model Evaluation**
   - Offline validation metrics (MAE, RMSE)
   - Prediction latency overhead
   - Model size analysis

### Phase 3: Production Deployment
1. **LSTM Predictor**
   - `block/predictor/cara/cara_predictor.py`
   - Load checkpoint
   - Real-time inference

2. **Integration with CARA Scheduler**
   - Use predictions for scheduling decisions
   - Compare vs dummy heuristics

## Troubleshooting

### Predictor not receiving feedback
- Check `enable_predictor_feedback=True` in instance creation
- Verify predictor ports in host config
- Check logs: `tail -f experiment_output/logs/predictor_output.log`

### No training data being saved
- Verify `enable_data_collection=True` in config
- Check `data_collection_sample_rate` (should be > 0)
- Ensure output directory is writable

### /schedule_trace timeout
- Increase `schedule_trace_timeout` in config
- Check vLLM instance is running
- Verify network connectivity

## References

- **CARA Paper**: Co-optimizing Super Heterogeneous LLM Serving
- **vLLM /schedule_trace PR**: [Link to PR when merged]
- **Block Project**: `block/predictor/` for comparison