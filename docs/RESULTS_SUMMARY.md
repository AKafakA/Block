# Block Results Summary

## A30 Main Results (12 nodes, Llama-2-7B, ShareGPT)

### Capacity (TTFT P99 < 3s SLO)

| System | Max QPS | vs Llumnix-- |
|--------|---------|-------------|
| Block (N=12) | 31.1 | +1.3% |
| Block Po2 (N=2) | 30.2 | -1.3% (97.7% of full) |
| Block* (oracle) | 32.0 | +4.6% (ceiling) |
| Llumnix-- (baseline) | 30.6 | -- |
| Round-robin (bs=24) | 27.2 | -11.1% |

### Latency at QPS=30

| Metric | Block | Po2 | Llumnix-- |
|--------|-------|-----|-----------|
| Mean E2E (ms) | 10,923 | 10,780 | 10,938 |
| P99 E2E (ms) | 33,405 | 32,896 | 33,562 |
| TTFT P50 (ms) | 284 | 321 | 338 |
| TTFT P99 (ms) | 2,381 | 2,531 | 2,694 |
| Token lat P50 (ms) | 22.2 | 22.0 | 22.0 |

### Error Robustness (4x4 grid)

| Length error | Latency error | Capacity loss |
|-------------|--------------|---------------|
| 0% | 0% | 0% (baseline) |
| 25% | 25% | <1% |
| 50% | 50% | ~1-2% |
| 100% | 100% | ~2-3% |

### Burstiness (Gamma arrivals)

| α | Block QPS | RR QPS | Diff |
|---|-----------|--------|------|
| 0.25 (very bursty) | 30.8 | 30.4 | ±1% |
| 0.50 | 31.0 | 30.5 | ±1% |
| 1.00 (Poisson) | 31.1 | 30.7 | baseline |
| 2.00 (regular) | 31.2 | 30.8 | ±1% |

### CPU Overhead

| Config | Per-predictor CPU | Total (16 predictors) | Memory |
|--------|------------------|----------------------|--------|
| Standard | 21-32% | ~4 cores | ~1.8 GB/predictor |
| Po2 (N=2) | ~11-16% | ~2 cores | Same |

## A100 Results (2 nodes, Llama-2-70B TP=4)

### QPS=28 (normal load)

| System | Throughput (tok/s) | Mean lat (ms) | Token lat (ms) |
|--------|-------------------|---------------|----------------|
| Block | 15,009 | 6,599 | 20.97 |
| Llumnix sched-only | 14,872 | 6,423 | 20.40 |
| Llumnix migration | 15,047 | 7,635 | 30.15 |

### QPS=36 (high load — KEY FINDING)

| System | Throughput | Mean lat | Status |
|--------|-----------|----------|--------|
| Block | 18,798 | 7,505ms | Stable |
| Llumnix sched-only | 18,843 | 7,458ms | Stable |
| Llumnix migration | 12,936 | **89,027ms** | **COLLAPSED** |

Chunked prefill is more critical than KV migration for high-load serving.

## Prediction Accuracy

| Model | GPU | MAE | Metric |
|-------|-----|-----|--------|
| Llama-2-7B | A30 | ~12% | E2E latency |
| Llama-2-70B | A100-40GB | 16.6% | E2E latency at QPS=3 |
| RoBERTa length | - | 10-15% | Response token count |
