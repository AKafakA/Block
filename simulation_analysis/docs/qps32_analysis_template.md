# QPS=32 Scheduler Analysis (Template)

## Experiment Metadata
- Trace: data/trace_data/sharegpt/sharegpt_val_10k_llama2.csv
- Predicted trace: data/trace_data/sharegpt/generate/llama/sharegpt-llama-7b-val-10k-predicted.json
- Replicas: 12
- QPS: 32
- Noise: 10% uniform (deterministic when fast path enabled)
- Seed: 42

## Runtime Summary
- Small-slice speedup (12×240 requests):
  - Baseline Block (fast_predict=off): TODO s
  - Optimized Block (fast_predict=on, parallel): TODO s
  - Speedup: TODO×
  - Notes: Full-trace sequential baseline omitted due to runtime; fast-path used for all full runs.

## Metric Parity
| Scheduler | Mismatched Cells | Notes |
|-----------|------------------|-------|
| block_offline | TODO | |
| block_star_offline | TODO | |

## Latency Metrics (p50/p95/p99)
Headline metrics highlighted: E2E latency and TTFT.
Current heuristics (from `experiment_output/offline/qps32_full`):
- infass_pp: p50 6.26s / p95 17.68s / p99 21.29s
- llumnix_minus: p50 6.24s / p95 17.75s / p99 21.30s
- random: p50 6.93s / p95 21.09s / p99 28.40s
- round_robin: p50 6.44s / p95 18.90s / p99 24.27s

_Add Block/Block* entries once parity runs finish._

## Observations
- TODO: describe relative gains vs heuristics and alignment with online experiments.

## Action Items
- [ ] Validate parity using `compare_block_runs.py`.
- [ ] Update table with latency percentiles.
- [ ] Cross-check against online paper results.
- [ ] Integrate into appendix draft.
