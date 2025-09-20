# Large-Scale (10×) Simulation Analysis (Template)

## Experiment Metadata
- Replicas: 120
- QPS: 320
- Trace: data/trace_data/sharegpt/sharegpt_val_10k_llama2.csv
- Noise: 10% uniform (deterministic noise when fast path enabled)
- Seed: 42

## Runtime Summary
- Block baseline runtime: TODO s
- Block optimized runtime: TODO s
- Speedup: TODO×

## Scheduler Metrics
_To be filled using `summarize_request_metrics.py` once runs finish._

## Scaling Observations
- TODO: comment on latency scaling vs 12-replica case.
- TODO: highlight any queueing or restart behavior changes.

## Appendix Integration Notes
- Include concise paragraphs summarizing methodology and key findings.
- Prepare figure/table references for the paper appendix.
