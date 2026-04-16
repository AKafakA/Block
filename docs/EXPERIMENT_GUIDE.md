# Block Experiment Guide

## Experiment Scripts

All scripts are in `block/exp/end_to_end_exp_scripts/`. Each script handles full deployment + benchmark + teardown.

### Main Results (A30, 12 nodes)

| Script | Purpose | Duration | Paper Section |
|--------|---------|----------|---------------|
| `a30_main/main_experiment.sh` | QPS sweep (20-36), capacity curves | ~8h | Fig 4-6 |
| `ablation/po2_ablation_exp.sh` | Power-of-Two (N=2) vs N=12 | ~6h | §6.8 |
| `ablation/burstiness_exp.sh` | Gamma arrivals (α=0.25, 0.5, 1.0, 2.0) | ~2h | §6.9 |
| `ablation/error_heatmap_exp.sh` | Length × latency noise grid (4×4) | ~4h | §6.10 |
| `ablation/cpu_tracking_experiment.sh` | Predictor CPU/memory overhead | ~1h | §6.11 |
| `ablation/extension_experiment.sh` | Qwen2-7B + BurstGPT dataset | ~4h | §6.12 |
| `ablation/config_search_experiment.sh` | Batch size × chunk size sweep | ~3h | Appendix |
| `ablation/block_nosim_ablation_exp.sh` | Block-NoSim (no Vidur) ablation | ~4h | §6.X |
| `ablation/prediction_experiment.sh` | Length prediction accuracy eval | ~1h | §6.6 |
| `ablation/auto_provision_exp.sh` | Auto-provisioning demonstration | ~2h | §6.5 |

### A100 Supplementary (2 nodes, Llama-70B)

| Script | Purpose | Duration | Paper Section |
|--------|---------|----------|---------------|
| `a100_supplementary/full_comparison.sh` | Block vs Llumnix (sched-only + migration) | ~4h | §6.7 |
| `a100_supplementary/a100_llama70b_exp.sh` | A100 latency sweep (QPS 2-40) | ~6h | Appendix |
| `a100_supplementary/a100_40gb_profiling.sh` | Fresh A100-40GB profiling | ~20min | Pre-req |

### Running an Experiment

```bash
# 1. Ensure cluster is deployed (DEPLOYMENT.md)
# 2. Set environment
export PYTHONPATH=$PWD
export VLLM_USE_V1=0

# 3. Run experiment (from repo root)
bash block/exp/end_to_end_exp_scripts/a30_main/main_experiment.sh

# 4. Results land in experiment_results_a30/ (gitignored)
# 5. Plot with:
python experiments_analysis/experiment_plot.py --data_dir experiment_results_a30/
```

## Key Benchmark Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--qps` | varies | Requests per second (Poisson arrival) |
| `--num_sampled_requests` | 10000 | Total requests per run |
| `--dataset_type` | sharegpt | Dataset: sharegpt, burstgpt, arxiv_summ |
| `--max_request_len` | 4096 | Max prompt + response length |
| `--backend` | block | block, vllm, or round_robin |
| `--timeout_in_seconds` | 3600 | Max benchmark duration |

## Scheduler Variants for Comparison

| Variant | `--metrics_type` | `num_query_predictor` | Description |
|---------|-----------------|----------------------|-------------|
| Block (full) | min_new_request_latency | 12 | Full simulation, all instances |
| Block Po2 | min_new_request_latency | 2 | Power-of-Two choices |
| Block* (oracle) | min_new_request_latency | 12 | Perfect length prediction |
| Block-NoSim | min_lunmnix_load | 12 | No Vidur simulation |
| Round-robin | round_robin | 1 | Baseline round-robin |
| Random | random | 12 | Baseline random |

## Error Injection (for §6.10 heatmap)

Set on the global scheduler:
```bash
python block/global_scheduler/api_server.py \
    --length_error_pct 25 \   # 25% Gaussian noise on length
    --latency_error_pct 25    # 25% Gaussian noise on latency
```

Grid: `{0, 10, 25, 50, 100}` × `{0, 10, 25, 50, 100}` (16-25 data points)

## Expected Results at a Glance

**A30 Capacity (TTFT P99 < 3s SLO):**
- Block (N=12): 31.1 QPS
- Block Po2 (N=2): 30.2 QPS (97.7% of full)
- Round-robin: 27.2 QPS (at bs=24 suboptimal config)

**A100 Comparison (QPS=36):**
- Block: 18,798 tok/s, 7.5s mean latency (stable)
- Llumnix sched-only: 18,843 tok/s, 7.5s (comparable)
- Llumnix migration: 12,936 tok/s, 89s (collapsed under high load)
