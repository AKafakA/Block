# Block — Artifact Evaluation for the SYSTOR '26 Paper

This branch (`systor-ae`) is the artifact for:

> **Block: Balancing Load in LLM Serving with Randomized Predictive Scheduling.**
> Wei Da and Evangelia Kalyvianaki. *SYSTOR '26.* (extended arXiv version: [2508.03611](https://arxiv.org/abs/2508.03611))

It contains the exact scheduler/predictor source, experiment drivers, and plotters used for the paper, plus a numbered, idempotent script suite (`block/exp/end_to_end_exp_scripts/artifact_evaluation/`, scripts `00`–`16`) that reproduces **every experimental figure and table** end to end. Detailed per-script docs live in [`artifact_evaluation/README.md`](block/exp/end_to_end_exp_scripts/artifact_evaluation/README.md); a fully manual walkthrough is in [`docs/REPRODUCE_FROM_SCRATCH.md`](docs/REPRODUCE_FROM_SCRATCH.md).

## 1. Hardware and software prerequisites

| Resource | Used for | Spec |
|---|---|---|
| A30 cluster | Figs 5–7, 9–13; Tables 1–3 | 12× CloudLab **d7525** (1× NVIDIA A30 24 GB, 2× AMD 7302, 128 GB RAM each) |
| A100 cluster | Fig 5 (70B row), Fig 8 | 2× CloudLab **d8545** (4× A100-40GB SXM4 each, 100 Gbps NIC) |
| Any single GPU (≥24 GB) | RoBERTa length-model training (script 03) | e.g., one A30/L40 node |

- Hugging Face token with access to `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-70b-hf`, `Qwen/Qwen2-7B` (export `HF_TOKEN` before running).
- Software stack is installed by `block/exp/setup.sh` (vLLM fork **first**, from https://github.com/AKafakA/vllm/tree/block, then PyTorch, then `transformers==4.50.3`); always `export VLLM_USE_V1=0`.
- Python plotting deps: `numpy`, `matplotlib` (any recent versions).

Total compute for the full suite: ~60 A30 GPU-hours + ~12 A100 GPU-hours (~55 h wall-clock; per-script times below).

## 2. Quick start (full reproduction)

```bash
export HF_TOKEN=<your token>
cd Block   # repo root, branch systor-ae
AE=block/exp/end_to_end_exp_scripts/artifact_evaluation

# One-time prerequisites
sh $AE/00_setup_cluster.sh            # SSH check + code sync to 12 nodes (~1h)
sh $AE/01_profile_vidur_a30.sh        # Vidur HW profile, A30 (~3-4h)
sh $AE/02_profile_vidur_a100.sh       # Vidur HW profile, A100 (~5-6h)
parallel-scp -h block/config/hosts -r data/profiling Block/data/
sh $AE/03_train_length_model.sh llama # RoBERTa length model (~3h)
sh $AE/03_train_length_model.sh qwen  # (~3h)
parallel-scp -h block/config/hosts -r model/ Block/
sh $AE/04_warmup_llama.sh             # predictor cache warmup (~5m)

# Experiments (each script = one paper artifact group; see map below)
sh $AE/05_prediction_error_a30.sh     # ~1h
sh $AE/06_prediction_error_a100.sh    # ~1.5h
sh $AE/07_main_sweep_a30.sh           # ~26h
sh $AE/08_n_ablation.sh               # ~1h
sh $AE/09_burstiness.sh               # ~30m
sh $AE/10_error_heatmap.sh            # ~2.5h
sh $AE/11_capacity_refine.sh          # ~3h
sh $AE/12_generality.sh               # ~5h
sh $AE/13_cpu_overhead.sh             # ~45m
sh $AE/14_a100_llumnix.sh             # ~3h
sh $AE/15_a100_block.sh               # ~9h

# Render every figure + table input (no cluster needed)
sh $AE/16_generate_figures.sh         # <5 min → figures_output/
python3 experiments_analysis/paper_figures/aggregate_data.py   # → figures_output/tables/
```

## 3. Figure/table → script map (SYSTOR '26 camera-ready numbering)

Figures 1–4 are illustrations (no experiment). Everything else:

| Paper artifact | Section | Produced by | Rendered by | Output file |
|---|---|---|---|---|
| **Fig 5** — prediction error / selected-instance rank (3 rows × 5 QPS) | §6.2 | scripts 05 (A30) + 06 (A100 70B) | `experiments_analysis/prediction_plot.py` | `experiment_output/results/…profiling.png` |
| **Fig 6** — request metrics, 8-panel QPS sweep + capacity bars | §6.3 | script 07 | script 16 (`regen_figures.py`) | `figures_output/exp_plots/cluster_metrics/qps.png` |
| **Fig 7** — free GPU blocks / variance / preemptions | §6.3 | script 07 | script 16 | `figures_output/exp_plots/cluster_metrics/linear.png` |
| **Fig 8** — Block (CP / no-CP) vs full Llumnix on A100 | §6.4 | scripts 14 + 15 | script 16 (`plot_llumnix_aggregate.py --default po2`) | `figures_output/llumnix_comparison_v2_po2.png` |
| **Fig 9** — power-of-k ablation (Po2/Po4/Po8/Fanout) | §6.5 | scripts 08 + 11 | script 16 | `figures_output/po2_comparison.png` |
| **Fig 10** — oracle-length upper bounds | §6.6 | scripts 07 + 11 | script 16 | `figures_output/oracle_comparison.png` |
| **Fig 11** — burstiness (γ-arrivals, α=0.25–2.0) | §6.7 | script 09 | script 16 | `figures_output/burstiness_lines.png` |
| **Fig 12** — prediction-error sensitivity heatmap (4×4) | §6.8 | script 10 | script 16 | `figures_output/prediction_error_heatmap.png` |
| **Fig 13** — per-predictor CPU + RSS across QPS | §6.9 | script 13 | script 16 | `figures_output/cpu_overhead.png` |
| **Table 1** — response-length prediction accuracy (MAPE) | §6.2 | script 03 (`eval_roberta`) | printed by eval run | eval stdout / `model/…/eval` |
| **Table 2** — per-scheduler TTFT/E2E snapshot @ QPS 20–36 | §6.3 | script 07 | `aggregate_data.py` | `figures_output/tables/aggregated.csv` (+ `summary.md`) |
| **Table 3** — capacity across workload/config changes | §6.10 | scripts 12 + 11 | `aggregate_data.py` | `figures_output/tables/summary.md` |

## 4. Expected headline results (for spot-checking)

At SLO = TTFT P99 ≤ 10 s, Llama-2-7B/ShareGPT, batch 48 / chunk 512 on the 12×A30 cluster:

| Scheduler | Capacity (QPS) |
|---|---|
| Block (Po2, estimated lengths) | **31.6** |
| Block Po2, oracle lengths | 32.4 |
| Po4-est / Po8-est | 31.9 / 31.7 |
| Fanout-est / Fanout-oracle | 31.7 / 32.6 |
| Llumnix- (scheduler-only) | 31.5 |

Qwen2-7B: Block 73.9 vs Llumnix- 69.7 QPS. CPU: Po2 ≈20% vs Fanout ≈56% mean per-predictor at QPS=20 (~2.8×). A100 @QPS=36: Block ≥2.57× Llumnix throughput. Single-run tolerance: capacities ±0.3 QPS; heatmap cell deltas <2% are within deploy noise (±4%).

## 5. Practical notes

- Scripts are **idempotent and self-verifying**; each syncs its NPZs into `experiment_results_a30/` or `experiment_results_a100/` and runs `util_verify_npz.py`. Layout expected by the plotters is documented in `artifact_evaluation/README.md` §Output layout.
- **Both schedulers by default**: scripts 09/10/13 run Po2 *and* Fanout (set `RUN_BOTH=false` for Po2-only).
- **Capacity search**: `early_stop` (default, matches paper) vs `MODE=full_sweep` for dense 0.1-QPS curves.
- Predictor count is strictly 16/node — scripts abort rather than continue with fewer (a partial predictor set corrupts scheduling results).
- The A100 script 15 syncs after *every* config; do not disable this (protects against the fixed-output-path overwrite in `run_benchmark.sh`).
