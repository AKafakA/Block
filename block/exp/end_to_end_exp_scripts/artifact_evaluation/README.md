# Artifact Evaluation — Block (SYSTOR '26)

End-to-end reproduction guide for all paper figures from a clean cluster reservation.
For the figure/table → script map keyed to the SYSTOR '26 camera-ready, see [`AE.md`](../../../../AE.md) at the repo root.

## Prerequisites

- **A30 cluster**: 12× CloudLab d7525 nodes (Llama-2-7B, A30 GPUs, 64 cores/node)
- **A100 cluster**: 2× CloudLab d8545 nodes (Llama-2-70B TP=4, 4× A100-40GB SXM4 each)
- **HF token** with access to `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-70b-hf`, `Qwen/Qwen2-7B`

## Script order

Scripts numbered in strict dependency order. Each one is idempotent and self-verifying.

| # | Script | Paper section | Time | What |
|---|---|---|---|---|
| 00 | `00_setup_cluster.sh` | prereq | ~1h | Verify SSH + apply code patches + sync to 12 nodes |
| 01 | `01_profile_vidur_a30.sh` | prereq (Sec 6.2 input) | ~3-4h | Vidur HW profile: MLP / attention / collectives / CPU-overhead CSVs for Llama-2-7B on A30 |
| 02 | `02_profile_vidur_a100.sh` | prereq (Sec 6.2 input) | ~5-6h | Vidur HW profile for Llama-2-70B on A100-40GB (TP=1/2/4) |
| 03 | `03_train_length_model.sh [llama\|qwen]` | prereq (Sec 6.2 input) | ~3h × 2 | RoBERTa length predictor on 40k sharegpt prompts |
| 04 | `04_warmup_llama.sh` | prereq | ~5m | Populate `~/Block/cache/*.pkl` (RandomForest predictor cache) |
| 05 | `05_prediction_error_a30.sh` | **Sec 6.2 Latency Prediction Metrics (A30)** | ~1h | `profiling_sampling_rate=0.01`, Fanout N=12, oracle lens × 5 QPS |
| 06 | `06_prediction_error_a100.sh` | **Sec 6.2 Latency Prediction Metrics (A100)** | ~1.5h | Same on Llama-2-70B |
| 07 | `07_main_sweep_a30.sh` | **Sec 6.3 Main TTFT / throughput** | ~26h | Phase 1.1 (6 schedulers × 17 QPS) + Phase 1.2 (Po2 oracle+est × 17 QPS) |
| 08 | `08_n_ablation.sh` | **Sec 6.4 + 6.6 N-tunable** | ~1h | Phase 2 (N=4,6,8 @ QPS=30) + Phase 7b (Po4 + Po8 capacity float) |
| 09 | `09_burstiness.sh` | **Sec 6.5 Burstiness** | ~30m | Po2-est + Llumnix-N12 × burst {0.25, 0.5, 1.0, 2.0} @ QPS=32 |
| 10 | `10_error_heatmap.sh` | **Sec 6.6 Error sensitivity** | ~2.5h | Po2-est × 15 (length_err, latency_err) cells, fresh-deploy-per-cell |
| 11 | `11_capacity_refine.sh` | **Sec 6.x Capacity table** | ~3h | Float capacity refine (Po2-est/oracle, Fanout-est/oracle, Llumnix) |
| 12 | `12_generality.sh` | **Sec 6.7 Generality** | ~5h | chunk2048 + batch24 + BurstGPT + Qwen2-7B |
| 13 | `13_cpu_overhead.sh` | **Sec 6.7 CPU overhead** | ~45m | Po2-est × 5 QPS with `--enable_cpu_tracking` (requires patches) |
| 14 | `14_a100_llumnix.sh` | **Sec 6.8 A100 baseline** | ~3h | Llumnix sweep, 21 QPS points |
| 15 | `15_a100_block.sh` | **Sec 6.8 A100 Block** | ~9h | Po2/Fanout × CP-on/off × 6 QPS (per-config sync — critical) |
| 16 | `16_generate_figures.sh` | **Render all paper figures** | <5 min | Runs `regen_figures.py all` + `plot_llumnix_aggregate.py --default po2`; all PNGs land in `figures_output/`. No cluster required. |

Plus helpers:
- `util_predictor_health.sh` — cluster-wide 192-predictor health check (run during long sweeps)
- `util_verify_npz.py` — sanity-check NPZs post-phase (sample count, required fields)

## Running the full suite

```bash
# Prereqs (one-time per cluster/device class)
sh artifact_evaluation/00_setup_cluster.sh
sh artifact_evaluation/01_profile_vidur_a30.sh          # one A30 node, 3-4h
sh artifact_evaluation/02_profile_vidur_a100.sh         # one A100 node, 5-6h
# Sync profiling data to all cluster nodes:
parallel-scp -h block/config/hosts -r data/profiling Block/data/
sh artifact_evaluation/03_train_length_model.sh llama   # dedicated GPU VM, 3h
sh artifact_evaluation/03_train_length_model.sh qwen    # 3h
# Copy checkpoint to cluster nodes:
parallel-scp -h block/config/hosts -r model/ Block/
sh artifact_evaluation/04_warmup_llama.sh               # 5 min

# Section 6.2 — prediction metrics
sh artifact_evaluation/05_prediction_error_a30.sh       # 1h
sh artifact_evaluation/06_prediction_error_a100.sh      # 1.5h

# Sections 6.3+
sh artifact_evaluation/07_main_sweep_a30.sh             # 26h
sh artifact_evaluation/08_n_ablation.sh                 # 1h
sh artifact_evaluation/09_burstiness.sh                 # 30 min
sh artifact_evaluation/10_error_heatmap.sh              # 2.5h
sh artifact_evaluation/11_capacity_refine.sh            # 3h
sh artifact_evaluation/12_generality.sh                 # 5h
sh artifact_evaluation/13_cpu_overhead.sh               # 45 min
sh artifact_evaluation/14_a100_llumnix.sh               # 3h
sh artifact_evaluation/15_a100_block.sh                 # 9h

# Final step — render all paper figures from collected data, no cluster needed
sh artifact_evaluation/16_generate_figures.sh           # <5 min — figures in figures_output/
```

Serving experiments (05 onwards) require predictors deployed and listening on all 12 A30 nodes. `04_warmup_llama.sh` starts 1 predictor per node and populates the RF cache; subsequent scripts deploy the full 16 predictors per node which load from cache (~30s each).

## Dual-paper support: Po2 and Fanout both run by default

The artifact reproduces **both** the SYSTOR '26 paper (randomized Po2 as the default scheduler) and the original arXiv version (Fanout-N=12 emphasis). Scripts that would otherwise be Po2-only run both variants by default:

| Script | Default passes | Env override |
|---|---|---|
| 07 `main_sweep_a30` | Po2 (N=2) oracle+est **AND** Fanout (N=12) oracle+est + 4 baselines | fixed |
| 09 `burstiness` | Po2-est + Fanout-est + Llumnix-N12 × 4 burst levels | `RUN_BOTH=false` → Po2 only |
| 10 `error_heatmap` | Po2-est + Fanout-est × 15 cells | `RUN_BOTH=false` → Po2 only |
| 11 `capacity_refine` | Po2-est/oracle + Fanout-est/oracle + Llumnix | fixed |
| 12 `generality` | All three schedulers per variant | fixed |
| 13 `cpu_overhead` | Po2-est + Fanout-est × 5 QPS | `RUN_BOTH=false` → Po2 only |
| 14+15 A100 | Llumnix + Po2+CP + Fanout+CP + Po2-noCP + Fanout-noCP | fixed |

This makes the output self-contained: Po2 data validates the SYSTOR '26 paper, Fanout data the arXiv version — no external priors needed.

## Capacity-search modes

When a script refines capacity at 0.1 QPS resolution, there are two modes (relevant to scripts 08, 11, 12):

- **early_stop** (default, time-efficient): after integer bracket `[lo, hi]`, do binary probes until TTFT P99 lands in the **9.X band (9000-10000 ms)** or **10.X band (10000-11000 ms)**, then stop. ~3-5 probes per scheduler.
- **full_sweep** (rigorous, reviewer-requested): after integer bracket, probe **all ten 0.1-step points** (`lo.0, lo.1, ..., hi.0`) to draw the dense TTFT-vs-QPS curve. ~10 probes per scheduler.

The paper uses **early_stop** values (matches what was run in the campaign). Set `MODE=full_sweep` at the script level if a reviewer wants the dense curve.

## Correctness / completeness review

### Sec 6.2 Latency Prediction Metrics
- Script 05 (A30, Llama-7B) and 06 (A100, Llama-70B): `PROFILING_SAMPLE_RATE=0.01`, oracle lens.
  - Produces NPZ fields `sampled_predict_accuracies`, `sampled_mean_error_ratios`.
  - Paper's "Latency Prediction Metrics" figure reads these across QPS sweep.
- Prereqs 01/02/03: all the inputs needed — Vidur HW profile + RoBERTa checkpoint.

### Sec 6.3 Main sweep
- Script 07: 6 schedulers (min_new_request_latency est+oracle, min_lunmnix_load, min_infass_load, random, round_robin, request_per_seconds) × QPS 20-36.
- Plus Po2 (N=2) oracle+est × QPS 20-36.
- All on Llama-2-7B, batch=48, chunk=512, sharegpt.
- Output: 119 NPZs for Phase 1.1, 34 for Phase 1.2.

### Sec 6.4 N-ablation + Sec 6.6 N-tunable
- Script 08 combines Phase 2 (N=4/6/8 @ QPS=30) and Phase 7b (Po4 + Po8 capacity refinement).
- Po4-est capacity = best N (31.9), Po8 matches Fanout (31.7) — diminishing returns above N=4.

### Sec 6.5 Burstiness
- Script 09: `burstiness_exp.sh` with fixed per-scheduler N (Po2=2, Llumnix=12).
- 4 burst levels × 2 schedulers = 6-8 cells at QPS=32.
- **Llumnix N=12 fix verified at script start** — the script aborts if the branch is missing (was a campaign bug).

### Sec 6.6 Error sensitivity
- Script 10: Po2-est × 15 cells of (length_err_pct, latency_err_pct) at QPS=32.
- Skips baseline (0,0) — covered by Phase 1.1.
- **Fresh-deploy-per-cell** to avoid ±4% single-deploy noise.
- Note: single-deploy variance is ±4%, so cell deltas <2% are not statistically distinguishable.

### Sec 6.7 Generality + CPU
- Script 12 covers: chunk2048 (different chunked-prefill), batch24 (different batch cap), BurstGPT dataset, Qwen2-7B model.
- Qwen sub-phase includes warmup for Qwen cache before 16-predictor deploy (avoids OOM).
- Qwen sharegpt yields **9963/10000** valid samples (tokenizer filter) — uniform across schedulers, comparison fair.
- Script 13 collects CPU/memory profiles for both Po2-est and Fanout-est (QPS 20/24/28/32/36) under `phase7_cpu_tracker/{po2,fanout}/`.

### Sec 6.8 A100
- Scripts 14 + 15: Llama-70B TP=4.
- Script 15 does **per-config immediate sync** — this prevents the A100 `run_benchmark.sh` fixed-path bug from silently overwriting earlier configs' NPZs (caught during the campaign — ~1 day of raw data was lost before this lesson).

## Critical patches (already applied in-tree)

Bugs discovered during the campaign, patched in this repo:

1. **Sequential predictor deploy** in `block/exp/experiment.sh`
   - Parallel deploy had ~8% per-node failure rate; sequential is deterministic.
2. **CPU tracking pipeline** across 3 files:
   - `block/predictor/api_server.py` — captures `cpu_percent`, `memory_rss_mb`, `cpu_cores`
   - `block/global_scheduler/api_server.py` — aggregates into `single_metric`
   - `block/benchmark/benchmark_serving.py` — collects per-request and saves to NPZ
3. **`verify_predictors.sh` SSH timeout** — explicit `< /dev/null` + `timeout 30 ssh` + 60s bounded `wait`. Without this, `nohup &` over SSH hangs indefinitely.
4. **Llumnix N=12 branching** in `burstiness_exp.sh` — per-scheduler N mapping; Po2 stays at 2, Llumnix/INFaaS switch to 12 for broadcast.
5. **Heatmap fresh-deploy-per-cell** in `error_heatmap_exp.sh` — required for absolute comparisons (single-deploy variance swamps sub-5% effects).


## Output layout

All A30 NPZs land under `experiment_results_a30/`, A100 under `experiment_results_a100/`. Each phase script syncs immediately after completion and runs `util_verify_npz.py` on its dir.

```
experiment_results_a30/
  phase_prediction_a30/        # script 05 — Sec 6.2
  phase11_main/                # script 07 — Sec 6.3 (Phase 1.1)
  phase12_po2/                 # script 07 — Sec 6.3 (Phase 1.2)
  phase2_n_ablation/           # script 08 — Sec 6.4
  phase7_po4po8/               # script 08 — Sec 6.6 N-tunable
  phase3_1_burstiness/         # script 09 — Sec 6.5
  phase3_2_error_heatmap/      # script 10 — Sec 6.6
  phase4_1_float/              # script 11 — Sec 6.x
  phase4_2_generality/         # script 12 — Sec 6.7
  phase4_2_burstgpt/           # script 12 — Sec 6.7
  phase4_2_qwen/               # script 12 — Sec 6.7
  phase7_cpu_tracker/          # script 13 — Sec 6.7
experiment_results_a100/
  phase_prediction_a100/       # script 06 — Sec 6.2
  llumnix_sweep/               # script 14 — Sec 6.8
  phase57_block/{po2_cp,fanout_cp,po2_nocp,fanout_nocp}/  # script 15 — Sec 6.8
```

## See also
- `AE.md` (repo root) — figure/table → script reproduction map for the SYSTOR '26 camera-ready
- `docs/REPRODUCE_FROM_SCRATCH.md` — manual reproduction walkthrough with patch details
