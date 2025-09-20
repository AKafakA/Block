# Simulation Analysis Workspace

This directory hosts long-running offline simulation sweeps and the analysis
artifacts that back the paper appendix draft.

## Layout
- `full_runs/` – canonical outputs for QPS=32, 12 replicas.
  - `baseline_seq/` – Block and Block* with `fast_predict=off`, sequential what-ifs.
  - `optimized_fast/` – Block and Block* with `fast_predict=on`, parallel what-ifs (deterministic noise).
- `exp_progress.md` – chronological log for hand-off between sessions.
- `compare_block_runs.py` – helper to check parity and runtime deltas for Block runs.
- `summarize_request_metrics.py` – aggregate latency/scheduling stats for the latest run of each scheduler. Highlights E2E latency and TTFT (from `prefill_e2e_time`) first, then waiting time (arrival→first-schedule), with backward compatibility for `request_scheduling_delay`.
- `run_qps32_schedulers.sh` – sequential driver to reproduce the QPS=32 sweep once Block parity is confirmed (auto-toggles parallelism based on the fast-predict flag).
- `run_large_scale.sh` – sequential driver for the 10× QPS/replica experiments (auto-toggles parallelism based on the fast-predict flag).

## Experiment Policy (Updated)
- Headline metrics: E2E latency and TTFT.
- Waiting time (arrival→first-schedule) is exposed as `request_waiting_time` (alias of `request_scheduling_delay`) to avoid confusion with “scheduling overhead” used in the paper.
- Fast-predict: For Block/Block*, skip full-trace sequential (fast-predict=off) runs due to runtime. Report fast/slow speedup using a small slice (12 replicas, 240 requests). For full-trace runs, use fast-predict on with parallel evaluation enabled.

### Latest Parity & Speed Findings (12×120 smoketests)
- Metrics parity holds bit-for-bit across all configurations (noise ∈ {0,10}%, fast_predict on/off, sequential vs process) for both `block_offline` and `block_star_offline`.
- Runtime snapshots (per `run.log`):
  - `block_offline` baseline seq (fast off) ≈ 1 480 s (noise 0) / 1 453 s (noise 10%).
  - Sequential fast_predict alone offers no gain on this slice (~0.94–1.00× vs baseline); process parallelism cuts wall-clock to ≈534 s (noise 0) and ≈475 s (noise 10%), i.e. 2.7–3.1× faster with identical outcomes.
  - `block_star_offline` (noise 10%, deterministic) benefits modestly from fast_predict (1.08×) and further from process evaluation (1.33×). In the noise-free case, sequential fast_predict remains faster than the process backend (≈19 s vs 53 s).
- Execution-time predictor fallbacks now emit `WARNING` logs instead of raw `print` lines; expect messaging like “Execution-time predictor miss … using cached execution time …”.

### Automation Helpers
- `run_experiment_suite.py`: orchestrates a full scheduler sweep, prints progress every 30 min (configurable), and emits `analysis_summary.json`/`.csv` with TTFT/E2E/waiting percentiles **and throughput**. Comparative improvements for Block/Block* versus heuristics are included in the JSON output under `comparisons`.
- `run_qps32_suite.sh`, `run_large_scale_suite.sh`: wrappers for the 12-replica QPS=32 and 120-replica QPS=320 suites respectively. Append additional `run_experiment_suite.py` flags via environment variable `REMOTE_EXTRA_ARGS` or direct CLI arguments when launching locally.
- Remote automation scripts under `scripts/`:
  - `remote_run_suite.sh`: split into setup and run steps, runs detached, supports log collection. Installs Python deps under `${REMOTE_ROOT}/.pyuser` and wires PYTHONPATH automatically (no sudo/venv needed).
    - Setup (repo/deps/branch): `bash scripts/remote_run_suite.sh setup qps32` or `setup large`
    - One-click setup+run: `bash scripts/remote_run_suite.sh one_click_qps32` or `one_click_large` (or `one_click_both`)
    - Run (detached; prints remote log path): `bash scripts/remote_run_suite.sh run_qps32` or `run_large`
    - Tail latest log: `bash scripts/remote_run_suite.sh tail_qps32` or `tail_large`
    - Collect logs + outputs to local (defaults under `simulation_analysis/remote_collected/<mode>`):
      - `bash scripts/remote_run_suite.sh collect_qps32`
      - `bash scripts/remote_run_suite.sh collect_large`
      - `bash scripts/remote_run_suite.sh collect_all`
    - Defaults:
      - Hosts: `NORMAL_SCALE_SIMULATION_HOST=wd312@caelum-104` (QPS=32), `LARGE_SCALE_SIMULATION_HOST=wd312@caelum-105` (large). Override by exporting those env vars.
      - Branch: `REMOTE_BRANCH=simulator`. Override by exporting `REMOTE_BRANCH`.
  - `remote_collect_results.sh`: fetches remote `simulation_analysis/...` outputs back into the local tree (requires `parallel-rsync`; install via `apt-get install pssh` or similar).

## How to Reproduce
1) Small-slice Block speedup (12×240)
   - Fast off:
     - `PYTHONPATH=. python scripts/run_offline_simulations.py --schedulers block_offline block_star_offline --num-replicas 12 --trace-file data/trace_data/sharegpt/sharegpt_val_10k_llama2.csv --predicted-trace-file data/trace_data/sharegpt/generate/llama/sharegpt-llama-7b-val-10k-predicted.json --qps 32 --max-requests 240 --block-noise 10 --fast-predict off --output-dir simulation_analysis/speed/fast_off`
   - Fast on + parallel (deterministic noise):
     - `PYTHONPATH=. python scripts/run_offline_simulations.py --schedulers block_offline block_star_offline --num-replicas 12 --trace-file data/trace_data/sharegpt/sharegpt_val_10k_llama2.csv --predicted-trace-file data/trace_data/sharegpt/generate/llama/sharegpt-llama-7b-val-10k-predicted.json --qps 32 --max-requests 240 --block-noise 10 --fast-predict on --block-parallel-enable on --deterministic-noise on --output-dir simulation_analysis/speed/fast_on`
   - Summarize: `PYTHONPATH=. simulation_analysis/summarize_request_metrics.py simulation_analysis/speed --schedulers block_offline block_star_offline --csv simulation_analysis/speed/summary.csv`

2) Full QPS=32 (12 replicas, all requests)
   - Fast on for Block variants:
     - `./simulation_analysis/run_qps32_schedulers.sh simulation_analysis/full_runs/qps32_fast on`
   - Summarize:
     - `PYTHONPATH=. simulation_analysis/summarize_request_metrics.py simulation_analysis/full_runs/qps32_fast --csv simulation_analysis/qps32_full_summary.csv`

3) Large-scale (10×)
   - `./simulation_analysis/run_large_scale.sh simulation_analysis/large_scale on`
   - Summarize:
     - `PYTHONPATH=. simulation_analysis/summarize_request_metrics.py simulation_analysis/large_scale --csv simulation_analysis/large_scale_summary.csv`

## Next Steps Checklist
- [ ] Run small-slice speedup (12×240) and record runtime + speedup.
- [ ] Launch QPS=32 full-trace sweep (fast-predict on for Block/Block*), collect E2E/TTFT summaries.
- [ ] Launch 10× sweep and capture supplementary results.
- [ ] Keep exp_progress.md updated for smooth handoff.
