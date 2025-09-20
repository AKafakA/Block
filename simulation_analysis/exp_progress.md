# Experiment Progress Log

## 2025-09-18T21:45Z
- Context restored after parity-focused refactor.
- Sanity suites (50 req × 3 replicas) confirm identical metrics for:
  - Fast-predict off vs on
  - Fast vs fast (with noise)
  - Parallel vs sequential (noise 0 and deterministic noise) 
- Ready to launch large-scale benchmarks (10k req × 12 replicas) for baseline vs optimized configurations.
- TODO (overnight plan):
  1. Record baseline run (`fast_predict=off`, sequential).
  2. Record optimized run (`fast_predict=on`, parallel 12 workers, deterministic noise on).
  3. Compare request metrics for equality (Block & Block*).
  4. Capture runtime stats.
  5. Re-run broader scheduler sweep at QPS=32 (full trace) and analyse.
  6. Run 10× scale (QPS ×10, replicas ×10) for appendix data.
  7. Update documentation (`AGENTS.md`, `simulation_analysis/README.md`, paper appendix draft).

## 2025-09-19T13:33Z
- Verified qps32_full baseline heuristics completed (random, round_robin, infass_pp, llumnix_minus).
- Block_offline run with fast_predict on aborted around start; request_metrics absent.
- No sequential baseline (fast_predict off) stored under simulation_analysis/full_runs yet.
- Ready to relaunch Block baseline and optimized paired runs with enhanced logging.

## 2025-09-19T13:36Z
- Launched full-run baseline (fast_predict off, sequential) storing outputs under simulation_analysis/full_runs/baseline_seq.
- Launched fast-path run (fast_predict on, parallel threads=12, deterministic noise) under simulation_analysis/full_runs/optimized_fast.
- Monitoring via run.log files; both commands running under nohup.
  - baseline_seq PID: 87182
  - optimized_fast PID: 87904

## 2025-09-19T13:45Z
- Added simulation_analysis/compare_block_runs.py to automate metric parity + runtime summary.
- Documented workspace structure and TODOs in simulation_analysis/README.md.

## 2025-09-19T13:50Z
- Added summarize_request_metrics.py helper to aggregate request-level statistics across schedulers.

## 2025-09-19T13:55Z
- Stubbed simulator appendix drafts (Markdown + LaTeX) under simulation_analysis/docs/ for later fill-in.

## 2025-09-19T14:00Z
- Implemented summarize_request_metrics.py to skip missing outputs and generated baseline heuristics summary (simulation_analysis/qps32_full_summary.json).
  - baseline_seq PID: 87182 (active); optimized_fast PID: 87904 (active).

## 2025-09-19T14:05Z
- Drafted run_qps32_schedulers.sh helper to re-launch the scheduler sweep once Block parity is settled.

## 2025-09-19T14:10Z
- Prepared run_large_scale.sh to automate 10x QPS/replica sweeps after parity confirmation.

## 2025-09-19T14:12Z
- Added qps32_analysis_template.md to capture metrics once runs finish.
  - Added narrative context to simulator appendix overview/fidelity sections.
  - Mirrored appendix narrative updates in LaTeX draft.

## 2025-09-19T14:15Z
- Added large_scale_analysis_template.md for future 10× experiment write-up.
  - Updated run_qps32_schedulers.sh and run_large_scale.sh to auto-toggle parallel flags.
  - Added appendix_checklist.md to track remaining deliverables.n  - Pre-filled heuristics latency percentiles in qps32 analysis template.
  - Monitoring long-running Block sweeps; expect outputs once block_offline stage completes per run (~30 min baseline).
  - Block baseline run has passed 50 min (PID 87182) with ~35  - Block baseline run past 50 min (PID 87182) using ~35% RAM; monitoring until metrics appear.
  - Baseline Block run now >1h (PID 87182) with ~50% RAM; continuing to monitor for completion.

## 2025-09-19T15:38Z
- Detected that overnight baseline/optimized parity runs terminated early (only config.json written; no request_metrics.csv).
- run.log files show simulations stopped immediately after initialization, likely due to host shutdown.
- Plan: relaunch baseline (fast_predict=off, sequential) and optimized (fast_predict=on, parallel) runs with nohup logging under simulation_analysis/full_runs.
- Will monitor via fresh run.log files and document runtime checkpoints once progress appears.

## 2025-09-19T15:39Z
- Relaunched baseline parity run: `fast_predict=off`, sequential. PID 11938 writing to simulation_analysis/full_runs/baseline_seq/run.log.
- Relaunched optimized parity run: `fast_predict=on`, parallel workers=12, deterministic noise. PID 12354 writing to simulation_analysis/full_runs/optimized_fast/run.log.
- Monitoring plan: check log tail every 15–20 min for `Processed` counts and eventual `Simulation complete` message; expect request_metrics.csv under each scheduler once finished.

## 2025-09-19T17:09Z
- Confirmed parity reruns (baseline PID 11938, optimized PID 12354) still active after ~70 min with ~99% CPU utilization.
- Logs remain on `Processed 0 requests` because simulator emits progress every 1000 scheduled requests; expect first update once that boundary passes.
- New output buckets present: `baseline_seq/block_offline/2025-09-19_16-39-44-813669` and `optimized_fast/block_offline/2025-09-19_16-39-44-813671` (only `config.json` so far).
- Next actions when resuming: 1) wait for both runs to finish and verify parity via `python simulation_analysis/compare_block_runs.py baseline_seq optimized_fast`; 2) if identical, trigger `run_qps32_schedulers.sh` for full scheduler sweep and `run_large_scale.sh` for 10× experiments; 3) populate docs/appendix drafts with parity + large-scale findings.

## 2025-09-19T20:10Z
- Clarified metrics: emphasized E2E and TTFT; renamed “scheduling delay” to a clearer alias “waiting time” in CSV summaries (kept original column for compatibility).
- Updated heuristic baselines (INFaaS++, Llumnix-) to match paper load definitions; 12×1k slice shows Llumnix- < INFaaS++ on E2E/TTFT as expected.
- Revised analysis README to: (a) skip full fast-predict=off runs; (b) add small-slice (12×240) speedup procedure; (c) focus tables on E2E and TTFT.
- Cleanup plan: prune ad-hoc debug outputs under `experiment_output/offline/` (keep `qps32_full` as stable baseline). New runs will write under `simulation_analysis/{full_runs,large_scale,speed}`.
- Launch plan:
  1) Small-slice Block speedup (12×240) for fast off vs on.
  2) QPS=32 full sweep (12 replicas, all requests), fast on.
  3) Large-scale 10× sweep, fast on.

## 2025-09-19T20:30Z
- Stopped all ongoing runs to enable periodic (5‑minute) progress logging in the simulator.
- Relaunched experiments:
  - Small-slice (12×240): logs → `simulation_analysis/speed/fast_off/run.log`, then `simulation_analysis/speed/fast_on/run.log`.
  - QPS=32 sweep (12 replicas): logs per scheduler → `simulation_analysis/full_runs/qps32_fast/<scheduler>/run.log`.
  - 10× sweep (120 replicas): logs per scheduler → `simulation_analysis/large_scale/<scheduler>/run.log`.
- Monitoring guidance:
  - `tail -f` the respective `run.log` files; expect a progress line at least every 5 minutes.
  - Summaries via `simulation_analysis/summarize_request_metrics.py` with `--csv` once runs finish.

## 2025-09-20T18:08Z
- Completed full 12×120 smoketest matrix covering Block/Block*, noise∈{0,10}%, and execution modes (sequential fast_predict off, sequential fast_predict on, process parallel fast_predict on).
  - Outputs stored under `simulation_analysis/smoketest_{seq,fast_seq,proc,seq_noise,fast_seq_noise,proc_noise}/` with per-scheduler timestamps.
- Request metrics remain bit-for-bit identical across all permutations:
  - `block_offline`: baseline seq ↔ fast seq ↔ process (noise 0 and 10%).
  - `block_star_offline`: fast seq ↔ process (noise 0) and baseline seq ↔ fast seq ↔ process (noise 10% deterministic). Old parity snapshot (2025-09-20_13-29-33) also matches.
- Runtime highlights (120 requests, 12 replicas):
  - `block_offline` baseline seq (fast off) ≈ 1 480 s (noise 0) / 1 453 s (noise 10%).
  - Sequential fast_predict alone offers no benefit (≈0.94–1.00× vs baseline) on this slice; process mode cuts wall-clock to ≈534 s (noise 0, 2.77× speedup) and ≈475 s (noise 10%, 3.06× speedup).
  - `block_star_offline` with noise 10% sees sequential fast_predict slightly faster than baseline (1.08×) and process mode at 31 s (1.33× vs baseline). In noise-free runs the process backend incurs overhead (≈53 s vs 19 s fast seq) because snapshots are cheap without noise; keep sequential fast_predict for that case.
- Updated `SimulatePredictReplicaScheduler` to log execution-time predictor misses as warnings instead of raw `print` errors (parity runs launched earlier still show the old text; new runs emit `WARNING`).
- Next: propagate these findings into the main README/appendix, and decide runtime defaults (likely enable process mode for Block when noise>0, keep Block* sequential for noise=0).

## 2025-09-20T18:20Z
- Added automation harness:
  - `run_experiment_suite.py` orchestrates scheduler sweeps, emits throughput-aware summaries, and reports Block/Block* improvements vs heuristics.
  - Shell wrappers (`run_qps32_suite.sh`, `run_large_scale_suite.sh`) plus remote utilities (`scripts/remote_run_suite.sh`, `scripts/remote_collect_results.sh`) streamline launching on dedicated hosts (`NORMAL_SCALE_SIMULATION_HOST`, `LARGE_SCALE_SIMULATION_HOST`).
- Ready to dispatch full-trace 12×10k and 120×10k runs remotely; merged outputs can be retrieved into `simulation_analysis/` for appendix integration.
