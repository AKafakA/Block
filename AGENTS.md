# Repository Guidelines

## Project Structure & Module Organization
- `block/` — Predictor, Global Scheduler, length estimator, benchmarks, configs, and experiment scripts.
  - `block/predictor/`, `block/global_scheduler/`, `block/length_estimation/`, `block/benchmark/`, `block/exp/`, `block/config/`.
- `vidur/` — Simulator and schedulers (fork with extensions).
- `data/` — Datasets and profiling artifacts (large files; avoid committing new large assets).
- `experiments_analysis/` — Plotting and analysis utilities.
- `images/` — Diagrams used in docs.

## Setup, Build, and Run
- Create env and install deps:
  - `python -m venv venv && source venv/bin/activate`
  - `pip install -r requirements.txt` (optionally `-r requirements-dev.txt`)
- Quick start (cluster + experiments): `sh block/exp/setup.sh` then scripts in `block/exp/end_to_end_exp_scripts/` (e.g., `main_experiment.sh`).
- Run simulator CLI: `python vidur/main.py --help`.
- Start Predictor API: `python -m block.predictor.api_server --port 8100 --config_path block/config/llama_config.json`.
- Start Global Scheduler API: `python -m block.global_scheduler.api_server --help`.
- Benchmark replay: `python block/benchmark/benchmark_serving.py`.

## Coding Style & Naming
- Python 3.10+, 4‑space indents, type hints where practical.
- Names: modules/vars `lower_snake_case`, classes `UpperCamelCase`, constants `UPPER_SNAKE_CASE`.
- Formatting/linting: run `black .`, `isort .`, `flake8` (optionally `pylint`). Keep functions short and cohesive.

## Testing Guidelines
- No formal unit test suite yet; validate changes by:
  - Running `vidur/main.py` with representative configs.
  - Exercising `block/exp/*` scripts for end‑to‑end checks.
- If adding tests, place under `tests/`, name files `test_*.py`, and keep them fast/deterministic.

## Commit & Pull Request Guidelines
- Commits: imperative mood, scoped when helpful (e.g., `block/predictor:`). Example: `vidur: add chunked prefill flag handling`.
- PRs must include:
  - Summary of changes and rationale.
  - How you validated (commands/logs; figures if applicable).
  - Linked issues and any config/data assumptions.
  - Screenshots/plots for user‑visible changes (place under `experiment_output/results/` or attach).

## Security & Configuration Tips
- Do not commit secrets (e.g., Hugging Face tokens). Populate them locally in scripts under `block/exp/` as instructed.
- Configuration lives under `block/config/` (e.g., `llama_config.json`, `hosts`). Keep JSON valid and document keys inline.
- Large data belongs in `data/` and should be referenced, not duplicated.

## Architecture Notes
- Predictor queries Vidur (or uses live stats) to estimate per‑instance cost; Global Scheduler selects a target based on configured metric. Prefer small, testable modules; avoid cross‑layer imports.

---

## Vidur Simulation + Block Context

- Core flow
  - Entry: `vidur/main.py:1` builds `SimulationConfig`, instantiates `Simulator`, runs event loop.
  - Events: arrivals → `GlobalScheduleEvent` → per‑replica schedule; see `vidur/simulator.py:1`.
  - Replica scheduler: Sarathi (chunked prefill + decode); `vidur/scheduler/replica_scheduler/sarathi_replica_scheduler.py:1`.
  - Execution‑time predictor: profile‑based; `vidur/execution_time_predictor/base_execution_time_predictor.py:1`.

- Global schedulers (baseline vs Block)
  - Baselines (non‑predictive heuristics):
    - `random` (uniform), `round_robin`, `lor` (least outstanding), `lodt` (requests/free blocks), `min_memory`, `infass_pp`, `llumnix_minus`.
    - Files: `vidur/scheduler/global_scheduler/*.py` (see names above).
  - Block offline schedulers:
    - `BLOCK_OFFLINE` and `BLOCK_STAR_OFFLINE` share the same class `BlockOfflineGlobalScheduler`; `vidur/scheduler/global_scheduler/block_offline_global_scheduler.py:19`.
    - For each new request, query a Request Timeline Predictor on each replica and pick the replica that optimizes a target metric (default: `min_latency`).
    - Target metric mapping lives in `vidur/request_timeline_predictor/base_request_timeline_predictor.py:29` and `vidur/types/optimal_global_scheduler_target_metric.py:12`.

- Request Timeline Predictor (what‑if simulation)
  - Implementation: `SimulateRequestTimelinePredictor` uses `SimulatePredictReplicaScheduler` to clone the replica scheduler and run a what‑if until the target request completes; `vidur/request_timeline_predictor/simulate_request_timeline_predictor.py:1`.
  - Noise wrapper: `NoisySimulateRequestTimelinePredictor` adds multiplicative noise for realism; `vidur/request_timeline_predictor/noisy_simulate_request_timeline_predictor.py:12`.

### Block vs Block*
- Length signal used for prediction:
  - Block (`block_offline`): uses ground‑truth decode length from the trace.
  - Block* (`block_star_offline`): uses predicted decode length (from estimator‑tagged trace).
- Wiring:
  - Offline runner sets `use_predicted_decode_tokens` only for Block*; see `scripts/run_offline_simulations.py:205` and `:218`.
  - Requests carry both truth and predicted fields; the predictor’s what‑if always sets the target request’s decode tokens to `num_predicted_decode_tokens`; see `vidur/scheduler/replica_scheduler/simulate_predict_replica_scheduler.py:27` and `:34`.
  - In Block runs, `num_predicted_decode_tokens == num_decode_tokens` (ground truth). In Block*, it comes from the predicted trace file.

### Fast‑Predict Toggle (parity‑preserving)
- Purpose: speed up what‑if simulations by avoiding deep copies and reusing state safely.
- Config:
  - CLI: `--fast-predict on|off` in `scripts/run_offline_simulations.py:145` and used at `:195`.
  - Passed via `BlockOfflineGlobalSchedulerConfig.fast_predict` / `BlockStarOfflineGlobalSchedulerConfig.fast_predict`; `vidur/config/config.py:589` and `:609`.
- Behavior:
  - When enabled, the Block offline scheduler calls `disable_copy_of_base_replica_scheduler()` on the predictor; `vidur/scheduler/global_scheduler/block_offline_global_scheduler.py:50`.
  - Predictor uses an optimized shallow clone `_optimized_clone` plus targeted snapshots instead of full `deepcopy`; `vidur/scheduler/replica_scheduler/simulate_predict_replica_scheduler.py:107`.
  - Also uses batch execution time caching for large batches; threshold and caches set in `vidur/request_timeline_predictor/simulate_request_timeline_predictor.py:8` and used in `simulate_predict_replica_scheduler.py:186`.
  - Intent: identical scheduling decisions and metrics vs slow path; only runtime differs.

### Offline Simulation: How‑To
- Driver: `scripts/run_offline_simulations.py:1`.
- Typical flags:
  - `--schedulers`: e.g., `block_offline infass_pp llumnix_minus random round_robin`.
  - `--num-replicas`: logical GPU replicas.
  - `--trace-file`: ground‑truth CSV (arrivals, prefill, decode lengths).
  - `--predicted-trace-file`: JSON/CSV with predicted lengths (used when `block_star_offline`).
  - `--qps`: target QPS (preferred) or `--time-scale-factor` (advanced).
  - `--max-requests`: limit to speed up sweeps.
  - `--block-noise`: percent noise for Block predictors (converted to fraction internally).
  - `--block-target-metric`: e.g., `min_latency` (default);
  - `--fast-predict on|off`: off by default for strict parity.
- Example (first 200 requests, QPS 320): see `readme.md:167` and `scripts/run_offline_simulations.py:152`.
- Outputs: under `simulation_analysis/offline/<scenario>/<scheduler>/...` with `config.json` and `request_metrics.csv`.

### File Map (quick references)
- Entry & simulator: `vidur/main.py:1`, `vidur/simulator.py:1`.
- Scheduler registry: `vidur/scheduler/global_scheduler/global_scheduler_registry.py:1`.
- Block offline scheduler: `vidur/scheduler/global_scheduler/block_offline_global_scheduler.py:19`.
- Request timeline predictor(s): `vidur/request_timeline_predictor/simulate_request_timeline_predictor.py:1`, `noisy_simulate_request_timeline_predictor.py:12`.
- What‑if simulator: `vidur/scheduler/replica_scheduler/simulate_predict_replica_scheduler.py:1`.
- Trace replay loader: `vidur/request_generator/trace_replay_request_generator.py:1`.
- Types/enums: `vidur/types/global_scheduler_type.py:1`, `vidur/types/optimal_global_scheduler_target_metric.py:1`, `vidur/types/request_timeline_predictor_type.py:1`.

### Determinism & Seeds
- Simulation seed: `--random-seed` in offline runner; seeded inside `vidur/main.py:1` via `vidur.utils.random.set_seeds`.
- For parity checks (slow vs fast predict), keep `--fast-predict off` initially, validate equivalence on a subset, then enable.

---

## Performance Plan (Parity‑Preserving)

Goal: Complete 10k requests on 12 replicas (~120k what‑ifs) in ≈20 minutes for `block_offline` and `block_star_offline`, with identical prediction results.

- Phase 1 — In‑place snapshot/restore (fast‑predict)
  - Replace the remaining clone cost by simulating directly on the live replica scheduler with full state snapshot/restore around each what‑if.
  - Location: `vidur/scheduler/replica_scheduler/simulate_predict_replica_scheduler.py` (now delegates to `snapshot_replica_scheduler_state` / `restore_replica_scheduler_state` in `vidur/scheduler/utils/replica_state.py` when `copy_replica_scheduler=False`).
  - Identity: exact state restoration; no behavioral change.

- Phase 2 — Deterministic execution‑time memoization
  - Add an LRU cache for `__get_execution_time` keyed by a deterministic batch signature (e.g., `(stage_id, contains_prefill, tuple(batch.request_ids), tuple(batch.num_tokens))`).
  - Location: same module; keep existing `threshold_batch_size_for_time_estimation` path and augment the cache map.
  - Identity: pure memoization → bit‑for‑bit equality.

- Phase 3 — Hot‑loop micro‑optimizations
  - Pre‑bind hot attributes/functions to locals, reduce temporary objects in `simulate()`, `__push_batch`, `__pop_batch`.
  - Identity: no logic changes; expect ~10–20% speedup.

- Phase 4 — Safe lower‑bound pruning (optional)
  - Compute an exact lower bound for the target metric per replica (e.g., current tail completion + solo target timeline). If strictly worse than best seen so far, skip full what‑if for that replica.
  - Identity: only prunes replicas that cannot win under current metric; decisions unchanged.

- Phase 5 — Parallel evaluation (guarded)
  - Parallelize per‑replica what‑ifs for a single request (ThreadPool or ProcessPool) only when noise is disabled (`--block-noise 0`).
  - Optional deterministic‑noise mode (hash‑seed RNG per `(request_id, replica_id)`) behind a new flag if parallelism with noise is desired; note this changes historical results, so default remains sequential.

Expected impact (single‑thread): 3–6× vs current fast‑predict; with parallelism (noise=0), up to ~10× on many‑core CPUs, making ≈20 minutes feasible.

---

## Sanity Checks (Identity Tests)

Before enabling fast paths or new backends by default, run reproducible experiments and assert bit‑for‑bit equality of outputs.

- Test A: Parity on 12×120 (no noise)
  - Baseline (sequential, slow path):
    - `PYTHONPATH=. python scripts/run_offline_simulations.py \
       --schedulers block_offline block_star_offline \
       --num-replicas 12 \
       --trace-file data/trace_data/sharegpt/sharegpt_val_10k_llama2.csv \
       --predicted-trace-file data/trace_data/sharegpt/generate/llama/sharegpt-llama-7b-val-10k-predicted.json \
       --qps 32 --max-requests 120 --block-noise 0 --fast-predict off --block-parallel-enable off`
  - Fast predict (sequential): repeat with `--fast-predict on` and `--block-parallel-enable off`.
  - Process parallel: repeat with `--fast-predict on --block-parallel-enable on --block-noise 0`.
  - Compare `request_metrics.csv` and `config.json` for equality across all runs.

- Test B: Parity on 12×120 (with noise)
  - As in Test A, but remove `--block-noise 0` and add `--deterministic-noise on` for the parallel run. Expect exact equality across runs (slow vs fast vs process parallel).

- Test C: Per‑replica metric spot‑check
  - For a fixed early request, log the per‑replica predicted target metric values (e.g., min_latency) with slow vs fast vs parallel backends; assert equality to many decimals.

- Diff helper (pseudo):
  - Load both CSVs with pandas, sort by `request_id`, and assert `(df1.columns == df2.columns)` and `allclose(df1.values, df2.values, rtol=0, atol=0)`.

Only flip `--fast-predict on` and parallel backends for large runs after these checks pass.

---

## Parallelism Notes

- Why it helps: For each arriving request, Block/Block* runs one independent what‑if per replica. Running these concurrently reduces wall‑clock time by ≈min(workers, replicas, CPU cores).

- Safe case (identical results): when `--block-noise 0`, per‑replica what‑ifs can run in parallel and collate results in the original replica order (stable tie‑breaks). This preserves decisions and metrics.

- With noise > 0 (deterministic parallelism): enable a deterministic noise schedule and then parallelize safely.
  - CLI/config knobs (clean, conflict‑free):
    - `--block-parallel-enable on|off` (default off; process backend only)
    - `--deterministic-noise on|off` (default off; turn on when combining noise with parallelism)
  - Validation rule: if `--block-parallel-enable on` AND `--block-noise > 0` AND `--deterministic-noise off`, abort with a clear error suggesting `--deterministic-noise on` or `--block-noise 0`.

- Implementation detail:
  - Thread backend removed due to nondeterministic races; only process-based evaluation is supported.

Default remains sequential to guarantee parity unless the safe conditions above are met.

### Process Backend (Distributed Predictor Simulation)
- Motivation: The paper’s design is distributed and stateless across instances (see paper.tex:54–78). Threads improve wall‑clock somewhat, but true process‑level parallelism better matches the real system and avoids GIL contention. This section specifies a process backend that evaluates per‑replica what‑ifs in parallel processes with deterministic outcomes.

- CLI/config:
  - `--block-parallel-enable on|off` (process backend only)
  - `--deterministic-noise on|off` (required with noise > 0 under parallelism)
  - `--fast-predict on|off` (snapshot/restore mode; required for parity and speed)

- Implementation used: Persistent per‑replica workers (stateful): one long‑lived process per replica keeps a local replica scheduler, receives snapshots from the main process, restores state, and evaluates the what‑if. Mirrors the paper’s distributed predictor and minimizes serialization.

- Serialization and state snapshot
  - Use `vidur/scheduler/utils/replica_state.py` to snapshot/restore queues, running batches, allocation maps, and per‑request mutable fields.
  - Main process sends minimal snapshots + request payloads; workers restore and evaluate locally.

- Worker API (persistent per‑replica design)
  - Start: given `ReplicaConfig`, `SarathiSchedulerConfig`, and `ExecutionTimePredictorConfig`, construct a replica scheduler and attach an execution‑time predictor inside the worker. Apply `fast_predict` by calling `disable_copy_of_base_replica_scheduler()` on the predictor.
  - Evaluate(request, metric):
    1) Snapshot local replica scheduler state.
    2) Build a `SimulatePredictReplicaScheduler` with `copy_replica_scheduler=False`, attach the target request, run `simulate()` until target completes.
    3) Compute the metric via `vidur/request_timeline_predictor/base_request_timeline_predictor.get_target_metric_value`.
    4) Restore the snapshot and return the scalar metric.
  - Shutdown: graceful process termination at atexit.

- Determinism and noise
  - Use `DeterministicNoiseProvider` keyed by `(request_id, replica_id, metric[, element_index])` with the simulation seed to preserve results across degrees of parallelism. See `vidur/request_timeline_predictor/deterministic_noise.py:1`.
  - Enforce: if `parallel` and `noise_fraction > 0` and `deterministic_noise` is off → raise (already enforced in CLI and scheduler constructors).

- Caching and performance notes
  - Batch execution‑time caching: keep per‑process caches only (no cross‑process sharing). Existing cache maps are disabled in parallel mode to avoid contention; for process workers, it’s safe to enable per‑worker caches.
  - Execution‑time memoization is already present per simulation via a deterministic key. Persistent workers will benefit from warm caches over time.
  - Expect 3–6× single‑machine speedup vs sequential fast‑predict; with N workers ≈ min(replicas, CPU cores).

- Failure handling
  - Timeouts per replica evaluation (e.g., 5–30s) with fallback to default large metric (so a stuck replica won’t be selected).
  - On worker crash, respawn lazily and mark metrics for that evaluation as worst‑case.

- File map
  - `vidur/scheduler/global_scheduler/block_offline_global_scheduler.py`: process evaluator when `parallel=True`.
  - `vidur/scheduler/global_scheduler/process_workers.py`: per‑replica worker processes.
  - `vidur/scheduler/utils/replica_state.py`: snapshot/restore.
  - `scripts/run_offline_simulations.py`: `--block-parallel-enable` toggle.

- Validation plan
  - Parity (12×120): Bit‑for‑bit equality across slow vs fast vs thread/process (when implemented), with noise=0 and with noise+deterministic noise.
  - Large‑scale 10×: Run 120× replicas at QPS 320 over the full 10k trace. Compare results between thread vs process backends (identical CSVs), and report wall‑clock speedup.
  - Performance: record wall‑clock time, requests/sec, CPU utilization for 12× and 120× runs.

- Step‑by‑step implementation plan
  1) Extend `BlockOfflineGlobalScheduler` to accept `process` backend and wire a placeholder dispatcher.
  2) Introduce `process_workers.py` with two modes:
     - `ProcessPoolEvaluator` (stateless tasks)
     - `PerReplicaWorker` (long‑lived processes)
  3) Add serialization utilities that build a minimal snapshot from a live replica scheduler using `snapshot_replica_scheduler_state()`; ensure the dict is picklable.
  4) Implement a reconstruction helper that applies a snapshot to a freshly constructed replica scheduler inside the worker.
  5) In workers, construct `SimulateRequestTimelinePredictor`, attach the execution‑time predictor, and call `disable_copy_of_base_replica_scheduler()` when `fast_predict` is set.
  6) Implement deterministic noise wiring in workers via `DeterministicNoiseProvider` (seed = simulation seed).
  7) Add timeouts and error handling: default worst‑case metric on timeout; respawn failed workers.
  8) Add CLI plumbing in `scripts/run_offline_simulations.py` help text for the process backend; keep guard rails (deterministic noise required if noise > 0).
  9) Tests: run the Sanity Checks A–C at 12×120 with backend=`thread` vs `process`; assert CSV equality.
  10) Benchmark: add Large‑scale 10× (120×, QPS 320, 10k req) to validation; document results under `simulation_analysis` and summarize with `simulation_analysis/summarize_request_metrics.py`.

---

## Deterministic Noise Schedule (for parallel Block*)

Purpose: Enable parallel per‑replica what‑ifs with noise while keeping results deterministic and independent of call ordering.

- Approach
  - Pre‑generate a noise multiplier in [1−f, 1+f] (f = noise_fraction) for every `(request_id, replica_id, metric)` at simulation start using a seeded RNG.
  - Expose a `DeterministicNoiseProvider` consulted by `NoisySimulateRequestTimelinePredictor` instead of drawing per call.
  - Indexing uses stable IDs: `Request.id` (generation order) and `replica_id`.

- Integration plan
  1) Add `DeterministicNoiseProvider` with API `get_multiplier(request_id, replica_id, metric_name)`.
  2) Plumb via `NoisySimulationRequestTimelinePredictor.configure(...)` (or from `Simulator`) so predictors can look it up.
  3) Add a CLI/config boolean `--deterministic-noise on|off` (maps to config), default off. Turn on when using noise with parallelism.
  4) Keep distribution identical (uniform by default); seed from `--random-seed`.

- Notes
  - Memory cost for 10k×12 is modest (≈120k floats per metric). To match previous independence between metrics, either precompute per‑metric multipliers or use one shared per `(request, replica)` if acceptable.
  - With `--deterministic-noise on`, parallelism does not affect noise consumption; results are stable across degrees of parallelism.

### Usage Examples
- Fast and identical (no noise):
  - `... --block-noise 0 --fast-predict on --block-parallel-enable on`
- With noise and still deterministic under parallelism:
  - `... --block-noise 10 --deterministic-noise on --fast-predict on --block-parallel-enable on`
- Force sequential (historical parity):
  - `... --block-parallel-enable off`

## Agent Tips
- When asked to compare Block vs baselines:
  - Block/Block* are predictive and use per‑replica what‑if scoring; baselines are heuristic and non‑predictive.
  - Block uses ground truth lengths; Block* uses estimator‑predicted lengths from the provided predicted trace file.
- For performance issues in offline Block:
  - Prefer enabling `--fast-predict on` after verifying parity; large speedups come from avoiding deep copies and from batch‑time caching.
- Minimal command to replay a small scenario:
  - `PYTHONPATH=. python scripts/run_offline_simulations.py --schedulers block_offline infass_pp --num-replicas 12 --trace-file data/trace_data/sharegpt/sharegpt_val_10k_llama2.csv --qps 32 --max-requests 200 --fast-predict off`
  - For Block*: add `block_star_offline` and ensure `--predicted-trace-file` points to the estimator outputs.

## 2025-09-19 Status (Handoff)
- Cleanup complete: removed ad‑hoc debug outputs; preserved `experiment_output/offline/qps32_full` as stable heuristics reference.
- Heuristic fixes: `infass_pp` and `llumnix_minus` now match paper definitions (usedMemory/batchSize vs (usedMemory+prefillMemory)/batchSize). Llumnix‑ outperforms INFaaS++ on 12×1k slice (E2E/TTFT).
- Metrics nomenclature: headline metrics are E2E latency and TTFT. CSV also includes `request_waiting_time` (alias of `request_scheduling_delay`) to avoid confusion with paper’s “scheduling overhead”.
- Progress logs: simulator now prints progress at least every 5 minutes in addition to count milestones.

### Active Experiment Plan
1) Small‑slice Block speedup (12 replicas × 240 requests)
   - Purpose: measure fast‑predict speedup; skip full sequential runs.
   - Commands (already running): see simulation_analysis/README.md → How to Reproduce.
   - Logs: `simulation_analysis/speed/fast_off/run.log`, then `simulation_analysis/speed/fast_on/run.log`.
   - Summarize: `PYTHONPATH=. simulation_analysis/summarize_request_metrics.py simulation_analysis/speed --schedulers block_offline block_star_offline --csv simulation_analysis/speed/summary.csv`.

2) QPS=32 full sweep (12 replicas; all requests)
   - Uses fast‑predict on + parallel for Block/Block*.
   - Launcher: `./simulation_analysis/run_qps32_schedulers.sh simulation_analysis/full_runs/qps32_fast on`.
   - Logs per scheduler under `simulation_analysis/full_runs/qps32_fast/<scheduler>/run.log`.
   - Summarize: `PYTHONPATH=. simulation_analysis/summarize_request_metrics.py simulation_analysis/full_runs/qps32_fast --csv simulation_analysis/qps32_full_summary.csv`.

3) Large‑scale 10× sweep (120 replicas; QPS 320)
   - Launcher: `./simulation_analysis/run_large_scale.sh simulation_analysis/large_scale on`.
   - Logs per scheduler under `simulation_analysis/large_scale/<scheduler>/run.log`.
   - Summarize: `PYTHONPATH=. simulation_analysis/summarize_request_metrics.py simulation_analysis/large_scale --csv simulation_analysis/large_scale_summary.csv`.

### Monitoring & Recovery
- Monitor progress (non‑invasive):
  - `tail -f simulation_analysis/full_runs/qps32_fast/block_offline/run.log`
  - `tail -f simulation_analysis/large_scale/block_offline/run.log`
  - Progress prints every ~5 min: `Processed X/Y requests (Z%) after Ns`.
- Check processes:
  - `ps -eo pid,ppid,etime,cmd | rg -e "run_offline_simulations.py|run_qps32_schedulers.sh|run_large_scale.sh"`
- If a job exits early, rerun via the same launchers above; outputs write under the specified roots.
