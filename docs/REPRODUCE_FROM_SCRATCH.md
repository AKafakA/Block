# Reproduce the SYSTOR '26 Paper Results from Scratch

**Goal**: Reproduce all main figures + ablations + generality + Section 6.6 + CPU overhead from a clean A30 + A100 cluster reservation.

**Estimated total time**: ~60 GPU-hours (A30) + ~12 GPU-hours (A100) when nothing breaks.

## ⚡ Artifact Evaluation Quick Path

For artifact evaluators or anyone wanting one-command-per-phase reproduction, **use the wrapper scripts under** `block/exp/end_to_end_exp_scripts/artifact_evaluation/`:

```bash
AE=block/exp/end_to_end_exp_scripts/artifact_evaluation

# Prereqs (one-time per cluster / device class)
sh $AE/00_setup_cluster.sh                        # verify + apply patches (~1h)
sh $AE/01_profile_vidur_a30.sh                    # Vidur HW profile A30 (~3-4h, single A30)
sh $AE/02_profile_vidur_a100.sh                   # Vidur HW profile A100 (~5-6h, single A100 node)
parallel-scp -h block/config/hosts -r data/profiling Block/data/
sh $AE/03_train_length_model.sh llama             # RoBERTa length model Llama (~3h, dev-GPU VM)
sh $AE/03_train_length_model.sh qwen              # RoBERTa length model Qwen (~3h)
parallel-scp -h block/config/hosts -r model/ Block/
sh $AE/04_warmup_llama.sh                         # predictor RF cache warmup (~5m)

# Section 6.2 — Latency Prediction Metrics
sh $AE/05_prediction_error_a30.sh                 # A30 Llama-7B (~1h)
sh $AE/06_prediction_error_a100.sh                # A100 Llama-70B (~1.5h)

# Sections 6.3 onwards
sh $AE/07_main_sweep_a30.sh                       # 6.3 (~26h)
sh $AE/08_n_ablation.sh                           # 6.4 + 6.6 (~1h)
sh $AE/09_burstiness.sh                           # 6.5 (~30m)
sh $AE/10_error_heatmap.sh                        # 6.6 (~2.5h)
sh $AE/11_capacity_refine.sh                      # 6.x (~3h)
sh $AE/12_generality.sh                           # 6.7 (~5h, includes Qwen warmup)
sh $AE/13_cpu_overhead.sh                         # 6.7 (~45m, needs CPU patches)
sh $AE/14_a100_llumnix.sh                         # 6.8 (~3h)
sh $AE/15_a100_block.sh                           # 6.8 (~9h)

# Verification helpers (anytime):
sh $AE/util_predictor_health.sh
python3 $AE/util_verify_npz.py experiment_results_a30/<phase>/
```

Each wrapper:
- Validates code patches are present (CPU pipeline, sequential deploy, verify_predictors timeout)
- Calls the underlying experiment script
- Syncs NPZs to `experiment_results_a30/<phase>/` immediately
- Prints capacity values / CPU stats from the log

Full README: [`block/exp/end_to_end_exp_scripts/artifact_evaluation/README.md`](../block/exp/end_to_end_exp_scripts/artifact_evaluation/README.md)

The rest of this document explains the **manual** reproduction path with detailed code patches and per-phase commands. Use the wrapper scripts for AE; use the manual path for debugging.

---

## Prerequisites

### Cluster reservations
- **A30**: 12× CloudLab d7525 nodes
- **A100**: 2× CloudLab d8545 nodes (4× A100-40GB SXM4 each)

### HuggingFace token
Set on local + every cluster node:
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
echo "export HF_TOKEN=hf_xxxxxxxxxxxxxxxx" >> ~/.bashrc
```

### Models needed (cached on each node)
- `meta-llama/Llama-2-7b-hf` (~14 GB) — A30
- `Qwen/Qwen2-7B` (~15 GB) — A30 (generality)
- `meta-llama/Llama-2-70b-hf` (~140 GB) — A100

---

## Step 0: Cluster setup (NEVER manual)

```bash
# Local
git clone <repo> Block_exp && cd Block_exp

# Configure host file
ls block/config/hosts  # 12 lines, one per A30 node
ls block/config/host_configs.json  # IP:port mapping

# Setup ALL 12 A30 nodes (NEVER manual installs — fix setup.sh if it fails)
parallel-ssh -h block/config/hosts -i "bash" < block/setup.sh

# Verify
parallel-ssh -h block/config/hosts -i "python -c 'import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)'"
# Expect: vllm-block branch, torch 2.x, transformers==4.50.3
```

**If setup.sh fails on any node**: fix the script, RELOAD the node (CloudLab reload), re-run setup.sh once. Never improvise installs.

---

## Step 1: Apply required code patches

These patches are critical and **MUST be in place before any experiment**. They were retrofitted during the campaign.

### Patch A: Sequential predictor deploy (in `block/exp/experiment.sh`)
```bash
# In experiment.sh, the predictor launch loop should be sequential, not parallel batches:
suffix_range=$(seq 1 $PREDICTOR_WORKERS)
for suffix in $suffix_range; do
    sh "${script_base}_${suffix}.sh" $PREDICTOR_CONFIG_PATH ... > /dev/null 2>&1
    sleep 2
done
sleep 30  # settle
```
**Why**: parallel deploy had ~8% per-node failure rate (race on pandas dataframe load).

### Patch B: CPU tracking pipeline (3 files)

**File 1**: `block/predictor/api_server.py`
```python
import psutil
# Module level
enable_cpu_tracking = False
process = None
cpu_cores = 0

# In init_app() after global decl:
enable_cpu_tracking = getattr(args, 'enable_cpu_tracking', False)
if enable_cpu_tracking:
    process = psutil.Process()
    cpu_cores = psutil.cpu_count()
    logging.info("CPU tracking enabled for predictor (%d logical cores)", cpu_cores)

# In predict() endpoint, after metric computed:
if enable_cpu_tracking and process is not None:
    cpu_percent = process.cpu_percent()
    metric["cpu_percent"] = cpu_percent
    metric["cpu_cores"] = cpu_cores
    memory_info = process.memory_info()
    metric["memory_rss_mb"] = memory_info.rss / (1024 * 1024)

# Add --enable_cpu_tracking arg in argparse
```

**File 2**: `block/global_scheduler/api_server.py` (after line 146 in the `if predict_results:` block)
```python
cpu_percents = [r.get('cpu_percent', 0) for r in predict_results if 'cpu_percent' in r]
memory_rss = [r.get('memory_rss_mb', 0) for r in predict_results if 'memory_rss_mb' in r]
cpu_cores_vals = [r.get('cpu_cores', 0) for r in predict_results if 'cpu_cores' in r]
single_metric['mean_cpu_percent'] = float(np.mean(cpu_percents)) if cpu_percents else 0
single_metric['max_cpu_percent'] = float(max(cpu_percents)) if cpu_percents else 0
single_metric['mean_memory_rss_mb'] = float(np.mean(memory_rss)) if memory_rss else 0
single_metric['max_memory_rss_mb'] = float(max(memory_rss)) if memory_rss else 0
single_metric['cpu_cores'] = int(cpu_cores_vals[0]) if cpu_cores_vals else 0
```

**File 3**: `block/benchmark/benchmark_serving.py` — 5 sites:
1. `__init__`: `self._cpu_percents = []`, `self._memory_rss_mb = []`, `self._cpu_cores = []`
2. Per-request loop: collect `output['mean_cpu_percent']`, `output['mean_memory_rss_mb']`, `output['cpu_cores']`
3. Return tuple from `benchmark()`: add `m._cpu_percents`, `m._memory_rss_mb`, `m._cpu_cores`
4. Caller unpack: add `cpu_percents`, `memory_rss_mb`, `cpu_cores_list`
5. Save to NPZ: add `cpu_percents`, `memory_rss_mb`, `cpu_cores` to `data` dict

### Patch C: verify_predictors.sh — bounded SSH timeout (in `block/exp/end_to_end_exp_scripts/a30_main/verify_predictors.sh`)
```bash
# Inner ssh: explicit </dev/null + timeout
timeout 30 ssh -n -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "$host" \
    "cd Block && export PYTHONPATH=. && nohup python block/predictor/api_server.py ... < /dev/null > log 2>&1 &" &

# Bounded wait with 60s cap (replaces plain `wait`)
end_time=$(($(date +%s) + 60))
while [ -n "$(jobs -p)" ] && [ "$(date +%s)" -lt "$end_time" ]; do
    sleep 2
done
[ -n "$(jobs -p)" ] && kill $(jobs -p) 2>/dev/null
wait 2>/dev/null || true
```
**Why**: `nohup &` over SSH can hold the SSH session open indefinitely. Without timeout, `wait` blocks forever.

### Sync patches to all 12 A30 nodes
```bash
parallel-scp -h block/config/hosts block/predictor/api_server.py Block/block/predictor/api_server.py
parallel-scp -h block/config/hosts block/global_scheduler/api_server.py Block/block/global_scheduler/api_server.py
parallel-scp -h block/config/hosts block/benchmark/benchmark_serving.py Block/block/benchmark/benchmark_serving.py
parallel-scp -h block/config/hosts block/exp/experiment.sh Block/block/exp/experiment.sh
parallel-scp -h block/config/hosts block/exp/end_to_end_exp_scripts/a30_main/verify_predictors.sh \
    Block/block/exp/end_to_end_exp_scripts/a30_main/verify_predictors.sh
```

---

## Step 2: Pre-warm Llama predictor cache (one-time)

```bash
sh block/exp/end_to_end_exp_scripts/warmup.sh "meta-llama/Llama-2-7b-hf"
# ~5 min. Trains ONE predictor on each node, saves to ~/Block/cache/*.pkl
# After this all subsequent 16-predictor deploys are fast (cache hit, ~30s each)
```

**Why**: Without warmup, 16 concurrent predictors race on lock files for ~5 min. With warmup, deploys take ~1 min total.

---

## Phase 1.1 — Main TTFT/throughput sweep (~20 hours)

```bash
nohup sh block/exp/end_to_end_exp_scripts/a30_main/main_experiment.sh \
    > /tmp/phase11_main.log 2>&1 &
```

Settings inside `main_experiment.sh`:
- `QPS="20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36"` (full range)
- 6 schedulers: `min_new_request_latency` (Block-Fanout est+oracle), `min_lunmnix_load` (Llumnix--), `min_infass_load` (INFaaS++), `round_robin` (RR), `request_per_seconds` (MinQPM), `random` (Random)
- N=12 for all (Fanout-style for Block, broadcast for baselines)
- `RESTART_VLLM=true` between schedulers (predictor metric must reset)

Outputs: `~/Block/experiment_output/main/sharegpt/<scheduler>/qps_X_..._batch_48_chunk_512/benchmark_all_metrics.npz`

After completion sync to local:
```bash
mkdir -p experiment_results_a30/phase11_main
rsync -az asdwb@<MASTER>:~/Block/experiment_output/main/ experiment_results_a30/phase11_main/
```

---

## Phase 1.2 — Po2 main sweep (~6 hours)

```bash
nohup sh block/exp/end_to_end_exp_scripts/a30_main/po2_main_experiment.sh \
    > /tmp/phase12_po2.log 2>&1 &
```

- Po2 (N=2): oracle + estimated × QPS 20-36 = 34 runs
- `RESTART_VLLM=true` for first run, then false within Po2 (same scheduler)

Sync: `experiment_results_a30/phase12_po2/`

---

## Phase 2 — N-ablation (~30 min)

```bash
nohup sh block/exp/end_to_end_exp_scripts/a30_main/n_ablation_exp.sh \
    > /tmp/phase2_n_ablation.log 2>&1 &
```

- N ∈ {4, 6, 8} × QPS=30 = 3 cells
- Sync: `experiment_results_a30/phase2_n_ablation/`

---

## Phase 3.1 — Burstiness ablation (~30 min)

**IMPORTANT — Llumnix N=12 fix**: in `burstiness_exp.sh`, ensure per-scheduler N branching:
```bash
if [ "$scheduler" = "min_new_request_latency" ]; then
    USE_LENGTH_ESTIMATION="true"
    N_THIS_RUN="$N_SELECTED"   # Po2 = 2
else
    USE_LENGTH_ESTIMATION="false"
    N_THIS_RUN="12"   # Llumnix uses broadcast, NOT N=2
fi
```

Then run:
```bash
nohup sh block/exp/end_to_end_exp_scripts/burstiness_exp.sh \
    > /tmp/phase31_burstiness.log 2>&1 &
```
- Po2-est + Llumnix-N12 × burst {0.25, 0.5, 1.0, 2.0} @ QPS=32
- `RESTART_VLLM=true` between scheduler types
- Sync: `experiment_results_a30/phase3_1_burstiness_po2/`

---

## Phase 3.2 — Error injection heatmap (~2.5 hours)

**IMPORTANT — fresh-deploy-per-cell** for absolute comparisons. The script must:
- For each (length_err, latency_err) cell:
  - reset.sh
  - deploy vLLM + 16 predictors fresh
  - verify_predictors hard gate
  - launch scheduler with `--length_error_pct $le --latency_error_pct $la`
  - run benchmark @ QPS=32
  - sync NPZ immediately

```bash
nohup sh block/exp/end_to_end_exp_scripts/error_heatmap_exp.sh \
    > /tmp/phase32_heatmap.log 2>&1 &
```
- Po2-est × 15 cells (skip baseline (0,0) — covered by Phase 1.1)
- Sync: `experiment_results_a30/phase3_2_error_heatmap_po2/`

**Note**: Single-deploy variance is ±4% e2e. Do not interpret cell deltas <2% as significant.

---

## Phase 4.1 — Float capacity refinement (~3 hours)

```bash
nohup sh block/exp/end_to_end_exp_scripts/a30_main/float_capacity_refine.sh \
    > /tmp/phase41_float.log 2>&1 &
```

Per scheduler: integer bracket from int sweep capacity → float refine at 0.1 resolution → 9.X (9000-10000ms) or 10.X (10000-11000ms) early-stop.

Schedulers covered: po2_est, po2_oracle, fanout_est, fanout_oracle, llumnix.

Output: capacities printed to `/tmp/a30_phase4_1_float_results.txt`. Sync NPZs to `experiment_results_a30/phase4_1_float/`.

---

## Phase 4.2 — Generality (chunk2048 + batch24 + BurstGPT + Qwen)

### chunk2048 + batch24
```bash
nohup sh block/exp/end_to_end_exp_scripts/a30_main/generality_float_simple.sh \
    > /tmp/phase42_simple.log 2>&1 &
```
- Same schedulers, vary chunk_size and batch_cap
- Sync: `experiment_results_a30/phase4_2_generality/`

### BurstGPT (~1.5 hours)
```bash
nohup sh block/exp/end_to_end_exp_scripts/a30_main/burstgpt_float_search.sh \
    > /tmp/phase42_burstgpt.log 2>&1 &
```
- Po2-oracle + Llumnix-N12 capacity search on BurstGPT dataset
- Sync: `experiment_results_a30/phase4_2_burstgpt_refine/`

### Qwen (~2 hours)
```bash
# IMPORTANT: pre-warm Qwen cache first to avoid OOM
sh block/exp/end_to_end_exp_scripts/warmup.sh "Qwen/Qwen2-7B"
# Then full capacity search
nohup sh block/exp/end_to_end_exp_scripts/a30_main/qwen_capacity_search.sh \
    > /tmp/phase42_qwen.log 2>&1 &
```
- Po2-est + Po2-oracle + Llumnix-N12 capacity search
- **Smart seeds**: Po2-oracle seed should be Po2-est-cap + 2-3 (saves probes); Llumnix seed should be Po2-est-cap - 5
- Sync: `experiment_results_a30/phase4_2_qwen/`

---

## Phase 7a — CPU tracker for Po2-est (~45 min)

Requires Patch B applied + synced.

```bash
# Restart predictors with --enable_cpu_tracking flag (10th arg = "true")
# Use cpu_tracker_full.sh which handles this:
nohup sh block/exp/end_to_end_exp_scripts/a30_main/cpu_tracker_full.sh \
    > /tmp/phase7_cpu_tracker.log 2>&1 &
```
- Po2-est × QPS {20, 24, 28, 32, 36}
- NPZ will contain `cpu_percents`, `memory_rss_mb`, `cpu_cores` arrays
- Sync: `experiment_results_a30/phase7_cpu_tracker_po2_v2/`

**Verification**: `python3 -c "import numpy as np; d=np.load(...); print('cpu_present:', 'cpu_percents' in d.keys() and len(d['cpu_percents'])>0)"`

---

## Phase 7b — Section 6.6 N-tunable (Po4 + Po8) (~30 min)

```bash
# Restart predictors WITHOUT --enable_cpu_tracking (clean comparison vs other data)
nohup sh block/exp/end_to_end_exp_scripts/a30_main/po4_then_po8_capacity.sh \
    > /tmp/phase7_po4po8.log 2>&1 &
```
- Po4-est seed=32, capacity search → ~31.9
- Po8-est seed=33, capacity search → ~31.7
- Sync: `experiment_results_a30/phase7_po4po8/`

If you want manual probes (faster):
- Po4: probe QPS=32, then 31.9 (decision)
- Po8: probe QPS=32, 31.8, 31.7

---

## A100 Phase 5 — Llumnix + Block sweeps (~10 hours)

### Setup A100 (2 nodes)
```bash
# Use deploy_block.sh on each A100 node — TP=4, Llama-2-70B
sh block/exp/end_to_end_exp_scripts/a100_supplementary/deploy_block.sh
```

### Llumnix sweep
```bash
nohup sh block/exp/end_to_end_exp_scripts/a100_supplementary/run_benchmark.sh \
    sweep llumnix 10000 "16 17 18 19 20 24 28 32 36" \
    > /tmp/a100_llumnix.log 2>&1 &
```

### Block sweeps (4 configs)
**CRITICAL**: `run_benchmark.sh` writes to fixed path `experiment_output/benchmark_output/block_sweep/block_qps${qps}`. Successive configs WILL overwrite. **Sync NPZs to per-config local dir IMMEDIATELY after each config completes.**

```bash
# Config 1: Po2 + chunked-prefill
nohup sh block/exp/end_to_end_exp_scripts/a100_supplementary/run_benchmark.sh \
    sweep block 10000 "16 20 24 28 32 36" \
    > /tmp/a100_po2_cp.log 2>&1 &
# After completion: rsync to a100_results/po2_cp/ THEN start next config
rsync -az <a100>:~/Block/experiment_output/benchmark_output/block_sweep/ a100_results/po2_cp/

# Config 2: Fanout + CP — repeat
# Config 3: Po2 -noCP — repeat
# Config 4: Fanout -noCP — repeat
```

**ALWAYS verify benchmark uses `--use_estimated_response_lens`** (was missing in earlier run, caused 8h waste).

---

## Verification & validation

### After each phase:
1. Sync NPZs to local
2. Check NPZ contents:
   ```python
   import numpy as np
   d = np.load(path)
   print(d.keys())  # Verify expected fields
   print(f"n_requests={len(d['request_latencies'])}")  # Should be 10000 for Llama, 9963 for Qwen
   print(f"TTFT_P99={np.percentile(d['prefill_token_latencies'], 99):.0f}ms")
   ```

### Predictor health audit (run during long sweeps):
```bash
for host in $(cat block/config/hosts); do
    short=$(echo $host | grep -oP 'd7525-10s\K\d+')
    count=$(ssh -n -o ConnectTimeout=3 "$host" \
        "ss -lnt | grep -E '0\.0\.0\.0:(8100|8300|8400|8500|8600|8700|8800|8900|9000|9100|9200|9300|9400|9500|9600|9700) ' | wc -l")
    echo "$short: $count/16"
done
```
**Any node showing <16 = corrupted data. Stop, abort, investigate.**

### Final sync to two locations
```bash
# Primary
rsync -az asdwb@<master>:~/Block/experiment_output/ experiment_results_a30/

# Backup mirror
rsync -az asdwb@<master>:~/Block/experiment_output/ experiment_results_a30_backup/

# Verify md5 match on samples before lease release
```

---

## Time budget

| Phase | Time | Cumulative |
|---|---|---|
| 0: Setup | 1h | 1h |
| 1: Patches + sync | 30 min | 1.5h |
| 2: Warmup Llama cache | 5 min | 1.6h |
| Phase 1.1 (main 6 sched) | 20h | 21.6h |
| Phase 1.2 (Po2) | 6h | 27.6h |
| Phase 2 (N-ablation) | 30 min | 28.1h |
| Phase 3.1 (burstiness) | 30 min | 28.6h |
| Phase 3.2 (heatmap fresh-per-cell) | 2.5h | 31.1h |
| Phase 4.1 (float refine) | 3h | 34.1h |
| Phase 4.2 generality (chunk+batch+burstgpt) | 3h | 37.1h |
| Phase 4.2 Qwen (warmup + sweep) | 2h | 39.1h |
| Phase 7a CPU tracker | 45 min | 39.9h |
| Phase 7b Po4/Po8 | 30 min | 40.4h |
| Buffer + sync | 2h | **42h A30 total** |
| A100 setup + sweeps | 12h | **12h A100 total** |

**Total wall clock from clean cluster reservation: ~3 days A30 + ~1 day A100.** Plan reservations accordingly.

---

## Known gotchas (from this campaign)

1. **Setup.sh failure**: Fix script, RELOAD nodes, run once. Never improvise.
2. **RESTART_VLLM=false**: NEVER set this — predictors must reset between schedulers.
3. **A100 fixed output_dir**: Sync NPZs immediately after each config; don't batch.
4. **15/16 predictors**: Hard abort, full reset. Never accept as transient.
5. **Llumnix N**: Always N=12 (broadcast). Never N=2.
6. **CPU tracking pipeline**: Patches A+B+C all required + synced + benchmark verified to have `cpu_percents` in NPZ.
7. **OOM with Qwen**: Pre-warm cache before 16-predictor deploy.
8. **verify_predictors hang**: Make sure Patch C is in place (timeout + bounded wait).
9. **Heatmap noise**: Single-deploy ±4% — for absolute comparisons, fresh-deploy-per-cell required.
10. **State-machine cron interfering with manual intervention**: Disable via `CronDelete <id>` before manual takeover.

---

## See also
- `AE.md` (repo root, systor-ae branch) — figure/table → script reproduction map
