# Operational Rules — Lessons from 648+ GPU-hours of Mistakes

These rules are distilled from 18+ documented incidents across the Block and Cara projects. Every rule exists because violating it wasted significant time or compute.

---

## Rule 1: NEVER silently change experiment parameters

**What**: Never change hyperparameters (batch_size, max_length, chunk_size, num_query_predictor, QPS, error_pct) without explicit user approval. Never add "sensible defaults" to scripts. If a parameter isn't specified, leave it unset.

**Why**: Changing max_length from 1024 to 512 to avoid OOM silently truncated 5% of prompts, invalidating all Phase 1 results. 9 out of 18 incidents were caused by silent design decisions.

**When facing OOM or infrastructure issues**: (1) STOP, (2) report the issue, (3) propose options with trade-offs, (4) WAIT for user decision.

---

## Rule 2: NEVER corrupt existing working state

**What**: Never `pip install` into system Python or existing working venvs — create a NEW venv. Never overwrite source data — create versioned copies. Never edit configs used by running processes. Never manually install on CloudLab/remote nodes — always use `setup.sh`.

**Why**: System pip installs break OS packages. Modifying running venvs causes cascading failures. Each incident wastes hours debugging.

**Test**: "Does this modify something that currently works?" If yes → COPY FIRST.

---

## Rule 3: ALWAYS smoke test before long runs

**What**: Run 3-5 samples with the EXACT same code/script/config that will run the full job. Check outputs manually: are values reasonable? Parse rate 100%? Distribution not degenerate? Only after smoke test passes, launch the full run. If ANY code changes after smoke test, re-test.

**Why**: Repeated pattern: launching untested code wastes hours. A 3-minute smoke test catches 90% of bugs that cost 4+ hours to discover mid-run.

**For Block specifically**: Send a warmup request to the global scheduler before timed benchmarks — first prediction takes ~5s for model loading.

---

## Rule 4: ALWAYS verify code matches paper/design

**What**: At start of any session modifying scheduling or prediction code: re-read the relevant paper section, trace one request through the full code path, verify every equation has a matching code line, check function names match what they actually compute.

**Why**: March 27 audit found 5 critical silent bugs that had been wrong for weeks — "fused" models actually separate, budget filter wrong formula, balance score wrong proxy, predict_ttft returning E2E, wrong normalization scope. All "worked" — just sub-optimally.

---

## Rule 5: ALWAYS use official libraries

**What**: Use official libraries (prometheus-eval, deepeval, routellm) instead of reimplementing. Copy parameters from official docs/examples. If a library doesn't work, read its source code before debugging.

**Why**: Custom Prometheus scoring used wrong template, wrong max_tokens, wrong reference, wrong scale, wrong decoding — 8+ GPU-hours wasted. The official library handles all of this correctly.

---

## Rule 6: ALWAYS report errors honestly

**What**: If a model fails to load, the system MUST fail — not silently degrade. If results look suspicious (identical values, degenerate distributions), investigate BEFORE reporting success.

**Why**: Silent degradation masks critical failures and produces false results.

---

## Rule 7: Preserve data pipeline integrity

**What**: Never add parameters to benchmark scripts without user approval. Pass ALL available fields through the data pipeline — dropping fields silently is as bad as adding wrong ones. Preserve raw data until verified and backed up. Verify data quality immediately after first batch.

**Why**: Three bugs in latency pipeline wasted 648 GPU-hours. Concurrency cap silently added, output_tokens field dropped, no source traceability.

---

## Rule 8: Use correct machine for correct task

| Machine | Use for | NOT for |
|---------|---------|---------|
| **VPS** (openclaw) | Code, writing, rsync, git | ANY GPU work |
| **GPU VM** (gxp-l40s-2) | Quick tests <10min | Long training (dept warned) |
| **CSD3 INTR** | Medium training <1h | Long runs (killed) |
| **CSD3 sbatch** | Long training (hours-days) | Quick experiments |
| **Vast.ai** | Medium-long training | Large models (>40GB) |
| **CloudLab** | Online serving experiments | Training, offline work |

---

## Rule 9: SSH access discipline

- **CloudLab**: plain SSH works reliably (no ControlMaster needed)
- **Cambridge/CSD3/Vast**: ALWAYS use SSH ControlMaster (flaky networks)
- **CSD3**: never attempt without user confirming MFA done
- If SSH fails, try once more then STOP — don't spam connections

---

## Rule 10: CSD3 workflow — interactive before batch

1. Install deps on login node (no GPU needed)
2. Smoke test via `srun --qos=INTR` (interactive A100, <1h)
3. Only after smoke passes, submit `sbatch` for full run

Never skip step 2. It catches 90% of issues (missing packages, wrong paths, OOM).

---

## Rule 11: Never modify remote cluster code directly

All code fixes must be made locally and deployed via `setup.sh` or `rsync`/`scp`. Never run `git checkout`, `git stash`, or `git pull` on cluster nodes. Remote code must stay consistent with what experiment scripts expect.

**Why**: `git checkout` on remote nodes reverted local fixes → 2+ hours debugging predictor failures.

---

## Rule 12: Never use wrong metrics

- **Quality prediction**: MAE, MAPE
- **Routing quality**: BestAcc (% picking best model), RankCorr (per-prompt cross-model)
- **Length prediction**: MAE (tokens), bucket accuracy
- **NEVER** use Spearman rho (measures within-model ranking, useless for cross-model routing)

---

## Block-Specific Rules

### Installation order (CRITICAL)
1. vLLM FIRST (`VLLM_USE_PRECOMPILED=1 pip install --editable .`)
2. PyTorch AFTER (`torch==2.6.0+cu126`)
3. transformers pinned (`==4.50.3`)
4. `export VLLM_USE_V1=0` before ANY vLLM process

### Predictor deployment
- Deploy in batches of 8, sleep 10s between (concurrent model loading → OOM)
- Set `profiling_sampling_rate=0.0` during experiments (0.1 adds ~20% overhead)
- Use cache-aware deployment: 1 predictor trains sklearn models, copy cache, then deploy rest

### Benchmarking
- Always send warmup request before timed benchmark
- Binary search capacity (don't waste runs on full QPS sweep)
- Reserve port 8200 for global scheduler only
