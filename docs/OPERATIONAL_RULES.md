# Operational Rules — Lessons from 648+ GPU-hours of Mistakes

These rules are distilled from documented incidents across the Block project. Every rule exists because violating it wasted significant time or compute.

---

## Rule 1: NEVER silently change experiment parameters

**What**: Never change hyperparameters (batch_size, max_length, chunk_size, num_query_predictor, QPS, error_pct) without explicit user approval. Never add "sensible defaults" to scripts. If a parameter isn't specified, leave it unset.

**Why**: Changing max_length from 1024 to 512 to avoid OOM silently truncated 5% of prompts, invalidating results. 9 out of 18 incidents were caused by silent design decisions.

**When facing OOM or infrastructure issues**: (1) STOP, (2) report the issue, (3) propose options with trade-offs, (4) WAIT for user decision.

---

## Rule 2: NEVER corrupt existing working state

**What**: Never `pip install` into system Python or existing working venvs — create a NEW venv. Never overwrite source data — create versioned copies. Never edit configs used by running processes. Never manually install on CloudLab/remote nodes — always use `setup.sh`.

**Why**: System pip installs break OS packages. Modifying running venvs causes cascading failures. Each incident wastes hours debugging.

**Test**: "Does this modify something that currently works?" If yes → COPY FIRST.

---

## Rule 3: ALWAYS verify code matches paper/design

**What**: At start of any session modifying scheduling or prediction code: re-read the relevant paper section, trace one request through the full code path, verify every equation has a matching code line, check function names match what they actually compute.

**Why**: March 27 audit found 5 critical silent bugs that had been wrong for weeks. All "worked" — just sub-optimally. Only caught by systematic line-by-line audit.

---

## Rule 4: ALWAYS report errors honestly

**What**: If a model fails to load, the system MUST fail — not silently degrade. If results look suspicious (identical values, degenerate distributions), investigate BEFORE reporting success.

**Why**: Silent degradation masks critical failures and produces false results.

---

## Rule 5: Use correct machine for correct task

| Machine | Use for | NOT for |
|---------|---------|---------|
| **VPS** (openclaw) | Code, writing, rsync, git | ANY GPU work |
| **GPU VM** (gxp-l40s-2) | Quick tests <10min | Long training (dept warned) |
| **CloudLab** | Online serving experiments | Training, offline work |

---

## Rule 6: Never modify remote cluster code directly

All code fixes must be made locally and deployed via `setup.sh` or `rsync`/`scp`. Never run `git checkout`, `git stash`, or `git pull` on cluster nodes. Remote code must stay consistent with what experiment scripts expect.

**Why**: `git checkout` on remote nodes reverted local fixes → 2+ hours debugging predictor failures.

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
- Always send warmup request before timed benchmark (first prediction ~5s for model load)
- Binary search capacity (don't waste runs on full QPS sweep)
- Reserve port 8200 for global scheduler only
- Smoke test 3-5 samples with EXACT config before full runs
