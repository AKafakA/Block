# Block_exp — Project Rules

## What This Repo Is
Block experiment repository for the SoCC 2026 camera-ready revision. Contains the Block predictive scheduler code, Vidur simulator, experiment scripts, and results. The Cara (NeurIPS) extension lives separately at `~/Code/llm/Block`.

## Critical Rules

### R1: Installation order is sacred
vLLM FIRST (precompiled) -> PyTorch -> transformers==4.50.3. See docs/DEPLOYMENT.md.

### R2: Always use setup.sh for CloudLab
Never manual installs on cluster nodes. If setup.sh fails, report — don't improvise.

### R3: No silent experiment parameter changes
Batch size, chunk size, max_model_len, num_query_predictor — these are experiment parameters. STOP and ask before changing.

### R4: Smoke test before long runs
Send a warmup request to the scheduler before starting timed benchmarks. First prediction takes ~5s for model load.

### R5: Set VLLM_USE_V1=0
The V1 engine lacks the schedule_trace API that Block depends on. Always export this before any vLLM process.

### R6: Deploy predictors in batches
Max 8 concurrent predictor startups. Sleep 10s between batches. Concurrent model loading causes OOM.

### R7: profiling_sampling_rate=0.0 during experiments
Non-zero sampling rate adds ~20% throughput overhead. Only use 0.1 for debugging.

## Key Paths
- Experiment scripts: `block/exp/end_to_end_exp_scripts/`
- Configs: `block/config/` (llama_config.json, host_configs.json)
- Results: `experiment_results_a30/`, `experiment_results_a100/` (gitignored)
- Paper: `Block_paper/` (gitignored)
- Profiling data: `data/profiling/` (gitignored)
- Docs: `docs/INDEX.md` (start here)

## Hardware
- **A30 cluster**: 12× CloudLab d7525, Llama-2-7B, 16 predictors/node
- **A100 cluster**: 2× CloudLab d8545 (4× A100-40GB SXM4), Llama-2-70B TP=4
- **vLLM branch**: https://github.com/AKafakA/vllm/tree/block
