# Known Issues & Operational Lessons

## Installation Order (CRITICAL)

1. vLLM FIRST (`VLLM_USE_PRECOMPILED=1 pip install --editable .`)
2. PyTorch AFTER (`torch==2.6.0+cu126`)
3. transformers pinned (`==4.50.3`, NOT 5.x)
4. Set `VLLM_USE_V1=0` and `HF_TOKEN` BEFORE starting ANY vLLM process

Violating this order causes hard-to-debug symbol errors. See DEPLOYMENT.md for details.

## Critical Do's

- Always use `setup.sh` for cluster deployment (never manual pip install)
- Deploy predictors in batches (8 at a time, 10s pause) to avoid OOM
- Use `generate_config.py` to extract CloudLab internal IPs from manifest XML
- Copy predictor sklearn cache between nodes (saves ~5 min per node)
- Set `profiling_sampling_rate=0.0` during experiments (0.1 adds ~20% overhead)
- Send a warmup request before timed benchmarks (first prediction ~5s for model load)

## Critical Don'ts

- Never install PyTorch before vLLM (ABI mismatch → undefined symbol)
- Never use `transformers>=5.0` (breaks vLLM block branch tokenizer API)
- Never manually modify remote code during deployment (git checkout/stash → state drift)
- Never use `pkill -f python` on cluster nodes (kills SSH sessions too)
- Never deploy 64+ predictors simultaneously (OOM kills the entire node)
- Never use `profiling_sampling_rate > 0.0` during experiments

## Known Bugs (Fixed)

| Date | Bug | Fix | Impact |
|------|-----|-----|--------|
| Mar 19 | A100 deployment fails | Wrong vLLM branch; need `block` not `main` | 4h wasted |
| Mar 19 | Predictor port collision | Reserve 8200 for scheduler; predictors on 8100+ | Config fix |
| Mar 19 | nohup SSH hang | Wrap in subshell: `(nohup python ... &)` | Script fix |
| Mar 20 | profiling_sampling_rate slowdown | Set to 0.0 for experiments | 20% throughput recovered |
| Mar 20 | Length predictor not loaded | Pre-tag dataset with RoBERTa predictions | Pipeline fix |

## Performance Tips

- Binary search capacity (don't run full QPS sweep — waste of cluster time)
- Use `TRANSFORMERS_OFFLINE=1 && HF_HUB_OFFLINE=1` when DNS fails but model is cached
- Increase SLO from 3s to 5s to show Po2 tail-latency advantage more clearly
- For A100: get `num_blocks` from `/simple_schedule_trace` after vLLM starts (varies by GPU memory)
- Cache-aware predictor deployment: 1 predictor trains sklearn models, copy cache, deploy rest
