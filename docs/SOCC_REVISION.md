# SoCC Revision Status

## Paper: "Block: Balancing Load in LLM Serving with Context, Knowledge and Predictive Scheduling"

**Status:** SoCC 2026 accepted; revising for camera-ready.

## Added Sections (SoCC revision)

| Section | Content | Status | Data Source |
|---------|---------|--------|-------------|
| §6.7 | Llumnix comparison (A100, Block vs Llumnix sched-only + migration) | Done | `SoCC_revision/llumnix_compare/` |
| §6.8 | Po2 ablation (N=2 achieves 97.7% capacity, 54% overhead reduction) | Done | `SoCC_revision/po2/` |
| §6.9 | Burstiness study (Gamma α=0.25-2.0, ±1% robust) | Done | `SoCC_revision/burstiness/` |
| §6.10 | Error sensitivity heatmap (100% noise, <3% degradation) | Done | `SoCC_revision/prediction_error/` |
| §6.11 | CPU overhead analysis (21-32% per predictor, ~4 cores for 16) | Done | `SoCC_revision/cpu_tracker/` |

## Key Numbers for Camera-Ready

- **Capacity gain:** 1.3% at optimal config (31.1 vs 30.7 QPS); 13.8% at bs=24
- **Po2 scalability:** 97.7% capacity with N=2, 54% scheduling overhead reduction
- **Error robustness:** <3% degradation at 100% Gaussian noise on both predictions
- **Burstiness:** Within ±1% across γ=0.25 (very bursty) to γ=2.0 (regular)
- **A100 Llumnix parity:** Block matches Llumnix scheduling-only; Llumnix migration collapses at QPS=36

## Remaining Camera-Ready Items

- [ ] Standardize "Llumnix--" vs "Llumnix (sched-only)" terminology
- [ ] Reconcile Block* (oracle) vs Block (predicted) gap discussion
- [ ] Add statistical significance (3+ runs with different seeds)
- [ ] Clarify abstract: which variant (Block vs Block*), which config (bs=48)
- [ ] Optional: trim §2-3 background for space

## SoCC_revision/ Directory Structure

```
SoCC_revision/
├── burstiness/          4 scenarios × 2 schedulers (Block + RR)
├── prediction_error/    20+ scenarios (length × latency error grid)
├── llumnix_compare/     Block vs Llumnix (sched-only + migration)
├── cpu_tracker/         CPU overhead profiling data
├── po2/                 N=2 vs N=12 comparison
├── a100_llama70b_results/   A100 supplementary
├── cpu_overhead.png/pdf     Generated figure
└── plot_cpu_overhead.py     Plotting script
```
