# Block Experiment Documentation — Index

## Quick Start
- **[../README.md](../README.md)** — Project overview, repo structure, how to run experiments
- **[DEPLOYMENT.md](DEPLOYMENT.md)** — Step-by-step A30 + A100 deployment guide
- **[EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)** — All experiment scripts, parameters, expected runtimes

## Paper & Results
- **`../Block_paper/`** (gitignored) — LaTeX source, figures, revision notes

## Reference
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — System architecture: predictor, scheduler, vLLM backend, Vidur simulator
- **[RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)** — Key numbers: capacity, latency, robustness, A100 comparison
- **[KNOWN_ISSUES.md](KNOWN_ISSUES.md)** — Installation order, critical fixes, operational lessons

## Data & Profiling
- **`../data/trace_data/`** (gitignored) — ShareGPT, BurstGPT, ArXiv-Summ traces
- **`../data/profiling/`** (gitignored) — A30 + A100 compute/network profiling CSVs
- **`../data/length_estimation/`** (gitignored) — RoBERTa training data

## Experiment Scripts (committed)
- **`../block/exp/end_to_end_exp_scripts/a30_main/`** — Main QPS sweep
- **`../block/exp/end_to_end_exp_scripts/ablation/`** — All ablation studies
- **`../block/exp/end_to_end_exp_scripts/a100_supplementary/`** — A100 Llumnix comparison
