# End-to-End Experiment Scripts

This directory contains automation scripts for running Block experiments, organized by experiment type and cluster.

## Directory Structure

```
end_to_end_exp_scripts/
├── a30_main/              # Main experiments on A30 cluster (12 nodes)
├── a100_supplementary/    # Supplementary experiments on A100 cluster (2 nodes)
├── ablation/              # Ablation studies
└── README.md
```

---

## A30 Main Experiments (`a30_main/`)

Main experiments for the Block paper, run on a 12-node A30 cluster.

| Script | Description |
|--------|-------------|
| `main_experiment.sh` | Main QPS sweep experiment (Block vs Llumnix-- vs baselines) |
| `warmup.sh` | Warmup script to prepare the cluster |

**Usage:**
```bash
cd a30_main
./warmup.sh
./main_experiment.sh
```

---

## A100 Supplementary Experiments (`a100_supplementary/`)

Supplementary experiments on a 2-node A100-40GB cluster, including Llumnix comparison and large model profiling.

### Llumnix Comparison Scripts

| Script | Description |
|--------|-------------|
| `deploy_block.sh` | Deploy Block system (vLLM + Predictors + Global Scheduler) |
| `deploy_llumnix.sh` | Deploy Llumnix 0.1.1 (Ray + Migration + FlashInfer) |
| `run_benchmark.sh` | Run standardized benchmarks for Block or Llumnix |
| `full_comparison.sh` | Orchestrate complete Block vs Llumnix comparison |

**Quick Start:**
```bash
cd a100_supplementary

# Full comparison (deploys both systems, runs all QPS levels)
./full_comparison.sh both

# Or run individually:
./deploy_block.sh start
./run_benchmark.sh block 28 10000
./deploy_block.sh stop

./deploy_llumnix.sh start
./run_benchmark.sh llumnix 28 10000
./deploy_llumnix.sh stop
```

### Large Model Profiling Scripts

| Script | Description |
|--------|-------------|
| `a100_40gb_profiling.sh` | Profile Llama-70B on A100-40GB |
| `a100_llama70b_exp.sh` | Run Llama-70B experiments |
| `run_a100_llama70b.sh` | Helper script for Llama-70B runs |

**Note:** These scripts are configured for the CloudLab A100 cluster. Update `NODE0_HOST`, `NODE1_HOST`, and IP addresses for different clusters.

---

## Ablation Studies (`ablation/`)

Various ablation experiments to evaluate individual components.

| Script | Description |
|--------|-------------|
| `po2_ablation_exp.sh` | Power-of-two choices (N=2 vs N=12) |
| `burstiness_exp.sh` | Burstiness sensitivity study |
| `error_heatmap_exp.sh` | Prediction error sensitivity heatmap |
| `cpu_tracking_experiment.sh` | CPU overhead tracking |
| `prediction_experiment.sh` | Prediction accuracy experiments |
| `block_nosim_ablation_exp.sh` | Block without simulation ablation |
| `auto_provision_exp.sh` | Auto-provisioning experiments |
| `config_search_experiment.sh` | Configuration search experiments |
| `extension_experiment.sh` | Extension experiments |
| `length_estimation.sh` | Length estimation experiments |

**Usage:**
```bash
cd ablation
./po2_ablation_exp.sh
./burstiness_exp.sh
# etc.
```

---

## Configuration

Most scripts have a **CLUSTER CONFIGURATION** section at the top that needs to be edited for your cluster:

```bash
# Example configuration variables
NODE0_HOST="user@hostname"      # SSH-accessible hostname
NODE0_INTERNAL_IP="10.0.0.1"    # Internal cluster IP
HF_TOKEN="hf_xxx"               # HuggingFace token
MODEL="meta-llama/Llama-2-7b-hf"
```

---

## Output Locations

| Experiment Type | Data Location | Plot Location |
|-----------------|---------------|---------------|
| Main experiment | `experiment_output/data/main/` | `experiment_output/results/main/` |
| Llumnix comparison | `experiment_output/data/SoCC_revision/llumnix_compare/` | `experiment_output/results/SoCC_revision/` |
| Po2 ablation | `experiment_output/data/SoCC_revision/po2/` | `experiment_output/results/SoCC_revision/` |
| Burstiness | `experiment_output/data/SoCC_revision/burstiness/` | `experiment_output/results/SoCC_revision/` |
| Prediction error | `experiment_output/data/SoCC_revision/prediction_error/` | `experiment_output/results/SoCC_revision/` |
| CPU overhead | `experiment_output/data/SoCC_revision/cpu_tracker/` | `experiment_output/results/SoCC_revision/` |

---

## Related Documentation

- `SoCC_revision/docs/LLUMNIX_COMPARISON_GUIDE.md` - Detailed Llumnix comparison guide
- `SoCC_revision/docs/EXPERIMENT_RESULTS.md` - Complete experimental results
- `experiments_analysis/` - Plot generation scripts