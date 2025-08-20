# Plot main results
TTFT_SLO=3
#!/bin/bash
export PYTHONPATH=.
python3 experiments_analysis/experiment_plot.py \
    --experiments-dir experiment_output/data/main/sharegpt \
    --output-dir experiment_output/results/main \
    --ttft-p99-slo $TTFT_SLO \
    --max-qps 36 \
    --min-qps 20 \
    --num-of-cdf-figures 5 \
    --zoom-for-slo \
    --show-slo-text \


python3 experiments_analysis/prediction_plot.py \
    --experiments-dir experiment_output/data/prediction/sharegpt \
    --output-dir experiment_output/results/prediction \

python3 experiments_analysis/experiment_plot.py \
    --experiments-dir experiment_output/data/extension/burstgpt \
    --output-dir experiment_output/results/burstgpt \
    --ttft-p99-slo $TTFT_SLO \
    --max-qps 64 \
    --min-qps 48 \
    --num-of-cdf-figures 5 \

python3 experiments_analysis/experiment_plot.py \
    --experiments-dir experiment_output/data/extension/sharegpt \
    --output-dir experiment_output/results/qwen \
    --ttft-p99-slo $TTFT_SLO \
    --max-qps 70 \
    --min-qps 55 \
    --num-of-cdf-figures 5 \

python3 experiments_analysis/experiment_plot.py \
    --experiments-dir experiment_output/data/config_search/batch/sharegpt \
    --output-dir experiment_output/results/batch_24 \
    --ttft-p99-slo $TTFT_SLO \
    --max-qps 36 \
    --min-qps 20 \
    --num-of-cdf-figures 5 \

python3 experiments_analysis/experiment_plot.py \
    --experiments-dir experiment_output/data/config_search/chunkSize/sharegpt \
    --output-dir experiment_output/results/chunkSize \
    --ttft-p99-slo $TTFT_SLO \
    --max-qps 36 \
    --min-qps 20 \
    --num-of-cdf-figures 5 \

python3 experiments_analysis/auto_provision_plot.py \
    --experiments-dir experiment_output/data/auto_provision/sharegpt \
    --output-dir experiment_output/results/auto_provision
