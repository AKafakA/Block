#!/bin/bash
# Convenience wrapper to launch the full 12×10k QPS=32 sweep on a remote host.
# Usage: ./simulation_analysis/run_qps32_suite.sh [additional run_experiment_suite.py args]

set -euo pipefail

PYTHONPATH=. python3 simulation_analysis/run_experiment_suite.py \
    --output-dir simulation_analysis/full_runs/qps32_remote \
    --analysis-file simulation_analysis/full_runs/qps32_remote/analysis_summary.json \
    --csv-file simulation_analysis/full_runs/qps32_remote/analysis_summary.csv \
    --num-replicas 12 \
    --qps 32 \
    --max-requests 0 \
    "$@"
