#!/bin/bash
# Launch the 10× large-scale sweep (120 replicas, QPS 320).
# Usage: ./simulation_analysis/run_large_scale_suite.sh [extra arguments]

set -euo pipefail

PYTHONPATH=. python simulation_analysis/run_experiment_suite.py \
    --output-dir simulation_analysis/large_scale/remote \
    --analysis-file simulation_analysis/large_scale/remote/analysis_summary.json \
    --csv-file simulation_analysis/large_scale/remote/analysis_summary.csv \
    --num-replicas 120 \
    --qps 320 \
    --max-requests 0 \
    "$@"

