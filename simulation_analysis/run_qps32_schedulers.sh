#!/bin/bash
# Convenience wrapper to launch the QPS=32, 12-replica sweep once Block parity is confirmed.
# Usage: ./simulation_analysis/run_qps32_schedulers.sh <output_dir> <fast_predict>
#    fast_predict: on|off (determines whether Block variants use the optimized path)
# The script runs the selected schedulers sequentially and streams logs to the
# respective subdirectories under <output_dir>.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <output_dir> <fast_predict>" >&2
  exit 1
fi

OUTPUT_ROOT=$1
FAST_PREDICT=$2
shift 2

declare -a SCHEDULERS=(
  block_offline
  block_star_offline
  infass_pp
  llumnix_minus
  random
  round_robin
)

mkdir -p "${OUTPUT_ROOT}"

for SCHED in "${SCHEDULERS[@]}"; do
  LOG_DIR="${OUTPUT_ROOT}/${SCHED}"
  mkdir -p "${LOG_DIR}"
  LOG_FILE="${LOG_DIR}/run.log"
  echo "[run_qps32] Launching ${SCHED} (fast_predict=${FAST_PREDICT})" | tee "${LOG_FILE}"
PARALLEL_ARGS=("--block-parallel-enable" "off")
if [[ "${FAST_PREDICT}" == "on" ]]; then
  PARALLEL_ARGS=("--block-parallel-enable" "on" "--deterministic-noise" "on")
else
  PARALLEL_ARGS=("--block-parallel-enable" "off" "--deterministic-noise" "off")
fi

PYTHONPATH=. nohup python scripts/run_offline_simulations.py \
    --schedulers "${SCHED}" \
    --num-replicas 12 \
    --trace-file data/trace_data/sharegpt/sharegpt_val_10k_llama2.csv \
    --predicted-trace-file data/trace_data/sharegpt/generate/llama/sharegpt-llama-7b-val-10k-predicted.json \
    --qps 32 \
    --max-requests 0 \
    --block-noise 10 \
    --fast-predict "${FAST_PREDICT}" \
    "${PARALLEL_ARGS[@]}" \
    --output-dir "${OUTPUT_ROOT}" \
    >> "${LOG_FILE}" 2>&1 &
  echo "[run_qps32] PID $!" | tee -a "${LOG_FILE}"
  wait $!
done
