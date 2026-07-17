#!/bin/bash
# Identify which result directories were produced while any A30 node had <16 predictors.
# Inputs:
#   $1 = remote result path (e.g., ~/Block/experiment_output/main_po2_gated/sharegpt/min_new_request_latency)
#   $2 = first host to query (usually first line of block/config/hosts)
#   $3 = path to predictor health audit log (default: /tmp/a30_pred_health_audit.log)
# Output:
#   Prints one line per corrupted run: "scheduler qps est_len"
#   Based on directory mtime (UTC) being within ±60s of any <16 audit entry.
#
# Audit log format (each line): "<UTC_TS> <node_short_id> <predictor_count>"

REMOTE_PATH="$1"
HOST="$2"
AUDIT_LOG="${3:-/tmp/a30_pred_health_audit.log}"

if [ -z "$REMOTE_PATH" ] || [ -z "$HOST" ]; then
  echo "usage: $0 <remote_result_path> <host> [audit_log]" >&2
  exit 1
fi

# Collect all UTC timestamps where count <16 (any node) from audit log
BAD_TS=$(awk '$3<16 && $3!="" {print $1}' "$AUDIT_LOG" 2>/dev/null | sort -u)

if [ -z "$BAD_TS" ]; then
  echo "# No <16/16 events detected in $AUDIT_LOG — no corrupted runs" >&2
  exit 0
fi

# Convert each bad TS to epoch seconds
BAD_EPOCHS=""
for ts in $BAD_TS; do
  epoch=$(date -u -d "$ts" +%s 2>/dev/null)
  [ -n "$epoch" ] && BAD_EPOCHS="$BAD_EPOCHS $epoch"
done

# List remote result dirs with their mtime (epoch seconds)
ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$HOST" \
  "find $REMOTE_PATH -maxdepth 1 -type d -name 'qps_*' -printf '%T@ %f\n' 2>/dev/null" |
while read mt_epoch dir; do
  mt_epoch_int=${mt_epoch%.*}
  # Directory lifespan ~= from mtime backwards ~10 min. We approximate: a run spans [mt-600, mt]
  RUN_START=$((mt_epoch_int - 600))
  RUN_END=$mt_epoch_int
  for be in $BAD_EPOCHS; do
    if [ "$be" -ge "$RUN_START" ] && [ "$be" -le "$RUN_END" ]; then
      # Extract QPS and est from dir name
      qps=$(echo "$dir" | grep -oP 'qps_\K\d+')
      est=$(echo "$dir" | grep -oP 'len_estimated_\Ktrue|len_estimated_\Kfalse')
      echo "CORRUPTED qps=$qps est=$est dir=$dir"
      break
    fi
  done
done
