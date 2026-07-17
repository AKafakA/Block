#!/bin/bash
# Verify 16/16 predictors on ALL 12 nodes. Exit 0 if all good, exit 1 if any node < 16.
# Used as a gate BEFORE running a benchmark or mid-run.

EXPECTED=16
FAILED_NODES=""
for host in $(cat block/config/hosts); do
  count=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$host" "ps aux | grep 'predictor/api_server' | grep -v grep | wc -l" 2>/dev/null)
  if [ -z "$count" ] || [ "$count" -lt "$EXPECTED" ]; then
    FAILED_NODES="$FAILED_NODES $host=$count"
  fi
done

if [ -n "$FAILED_NODES" ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] FAIL — predictors below 16:$FAILED_NODES"
  exit 1
fi
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] OK — all 12 nodes have 16/16 predictors"
exit 0
