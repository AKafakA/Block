#!/bin/bash
# util_predictor_health.sh — verify all 12 nodes have 16/16 predictors listening.
# Run during long sweeps to catch silent crashes early.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== predictor_health $ts ==="
all_ok=1
for host in $(cat block/config/hosts); do
    short=$(echo "$host" | grep -oP 'd7525-10s\K\d+')
    n=$(ssh -n -o ConnectTimeout=3 "$host" \
        "ss -lnt | grep -E '0\.0\.0\.0:(8100|8300|8400|8500|8600|8700|8800|8900|9000|9100|9200|9300|9400|9500|9600|9700) ' | wc -l" 2>/dev/null)
    if [ "$n" != "16" ]; then
        echo "ALERT: $short = $n/16"
        all_ok=0
    else
        echo "OK    $short = 16/16"
    fi
done

[ "$all_ok" = "1" ] && echo "=== all 192 predictors healthy ===" || { echo "=== FAILED: predictor crash detected ==="; exit 1; }
