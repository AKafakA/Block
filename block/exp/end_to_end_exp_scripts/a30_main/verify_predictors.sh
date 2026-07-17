#!/bin/bash
# Helper: verify all 12 nodes have 16 predictor processes. Auto-fixes any missing.
# Args: $1 = config_path, $2 = metric_type, $3 = batch_cap, $4 = workers (always 16), $5 = predictor_timeout
# Returns: 0 if all 192 predictors confirmed alive (after fixes), 1 if any remained missing after 3 retries.

set -u

CFG="${1:?missing config_path}"
METRIC="${2:?missing metric_type}"
BATCH="${3:?missing batch_cap}"
WORKERS="${4:-16}"
TIMEOUT="${5:-1000}"

# Port table: predictor_index -> port (matches run_exp_predictor_N.sh hardcoded ports).
# Verified from grep of all run_exp_predictor_*.sh: port 8200 is skipped, predictor_1 has no --port (uses 8100 default).
declare -A PORT=(
    [1]=8100  [2]=8300  [3]=8400  [4]=8500  [5]=8600  [6]=8700  [7]=8800  [8]=8900
    [9]=9000  [10]=9100 [11]=9200 [12]=9300 [13]=9400 [14]=9500 [15]=9600 [16]=9700
)

echo "[verify_predictors] starting $(date -u +%H:%M:%SZ) metric=$METRIC batch=$BATCH"

MAX_ATTEMPTS=3
attempt=0

while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    missing_any=0
    while IFS= read -r host; do
        [ -z "$host" ] && continue
        # Get running indices on this host
        running=$(ssh -n -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$host" "ps aux | grep 'predictor/api_server' | grep -v grep | grep -oP 'predictor_index \K\d+' | sort -nu | tr '\n' ' '" 2>/dev/null)
        short=$(echo "$host" | grep -oP 'd7525-10s\K\d+')
        # Find missing indices
        missing=""
        for idx in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
            if ! echo " $running " | grep -q " $idx "; then
                missing="$missing $idx"
            fi
        done
        if [ -n "$missing" ]; then
            missing_any=1
            echo "[verify_predictors] attempt $attempt: host $short missing indices:$missing — starting"
            for idx in $missing; do
                port="${PORT[$idx]}"
                # FIX: explicit </dev/null + ServerAliveInterval prevents nohup-from-ssh hang.
                # Wrap with timeout 30s as belt-and-suspenders so a stuck ssh can't block the script forever.
                timeout 30 ssh -n -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "$host" "cd Block && export PYTHONPATH=. && nohup python block/predictor/api_server.py --config_path $CFG --metric_type $METRIC --enable_time_estimation true --batch_size_cap $BATCH --workers $WORKERS --enable_chunked_prefill --threshold_batch_size_for_time_estimation 0 --port $port --predictor_timeout $TIMEOUT --predictor_index $idx > experiment_output/logs/predictor_${idx}.log 2>&1 < /dev/null &" &
            done
        fi
    done < block/config/hosts
    # FIX: bound wait so a single hung ssh can never block the script forever (60s cap)
    end_time=$(($(date +%s) + 60))
    while [ -n "$(jobs -p)" ] && [ "$(date +%s)" -lt "$end_time" ]; do
        sleep 2
    done
    if [ -n "$(jobs -p)" ]; then
        echo "[verify_predictors] WARNING: 60s wait elapsed, killing $(jobs -p | wc -l) pending ssh"
        kill $(jobs -p) 2>/dev/null || true
    fi
    wait 2>/dev/null || true

    if [ "$missing_any" = "0" ]; then
        echo "[verify_predictors] all 192 predictors confirmed after $attempt attempt(s)"
        return 0 2>/dev/null || exit 0
    fi

    # Give freshly-started predictors time to come up before re-checking
    sleep 20
done

echo "[verify_predictors] WARNING: some predictors still missing after $MAX_ATTEMPTS attempts"
return 1 2>/dev/null || exit 1
