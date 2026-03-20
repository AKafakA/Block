#!/bin/bash
# Sync experiment results from A30 node0 to local
# Run periodically or before cluster release

TARGET="asdwb@d7525-10s10339.wisc.cloudlab.us"
LOCAL_DIR="/home/wd312/Code/llm/Block_exp/experiment_results_a30"

echo "[$(date '+%H:%M:%S')] Syncing A30 experiment data..."

rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no" \
    $TARGET:~/Block/experiment_output/ \
    $LOCAL_DIR/ \
    --exclude "logs/" \
    --exclude "running_logs/" \
    --exclude "*.log" \
    2>&1 | tail -5

echo "[$(date '+%H:%M:%S')] Sync complete. Data at: $LOCAL_DIR"
ls -la $LOCAL_DIR/
