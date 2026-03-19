#!/bin/bash
# Latency benchmark QPS sweep — sends load through coordinator,
# sidecars on each instance collect training data automatically.
# Designed for overnight runs (6-8 hours).
#
# Usage: bash block/exp/cara/sweep_latency_benchmark.sh
# Run from ~/Block on the coordinator node (d8545-10s10301).
set -e

export PYTHONPATH="$HOME/Block:$PYTHONPATH"
cd "$HOME/Block"

COORDINATOR="127.0.0.1:8200"
TRAIN_DATA="$HOME/Block/data/cara/training_data/cara_v3_all_training_train.jsonl"
DATA_DIR="$HOME/Block/data/cara/latency_data"
HOSTS_FILE="$HOME/Block/block/config/hosts"

# Cluster-level QPS (18 instances total)
QPS_LEVELS="9 12 15 18 21 24 27 30 33 36"
NUM_PROMPTS=20000  # per QPS level
NUM_SWEEPS=2       # repeat full sweep for more data

# Resume support: skip already-completed (sweep, qps) pairs
# Set via env vars: START_SWEEP=1 START_QPS=12 to resume from sweep1 qps12
START_SWEEP="${START_SWEEP:-1}"
START_QPS="${START_QPS:-0}"

echo "================================================================"
echo "  Latency Benchmark QPS Sweep"
echo "  Coordinator: $COORDINATOR"
echo "  QPS levels: $QPS_LEVELS"
echo "  Prompts per level: $NUM_PROMPTS"
echo "  Sweeps: $NUM_SWEEPS"
echo "  Start: $(date)"
echo "================================================================"

for sweep in $(seq 1 $NUM_SWEEPS); do
    echo ""
    echo "################################################################"
    echo "  SWEEP $sweep / $NUM_SWEEPS  ($(date))"
    echo "################################################################"

    for qps in $QPS_LEVELS; do
        # Resume support: skip completed pairs
        if [ "$sweep" -lt "$START_SWEEP" ]; then continue; fi
        if [ "$sweep" -eq "$START_SWEEP" ] && [ "$qps" -lt "$START_QPS" ]; then
            echo "  Skipping sweep $sweep QPS=$qps (already completed)"
            continue
        fi

        echo ""
        echo "================================================================"
        echo "  Sweep $sweep, QPS = $qps  ($(date))"
        echo "================================================================"

        # Clean instance-side data from previous QPS level
        while IFS= read -r host; do
            ssh -o ConnectTimeout=5 "$host" "rm -f ~/Block/training_data/cara/*.jsonl" 2>/dev/null &
        done < "$HOSTS_FILE"
        wait

        # Send load through coordinator (random scheduling)
        # Samples (input_len, output_len) from real training data distribution
        python -m block.predictor.cara.offline_training.generate_latency_benchmark \
            --host 127.0.0.1 --port 8200 \
            --model cara \
            --num-prompts "$NUM_PROMPTS" \
            --request-rate "$qps" \
            --real-data "$TRAIN_DATA" \
            --tokenizer "Qwen/Qwen2.5-3B" \
            --output "$DATA_DIR/sweep${sweep}_qps${qps}/benchmark_client.jsonl"

        # Flush all sidecar buffers
        while IFS= read -r host; do
            ssh -o ConnectTimeout=5 "$host" \
                "curl -s -X POST http://localhost:8300/flush" 2>/dev/null &
        done < "$HOSTS_FILE"
        wait
        sleep 2

        # Collect sidecar training data from all instances
        mkdir -p "$DATA_DIR/sweep${sweep}_qps${qps}"
        while IFS= read -r host; do
            scp -o ConnectTimeout=5 "${host}:~/Block/training_data/cara/*.jsonl" \
                "$DATA_DIR/sweep${sweep}_qps${qps}/" 2>/dev/null || true
        done < "$HOSTS_FILE"

        # Report
        n_records=$(cat "$DATA_DIR/sweep${sweep}_qps${qps}"/training_data_*.jsonl 2>/dev/null | wc -l || echo 0)
        echo "  Collected $n_records training records at sweep=$sweep QPS=$qps"
    done
done

# Merge all data
echo ""
echo "================================================================"
echo "  Merging all sweep/QPS data"
echo "================================================================"
mkdir -p "$DATA_DIR/all"
cat "$DATA_DIR"/sweep*_qps*/training_data_*.jsonl > "$DATA_DIR/all/latency_all.jsonl" 2>/dev/null
total=$(wc -l < "$DATA_DIR/all/latency_all.jsonl" 2>/dev/null || echo 0)
echo "Total records: $total"
echo "Done! $(date)"
