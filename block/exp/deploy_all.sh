#!/bin/bash
# Deploy vLLM + all 16 predictors reliably.
# Deploys predictors in 2 batches (1-8, then 9-16) with verification.
# Usage: sh block/exp/deploy_all.sh [predictor_config] [metric_type]

PREDICTOR_CONFIG=${1:-block/config/llama_config.json}
METRIC_TYPE=${2:-min_new_request_latency}
BATCH_CAP=${3:-48}
MODEL=${4:-meta-llama/Llama-2-7b-hf}
PREDICTOR_TIMEOUT=${5:-1000}

echo "=== Deploy All: config=$PREDICTOR_CONFIG metric=$METRIC_TYPE ==="

# Step 1: Reset
echo "Step 1: Reset..."
sh block/exp/reset.sh
sleep 10

# Step 2: Deploy vLLM
echo "Step 2: Deploy vLLM..."
sh block/exp/run_exp_vllm.sh $BATCH_CAP $MODEL false 0 4096 true 1 512
echo "Waiting 90s for model load..."
sleep 90

# Verify vLLM
vllm_up=$(parallel-ssh -t 10 -h block/config/hosts "curl -s --connect-timeout 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo UP || echo DOWN" 2>&1 | grep -c "UP")
echo "vLLM: $vllm_up/12 nodes UP"
if [ "$vllm_up" -lt 12 ]; then
  echo "ERROR: Not all vLLM instances started. Aborting."
  exit 1
fi

# Step 3: Deploy predictors in 2 batches
echo "Step 3: Deploy predictors batch 1 (1-8)..."
for suffix in $(seq 1 8); do
  nohup sh block/exp/run_exp_predictor_${suffix}.sh $PREDICTOR_CONFIG $METRIC_TYPE true $BATCH_CAP true 16 main 0 $PREDICTOR_TIMEOUT > /dev/null 2>&1 &
done
echo "Batch 1 launched, waiting 60s..."
sleep 60

echo "Step 3b: Deploy predictors batch 2 (9-16)..."
for suffix in $(seq 9 16); do
  nohup sh block/exp/run_exp_predictor_${suffix}.sh $PREDICTOR_CONFIG $METRIC_TYPE true $BATCH_CAP true 16 main 0 $PREDICTOR_TIMEOUT > /dev/null 2>&1 &
done
echo "Batch 2 launched, waiting 60s..."
sleep 60

# Step 4: Verify all predictors on all nodes
echo "Step 4: Verify predictors..."
all_good=true
for host in $(cat block/config/hosts); do
  up=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $host "up=0; for port in 8100 8300 8400 8500 8600 8700 8800 8900 9000 9100 9200 9300 9400 9500 9600 9700; do curl -s --connect-timeout 1 http://127.0.0.1:\$port/health >/dev/null 2>&1 && up=\$((up+1)); done; echo \$up" 2>/dev/null)
  if [ "$up" != "16" ]; then
    echo "  $host: $up/16 — fixing missing predictors..."
    all_good=false
    # Find and restart missing predictors on this specific node
    for portidx in "1 8100" "2 8300" "3 8400" "4 8500" "5 8600" "6 8700" "7 8800" "8 8900" "9 9000" "10 9100" "11 9200" "12 9300" "13 9400" "14 9500" "15 9600" "16 9700"; do
      set -- $portidx; idx=$1; port=$2
      ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $host "curl -s --connect-timeout 1 http://127.0.0.1:$port/health >/dev/null 2>&1 || (cd ~/Block && export PYTHONPATH=. && nohup python block/predictor/api_server.py --config_path $PREDICTOR_CONFIG --metric_type $METRIC_TYPE --enable_time_estimation true --batch_size_cap $BATCH_CAP --workers 16 --enable_chunked_prefill --threshold_batch_size_for_time_estimation 0 --predictor_timeout $PREDICTOR_TIMEOUT --predictor_index $idx --port $port > experiment_output/logs/predictor_fix_${port}.log 2>&1 &)" 2>/dev/null &
    done
  else
    echo "  $host: 16/16 OK"
  fi
done
wait

if [ "$all_good" = "false" ]; then
  echo "Waiting 30s for fixes..."
  sleep 30
  # Final verify
  echo "Final verify:"
  for host in $(cat block/config/hosts); do
    up=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $host "up=0; for port in 8100 8300 8400 8500 8600 8700 8800 8900 9000 9100 9200 9300 9400 9500 9600 9700; do curl -s --connect-timeout 1 http://127.0.0.1:\$port/health >/dev/null 2>&1 && up=\$((up+1)); done; echo \$up" 2>/dev/null)
    echo "  $host: $up/16"
  done
fi

echo ""
echo "=== Deploy complete ==="
