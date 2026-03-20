#!/bin/bash
# A100 End-to-End Experiment Script
# Deploys vLLM, predictors, scheduler and runs benchmarks.
# Run from the local machine (needs SSH access + parallel-ssh to both A100 nodes).
#
# Configs:
#   1. Block Po2 (N=2) + chunked prefill
#   2. Block Po2 (N=2) - chunked prefill
#   3. Block broadcasting (N=12) - chunked prefill (reuses vLLM+predictors from config 2)
#
# Usage: bash a100_e2e_experiment.sh [1|2|3|all]
#
# IMPORTANT:
#   - Port 8200 is reserved for global scheduler — predictor ports must skip it
#   - Predictor ports: GPU0=8100-8103, GPU1=8300-8303, GPU2=8400-8403, GPU3=8500-8503
#   - 4 predictors per GPU instance (matching deploy_block.sh NUM_PREDICTORS_PER_INSTANCE=4)
#   - profiling_sampling_rate MUST be 0.0 for benchmarking (non-zero corrupts results)
#   - Build 1 predictor cache first, copy to node1, then deploy remaining
#   - Use fuser -k PORT/tcp to kill by port (pkill -f matches SSH command and kills session)

# === CLUSTER CONFIG ===
NODE0="NODE0_PLACEHOLDER"
NODE1="NODE1_PLACEHOLDER"
NODES="$NODE0 $NODE1"
NODE0_IP="10.10.1.1"

MODEL="meta-llama/Llama-2-7b-hf"
MAX_SEQS=96
MAX_LEN=4096
PREDICTOR_CONFIG="block/config/llama7b_a100_config.json"
HOST_CONFIG="block/config/a100_8x7b_host_configs.json"
HF_TOKEN=""
QPS_VALUES="16 20 24 28 32 36"
NUM_REQUESTS=10000

# Predictor port mapping (skip 8200 — used by global scheduler)
# GPU0: 8100-8103, GPU1: 8300-8303, GPU2: 8400-8403, GPU3: 8500-8503
GPU_PORTS="0:8000:8100 1:8001:8300 2:8002:8400 3:8003:8500"

ssh_cmd() {
    ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no "$@"
}

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

# === KILL ALL ===
kill_all() {
    log "Killing all processes..."
    for host in $NODES; do
        ssh_cmd $host 'for pid in $(pgrep -u $(whoami) python); do kill -9 $pid 2>/dev/null; done; true' &
    done
    wait
    sleep 5
    log "All killed"
}

# === KILL SCHEDULER ONLY (by port, not by name) ===
kill_scheduler() {
    ssh_cmd $NODE0 'fuser -k 8200/tcp 2>/dev/null; true'
    sleep 3
}

# === DEPLOY VLLM ===
deploy_vllm() {
    local chunked_flag="$1"
    log "Deploying vLLM (chunked=$([[ -n $chunked_flag ]] && echo yes || echo no))..."
    for host in $NODES; do
        ssh_cmd $host "cd ~/vllm && mkdir -p ~/Block/experiment_output/logs && \
            export HF_TOKEN=$HF_TOKEN VLLM_USE_V1=0 && \
            for gpu in 0 1 2 3; do \
                port=\$((8000 + gpu)); \
                (CUDA_VISIBLE_DEVICES=\$gpu nohup python -m vllm.entrypoints.api_server \
                    --model $MODEL --port \$port \
                    --max-num-seqs $MAX_SEQS --max-model-len $MAX_LEN \
                    $chunked_flag --swap-space 0 \
                    > ~/Block/experiment_output/logs/vllm_gpu\${gpu}.log 2>&1 &); \
            done && echo 'vLLM launched'" &
    done
    wait

    log "Waiting for model load (up to 10 min)..."
    for i in $(seq 1 60); do
        sleep 10
        up=0
        for host in $NODES; do
            host_up=$(ssh_cmd $host 'u=0; for p in 8000 8001 8002 8003; do curl -s --connect-timeout 2 http://127.0.0.1:$p/health >/dev/null 2>&1 && u=$((u+1)); done; echo $u')
            up=$((up + host_up))
        done
        log "  $up/8 vLLM instances UP"
        [ "$up" -eq 8 ] && break
    done

    if [ "$up" -ne 8 ]; then
        log "ERROR: Only $up/8 vLLM instances started"
        return 1
    fi

    # Verify schedule_trace endpoint
    result=$(ssh_cmd $NODE0 'curl -s --connect-timeout 3 http://127.0.0.1:8000/schedule_trace 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d))" 2>/dev/null')
    if [ -z "$result" ] || [ "$result" = "0" ]; then
        log "ERROR: schedule_trace endpoint not working"
        return 1
    fi
    log "vLLM ready, schedule_trace verified"
}

# === DEPLOY PREDICTORS ===
deploy_predictors() {
    local metric_type="$1"
    local enable_chunked="${2:-true}"
    local chunked_flag=""
    if [ "$enable_chunked" = "true" ]; then
        chunked_flag="--enable_chunked_prefill"
    fi
    log "Deploying predictors (metric=$metric_type, chunked=$enable_chunked)..."

    # Step 1: Build cache on EACH node in parallel (1 predictor each)
    log "Building predictor cache on both nodes in parallel..."
    for host in $NODES; do
        ssh_cmd $host "cd ~/Block && export PYTHONPATH=. && \
            python block/predictor/api_server.py \
                --config_path $PREDICTOR_CONFIG --metric_type $metric_type \
                --enable_time_estimation true --batch_size_cap $MAX_SEQS \
                --workers 1 $chunked_flag \
                --threshold_batch_size_for_time_estimation 0 \
                --predictor_timeout 2000 --predictor_index 0 --port 8100 \
                --instance-port 8000 \
                > experiment_output/logs/predictor_cache_build.log 2>&1 &" &
    done
    wait

    # Wait for both to finish
    for i in $(seq 1 60); do
        sleep 5
        ready=0
        for host in $NODES; do
            if ssh_cmd $host 'grep -q "Uvicorn running" ~/Block/experiment_output/logs/predictor_cache_build.log 2>/dev/null'; then
                ready=$((ready + 1))
            fi
        done
        [ "$ready" -eq 2 ] && log "Cache built on both nodes" && break
    done

    # Kill cache builders (by port)
    for host in $NODES; do
        ssh_cmd $host 'fuser -k 8100/tcp 2>/dev/null; true' &
    done
    wait
    sleep 3

    # Step 3: Deploy 4 predictors per GPU (16 per node, 32 total)
    # Port mapping: GPU0=8100-8103, GPU1=8300-8303, GPU2=8400-8403, GPU3=8500-8503
    for host in $NODES; do
        ssh_cmd $host "cd ~/Block && export PYTHONPATH=. && \
            for gpu_base in '0 8000 8100' '1 8001 8300' '2 8002 8400' '3 8003 8500'; do \
                set -- \$gpu_base; gpu=\$1; vllm_port=\$2; base=\$3; \
                for p in 0 1 2 3; do \
                    pred_port=\$((base + p)); \
                    (nohup python block/predictor/api_server.py \
                        --config_path $PREDICTOR_CONFIG --metric_type $metric_type \
                        --enable_time_estimation true --batch_size_cap $MAX_SEQS \
                        --workers 1 $chunked_flag \
                        --threshold_batch_size_for_time_estimation 0 \
                        --predictor_timeout 2000 \
                        --predictor_index \$((gpu * 4 + p)) \
                        --port \$pred_port \
                        --instance-port \$vllm_port \
                        > experiment_output/logs/predictor_gpu\${gpu}_\${p}.log 2>&1 &); \
                done; \
            done && echo 'launched'" &
    done
    wait

    log "Waiting 60s for predictors to start from cache..."
    sleep 60

    # Verify
    total_up=0
    for host in $NODES; do
        up=$(ssh_cmd $host 'u=0; for base in 8100 8300 8400 8500; do for p in 0 1 2 3; do port=$((base+p)); curl -s --connect-timeout 1 http://127.0.0.1:$port/health >/dev/null 2>&1 && u=$((u+1)); done; done; echo $u')
        log "  $host: $up/16 predictors UP"
        total_up=$((total_up + up))
    done
    log "Total: $total_up/32 predictors"
    if [ "$total_up" -ne 32 ]; then
        log "WARNING: Not all predictors started"
    fi
}

# === RUN BENCHMARK SWEEP ===
run_sweep() {
    local n_selected="$1"
    local output_prefix="$2"
    log "=== QPS Sweep: N=$n_selected, output=$output_prefix ==="

    for qps in $QPS_VALUES; do
        log "--- QPS=$qps ---"

        # Kill old scheduler by port
        kill_scheduler

        # Start new scheduler
        ssh_cmd $NODE0 "cd ~/Block && export PYTHONPATH=. && \
            (nohup python block/global_scheduler/api_server.py \
                --config_path $HOST_CONFIG --metrics_type min_new_request_latency \
                --num_query_predictor $n_selected --num_required_predictor $n_selected \
                --workers 1 --num_predictor_ports 4 \
                --profiling_sampling_rate 0.0 \
                --predictor_timeout 2000 --backend_timeout 3600 \
                --initial_available_instance 8 --max_slo_in_seconds 0 \
                > experiment_output/logs/global_scheduler.log 2>&1 &)"
        sleep 10

        # Run benchmark
        ssh_cmd $NODE0 "cd ~/Block && export PYTHONPATH=. && export HF_TOKEN=$HF_TOKEN && \
            python block/benchmark/benchmark_serving.py \
                --backend block --ip_ports 127.0.0.1:8200 \
                --tokenizer $MODEL --trust_remote_code \
                --dataset_type sharegpt \
                --dataset_path ~/Block/data/trace_data/sharegpt/generate/llama \
                --qps $qps --num_sampled_requests $NUM_REQUESTS \
                --max_request_len $MAX_LEN --timeout_in_seconds 1800 \
                --output_dir $output_prefix/sharegpt/min_new_request_latency/qps_${qps} \
                --log_filename benchmark.log \
                --use_estimated_response_lens"
        sleep 10

        # Check result
        result=$(ssh_cmd $NODE0 "grep 'successful_responses' ~/Block/experiment_output/$output_prefix/sharegpt/min_new_request_latency/qps_${qps}/benchmark_logs.txt 2>/dev/null")
        log "  Result: $result"
    done
}

# === CONFIG 1: Block Po2 + chunked prefill ===
run_config1() {
    log "============================================"
    log "CONFIG 1: Block Po2 (N=2) + Chunked Prefill"
    log "============================================"
    kill_all
    deploy_vllm "--enable-chunked-prefill --max-num-batched-tokens 512"
    deploy_predictors "min_new_request_latency" "true"
    run_sweep 2 "a100_po2_chunked"
}

# === CONFIG 2: Block Po2 - chunked prefill ===
run_config2() {
    log "============================================"
    log "CONFIG 2: Block Po2 (N=2) - Chunked Prefill"
    log "============================================"
    kill_all
    deploy_vllm ""
    deploy_predictors "min_new_request_latency" "false"
    run_sweep 2 "a100_po2_no_chunked"
}

# === CONFIG 3: Block broadcasting - chunked prefill ===
# Reuses vLLM + predictors from config 2 (same no-chunked setup)
run_config3() {
    log "============================================"
    log "CONFIG 3: Block Broadcasting (N=12) - Chunked Prefill"
    log "  (reusing vLLM + predictors from config 2)"
    log "============================================"
    run_sweep 12 "a100_broadcast_no_chunked"
}

# === MAIN ===
config=${1:-all}
log "A100 E2E Experiment — config=$config"

case $config in
    1) run_config1 ;;
    2) run_config2 ;;
    3) run_config3 ;;
    all) run_config2; run_config3; run_config1 ;;
    *) echo "Usage: $0 [1|2|3|all]"; exit 1 ;;
esac

log "============================================"
log "A100 experiment complete!"
log "Results in: experiment_output/a100_po2_chunked/"
log "            experiment_output/a100_po2_no_chunked/"
log "            experiment_output/a100_broadcast_no_chunked/"
log "============================================"
