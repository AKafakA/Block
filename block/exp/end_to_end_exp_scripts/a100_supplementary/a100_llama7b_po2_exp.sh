#!/bin/bash
# A100 Block Po2 Experiment for Llumnix Comparison
#
# Runs Block with Po2 (N=2) on A100 cluster for comparison with
# existing Llumnix results (Sched-Only + Migration).
#
# Two configurations:
#   1. Block Po2 + chunked prefill (512) — compare with Llumnix Sched-Only
#   2. Block Po2 - chunked prefill — compare with Llumnix Migration
#
# QPS: 16, 20, 24, 28, 32, 36 (6 levels x 2 configs = 12 runs)
#
# Usage: ./a100_llama7b_po2_exp.sh [chunked|no_chunked|both]

set +e  # Don't exit on errors — handle them explicitly

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cluster config — update these for your A100 reservation
NODE0_HOST="NODE1_PLACEHOLDER"
NODE1_HOST="NODE0_PLACEHOLDER"
NODE0_INTERNAL_IP="10.10.1.1"
NODE1_INTERNAL_IP="10.10.1.2"

HF_TOKEN=""
HF_HOME="/mydata/huggingface"

MODEL="meta-llama/Llama-2-7b-hf"
MAX_NUM_SEQS=96
MAX_MODEL_LEN=4096

BLOCK_HOST_CONFIG="block/config/a100_8x7b_host_configs.json"
PREDICTOR_CONFIG="block/config/llama_config.json"

NUM_INSTANCES=8
NUM_PREDICTORS_PER_INSTANCE=4
PROFILING_SAMPLE_RATE=0.1
PREDICTOR_TIMEOUT=2000
BACKEND_TIMEOUT=3600
METRICS_TYPE="min_new_request_latency"
N_SELECTED=2

DATASET_PATH="~/Block/data/trace_data/sharegpt/generate/llama"
DATASET_TYPE="sharegpt"
NUM_REQUESTS=10000
QPS_VALUES="16 20 24 28 32 36"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

ssh_cmd() {
    local host=$1; shift
    ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no "$host" "$@"
}

wait_for_service() {
    local host=$1 port=$2 service_name=$3 max_wait=${4:-300} waited=0
    log "Waiting for $service_name on $host:$port..."
    while [ $waited -lt $max_wait ]; do
        if ssh_cmd "$host" "curl -s http://127.0.0.1:$port/health 2>/dev/null" | grep -qE "OK|healthy|running"; then
            log "$service_name is ready on port $port"; return 0
        fi
        sleep 10; waited=$((waited + 10))
    done
    log "ERROR: $service_name on port $port did not become ready in $max_wait seconds"; return 1
}

stop_all() {
    log "Stopping all Block services..."
    for host in "$NODE0_HOST" "$NODE1_HOST"; do
        ssh_cmd "$host" "pkill -f 'vllm.entrypoints.api_server' 2>/dev/null || true"
        ssh_cmd "$host" "pkill -f 'predictor/api_server' 2>/dev/null || true"
        ssh_cmd "$host" "pkill -f 'global_scheduler/api_server' 2>/dev/null || true"
    done
    sleep 5; log "All services stopped"
}

deploy_vllm() {
    local enable_chunked=$1
    local chunked_flag=""
    if [ "$enable_chunked" = "true" ]; then
        chunked_flag="--enable-chunked-prefill --max-num-batched-tokens 512"
        log "Deploying vLLM WITH chunked prefill"
    else
        log "Deploying vLLM WITHOUT chunked prefill"
    fi

    for host in "$NODE0_HOST" "$NODE1_HOST"; do
        log "Starting 4 vLLM instances on $host..."
        ssh_cmd "$host" "cd ~/Block && mkdir -p experiment_output/logs"
        ssh_cmd "$host" "cd ~/Block && \
            export HF_HOME=$HF_HOME && export HF_TOKEN=$HF_TOKEN && \
            export VLLM_USE_V1=0 && export PYTHONPATH=. && \
            for gpu in 0 1 2 3; do \
                port=\$((8000 + gpu)); \
                CUDA_VISIBLE_DEVICES=\$gpu nohup python -m vllm.entrypoints.api_server \
                    --model $MODEL --port \$port \
                    --max-num-seqs $MAX_NUM_SEQS --max-model-len $MAX_MODEL_LEN \
                    $chunked_flag \
                    > experiment_output/logs/vllm_gpu\${gpu}.log 2>&1 & \
            done"
    done

    log "Waiting for vLLM instances..."
    for port in 8000 8001 8002 8003; do
        wait_for_service "$NODE0_HOST" "$port" "vLLM (node0:$port)" 300 || return 1
        wait_for_service "$NODE1_HOST" "$port" "vLLM (node1:$port)" 300 || return 1
    done
    log "All 8 vLLM instances ready"
}

deploy_predictors() {
    log "Deploying predictors (32 total)..."

    # Build cache on node0
    ssh_cmd "$NODE0_HOST" "cd ~/Block && export PYTHONPATH=. && \
        nohup python block/predictor/api_server.py \
            --config_path $PREDICTOR_CONFIG --metric_type min_new_request_latency \
            --enable_time_estimation true --batch_size_cap $MAX_NUM_SEQS \
            --workers 1 --enable_chunked_prefill \
            --threshold_batch_size_for_time_estimation 0 \
            --predictor_timeout $PREDICTOR_TIMEOUT --predictor_index 0 --port 8100 \
            > experiment_output/logs/predictor_cache_build.log 2>&1 &"
    sleep 30
    for i in {1..24}; do
        if ssh_cmd "$NODE0_HOST" "grep -q 'Uvicorn running' ~/Block/experiment_output/logs/predictor_cache_build.log 2>/dev/null"; then
            log "Predictor cache built"; break
        fi; sleep 5
    done
    ssh_cmd "$NODE0_HOST" "pkill -f 'predictor/api_server' 2>/dev/null || true"; sleep 3
    ssh_cmd "$NODE0_HOST" "scp -r ~/Block/cache/ ${NODE1_INTERNAL_IP}:~/Block/ 2>/dev/null || true"

    for host in "$NODE0_HOST" "$NODE1_HOST"; do
        log "Starting 16 predictors on $host..."
        ssh_cmd "$host" "cd ~/Block && export PYTHONPATH=. && \
            for gpu in 0 1 2 3; do \
                base_port=\$((8100 + gpu * 100)); \
                for p in 0 1 2 3; do \
                    pred_port=\$((base_port + p)); \
                    nohup python block/predictor/api_server.py \
                        --config_path $PREDICTOR_CONFIG --metric_type min_new_request_latency \
                        --enable_time_estimation true --batch_size_cap $MAX_NUM_SEQS \
                        --workers 1 --enable_chunked_prefill \
                        --threshold_batch_size_for_time_estimation 0 \
                        --predictor_timeout $PREDICTOR_TIMEOUT \
                        --predictor_index \$((gpu * 4 + p)) --port \$pred_port \
                        > experiment_output/logs/predictor_gpu\${gpu}_\${p}.log 2>&1 & \
                done; \
            done"
    done
    sleep 20; log "All 32 predictors deployed"
}

deploy_scheduler() {
    log "Deploying global scheduler (N=$N_SELECTED)..."
    ssh_cmd "$NODE0_HOST" "cd ~/Block && export PYTHONPATH=. && \
        nohup python block/global_scheduler/api_server.py \
            --config_path $BLOCK_HOST_CONFIG --metrics_type $METRICS_TYPE \
            --num_query_predictor $N_SELECTED --num_required_predictor 1 \
            --workers 1 --num_predictor_ports $NUM_PREDICTORS_PER_INSTANCE \
            --profiling_sampling_rate $PROFILING_SAMPLE_RATE \
            --predictor_timeout $PREDICTOR_TIMEOUT --backend_timeout $BACKEND_TIMEOUT \
            --initial_available_instance $NUM_INSTANCES --max_slo_in_seconds 0 \
            > experiment_output/logs/global_scheduler.log 2>&1 &"
    sleep 10; log "Global scheduler deployed (N=$N_SELECTED)"
}

run_benchmark() {
    local qps=$1 output_dir=$2
    log "Benchmark: QPS=$qps → $output_dir"
    ssh_cmd "$NODE0_HOST" "cd ~/Block && export PYTHONPATH=. && export HF_HOME=$HF_HOME && \
        mkdir -p $output_dir && \
        python -m block.benchmark.benchmark_serving \
            --backend block --ip_ports ${NODE0_INTERNAL_IP}:8200 \
            --tokenizer $MODEL --trust_remote_code \
            --dataset_type $DATASET_TYPE --dataset_path $DATASET_PATH \
            --qps $qps --num_sampled_requests $NUM_REQUESTS \
            --max_request_len $MAX_MODEL_LEN --timeout_in_seconds $BACKEND_TIMEOUT \
            --output_dir $output_dir --log_filename block_po2_qps${qps}_logs.txt \
            2>&1 | tee /tmp/benchmark_po2_qps${qps}.log"
    sleep 15
}

run_sweep() {
    local config_name=$1 output_base=$2
    log "Starting QPS sweep for $config_name"
    for qps in $QPS_VALUES; do
        run_benchmark "$qps" "experiment_output/benchmark_output/${output_base}/qps${qps}"
    done
    log "QPS sweep complete for $config_name"
}

run_chunked() {
    log "CONFIG 1: Block Po2 + Chunked Prefill (N=2)"
    stop_all; deploy_vllm "true"; deploy_predictors; deploy_scheduler
    run_sweep "Block Po2 + chunked prefill" "block_po2_chunked"
    stop_all
}

run_no_chunked() {
    log "CONFIG 2: Block Po2 - Chunked Prefill (N=2)"
    stop_all; deploy_vllm "false"; deploy_predictors; deploy_scheduler
    run_sweep "Block Po2 - chunked prefill" "block_po2_no_chunked"
    stop_all
}

main() {
    local command=${1:-both}
    log "A100 Block Po2 Experiment (N=$N_SELECTED)"

    case $command in
        chunked) run_chunked ;;
        no_chunked) run_no_chunked ;;
        both) run_chunked; run_no_chunked ;;
        *) echo "Usage: $0 [chunked|no_chunked|both]"; exit 1 ;;
    esac

    log "A100 Block Po2 experiment complete!"
    log "Results: experiment_output/benchmark_output/block_po2_{chunked,no_chunked}/"
}

main "$@"
