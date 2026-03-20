#!/bin/bash
NODE0="NODE0_PLACEHOLDER"
HF_TOKEN=""

ssh_cmd() { ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no "$@"; }

kill_scheduler() {
    ssh_cmd $NODE0 'fuser -k 8200/tcp 2>/dev/null; true'
}

run_qps() {
    local n=$1 qps=$2 output=$3
    echo "=== N=$n QPS=$qps ==="
    kill_scheduler
    sleep 5
    ssh_cmd $NODE0 "cd ~/Block && export PYTHONPATH=. && (nohup python block/global_scheduler/api_server.py --config_path block/config/a100_8x7b_host_configs.json --metrics_type min_new_request_latency --num_query_predictor $n --num_required_predictor $n --workers 1 --num_predictor_ports 4 --profiling_sampling_rate 0.0 --predictor_timeout 2000 --backend_timeout 3600 --initial_available_instance 8 --max_slo_in_seconds 0 > experiment_output/logs/global_scheduler.log 2>&1 &)"
    sleep 10
    ssh_cmd $NODE0 "cd ~/Block && export PYTHONPATH=. && export HF_TOKEN=$HF_TOKEN && python block/benchmark/benchmark_serving.py --backend block --ip_ports 127.0.0.1:8200 --tokenizer meta-llama/Llama-2-7b-hf --trust_remote_code --dataset_type sharegpt --dataset_path ~/Block/data/trace_data/sharegpt/generate/llama --qps $qps --num_sampled_requests 10000 --max_request_len 4096 --timeout_in_seconds 1800 --output_dir $output/sharegpt/min_new_request_latency/qps_${qps} --log_filename benchmark.log --use_estimated_response_lens"
    sleep 10
}

echo "=== Config 2: Po2 N=2 no-chunked ==="
for qps in 16 20 24 28 32 36; do
    run_qps 2 $qps "a100_po2_no_chunked"
done

echo "=== Config 3: Broadcasting N=12 no-chunked ==="
for qps in 16 20 24 28 32 36; do
    run_qps 12 $qps "a100_broadcast_no_chunked"
done

echo "=== ALL DONE ==="
