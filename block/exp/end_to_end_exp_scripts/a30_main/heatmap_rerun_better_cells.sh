#!/bin/bash
# Rerun the 4 "better than baseline" heatmap cells with FRESH vLLM+predictor deploy per cell.
# Cells: (25,25)=-2.1%, (25,50)=-1.1%, (50,50)=-2.2%, (100,100)=-3.2%
# Goal: isolate single-deploy artifact vs real noise. Output to *_RERUN dirs.

set -u

TARGET_HOST="$(head -1 block/config/hosts)"
BATCH_CAP=48
CHUNK_SIZE=512
PREDICTOR_WORKERS=16
GLOBAL_SCHEDULER_WORKERS=1
BACKEND_WORKERS=1
MAX_MODEL_LENGTH=4096
TIMEOUT_IN_SECONDS=1800
PREDICTOR_TIMEOUT_IN_SECONDS=1000
BRANCH_NAME="main"
MODEL="meta-llama/Llama-2-7b-hf"
DATASET_PATH="~/Block/data/trace_data/sharegpt/generate/llama"
SCHEDULER_NAME="min_new_request_latency"
N_SELECTED=2
QPS=32
NUM_REQUEST=10000
PROFILING_SAMPLE_RATE=0.000
AVAILABLE_INSTANCE="12"
MAX_SLO="0"
HOST_CONFIG_PATH='block/config/host_configs.json'
PRED_CONFIG_PATH='block/config/llama_config.json'
RESULTS_FILE=/tmp/a30_phase4_2_simple_results.txt

# 5 cells with negative e2e_P99 (better than baseline): len_err:lat_err
# (25,25)=-1.5% (25,50)=-0.7% (50,50)=-1.8% (100,0)=-0.9% (100,100)=-3.0%
CELLS="100:100 25:25 25:50 50:50 100:0"

deploy_fresh() {
    sh block/exp/reset.sh
    sleep 30
    nohup sh block/exp/run_exp_vllm.sh $BATCH_CAP $MODEL false 0 $MAX_MODEL_LENGTH true $BACKEND_WORKERS $CHUNK_SIZE > /dev/null 2>&1 &
    sleep 90
    for s in $(seq 1 7); do
        nohup sh block/exp/run_exp_predictor_${s}.sh $PRED_CONFIG_PATH $SCHEDULER_NAME true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
    done
    sleep 10
    for s in $(seq 8 $PREDICTOR_WORKERS); do
        nohup sh block/exp/run_exp_predictor_${s}.sh $PRED_CONFIG_PATH $SCHEDULER_NAME true $BATCH_CAP true $PREDICTOR_WORKERS $BRANCH_NAME 0 $PREDICTOR_TIMEOUT_IN_SECONDS > /dev/null 2>&1 &
    done
    sleep 60
    bash block/exp/end_to_end_exp_scripts/a30_main/verify_predictors.sh "$PRED_CONFIG_PATH" "$SCHEDULER_NAME" "$BATCH_CAP" "$PREDICTOR_WORKERS" "$PREDICTOR_TIMEOUT_IN_SECONDS"
}

echo "=== Heatmap better-cells rerun (5 cells with fresh deploy each) starting $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"

for cell in $CELLS; do
    le=$(echo $cell | cut -d: -f1)
    la=$(echo $cell | cut -d: -f2)
    echo "[rerun] cell ($le, $la) — fresh deploy + benchmark" | tee -a "$RESULTS_FILE"
    deploy_fresh
    ssh -n -o ConnectTimeout=5 "$TARGET_HOST" "pkill -f global_scheduler/api_server" 2>&1 || true
    sleep 5
    nohup sh block/exp/run_exp_global_scheduler.sh "$TARGET_HOST" $N_SELECTED $N_SELECTED $SCHEDULER_NAME $HOST_CONFIG_PATH $GLOBAL_SCHEDULER_WORKERS $PREDICTOR_WORKERS $PROFILING_SAMPLE_RATE $TIMEOUT_IN_SECONDS $PREDICTOR_TIMEOUT_IN_SECONDS $AVAILABLE_INSTANCE $MAX_SLO false $le $la > /dev/null 2>&1 &
    sleep 20
    OUTPUT_DIR="error_heatmap_po2_rerun/sharegpt/${SCHEDULER_NAME}/qps_${QPS}_len_err_${le}_lat_err_${la}_RERUN"
    parallel-ssh -i -t 0 --host "$TARGET_HOST" "cd Block && export PYTHONPATH=. && export HF_TOKEN= && python block/benchmark/benchmark_serving.py --ip_ports 127.0.0.1:8200 --tokenizer $MODEL --num_sampled_requests $NUM_REQUEST --dataset_type sharegpt --dataset_path $DATASET_PATH --qps $QPS --backend block --log_filename benchmark.log --output_dir $OUTPUT_DIR --data_start_index 0 --trust_remote_code --max_request_len $MAX_MODEL_LENGTH --timeout_in_seconds $TIMEOUT_IN_SECONDS --use_estimated_response_lens" > /dev/null 2>&1
    sleep 5
    npz=/tmp/p42_heatmap_rerun_${le}_${la}.npz
    rsync -az "asdwb@d7525-10s10327.wisc.cloudlab.us:~/Block/experiment_output/$OUTPUT_DIR/benchmark_all_metrics.npz" "$npz" > /dev/null 2>&1
    python3 -c "import numpy as np; d=np.load('$npz', allow_pickle=True); req=d['request_latencies']; ttft=d['prefill_token_latencies']; print(f'[($le,$la) RERUN] e2e_mean={float(np.mean(req)):.0f}, e2e_P99={float(np.percentile(req,99)):.0f}, TTFT_P99={float(np.percentile(ttft,99)):.0f}')" | tee -a "$RESULTS_FILE"
done

echo "=== Heatmap better-cells rerun complete at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$RESULTS_FILE"
echo ""
echo "Comparison vs original:" | tee -a "$RESULTS_FILE"
echo "  Original: (25,25)=14422, (25,50)=14574, (50,50)=14406, (100,0)=14900, (100,100)=14264 — baseline 14737" | tee -a "$RESULTS_FILE"
echo "  Original e2e_P99 deltas: (25,25)=-1.5% (25,50)=-0.7% (50,50)=-1.8% (100,0)=-0.9% (100,100)=-3.0%" | tee -a "$RESULTS_FILE"
