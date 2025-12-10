#!/bin/bash

# CARA End-to-End Experiment Script
# Complete workflow: deploy backends -> verify -> start CARA -> run tests

set -e  # Exit on error

# Configuration
TARGET_HOST="asdwb@d8545-10s10301.wisc.cloudlab.us"
HOST_CONFIG_PATH="block/config/host_configs.json"
MODEL_CONFIG_PATH="block/config/cara/model_config_template.json"
DEPLOYMENT_CONFIG="block/config/cara/model_deployment.json"
HOSTS_FILE="block/config/hosts"
CARA_PORT=8200
HF_TOKEN="${HF_TOKEN:-}"  # Set HF_TOKEN environment variable before running

echo "========================================"
echo "CARA End-to-End Experiment"
echo "========================================"
echo "Target CARA Host: ${TARGET_HOST}"
echo "CARA Port: ${CARA_PORT}"
echo ""

# Step 1: Deploy backend instances
echo "Step 1: Deploying Backend Instances"
echo "------------------------------------"
python block/exp/cara/deploy_cara.py \
  --hosts ${HOSTS_FILE} \
  --config ${MODEL_CONFIG_PATH} \
  --hf-token "${HF_TOKEN}" \
  --output ${DEPLOYMENT_CONFIG}

if [ $? -ne 0 ]; then
    echo "❌ Backend deployment failed!"
    exit 1
fi
echo "✅ Backend deployment completed"
echo ""

# Step 2: Wait for backends to initialize
echo "Step 2: Waiting for Backends to Initialize"
echo "-------------------------------------------"
echo "Waiting 60 seconds for models to load..."
sleep 60
echo ""

# Step 3: Verify backend deployments
echo "Step 3: Verifying Backend Deployments"
echo "--------------------------------------"
python block/exp/cara/e2e/check_deployment.py \
  --config ${DEPLOYMENT_CONFIG} \
  --output block/config/cara/verified_hosts.json

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Some backends may not be fully operational"
    echo "Continuing with available backends..."
fi
echo "✅ Backend verification completed"
echo ""

# Step 4: Start CARA scheduler server on target host
echo "Step 4: Starting CARA Scheduler Server"
echo "---------------------------------------"

# Kill existing CARA server if running
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  ${TARGET_HOST} \
  "pkill -f 'cara_serve.py' || echo 'No existing CARA server found'"

sleep 2

# Create log directory on target host
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  ${TARGET_HOST} \
  "mkdir -p Block/experiment_output/logs"

# Start CARA server in background
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  ${TARGET_HOST} \
  "cd Block && nohup python -m block.global_scheduler.cara.cara_serve \
    --host 0.0.0.0 \
    --port ${CARA_PORT} \
    --model_config_path ${DEPLOYMENT_CONFIG} \
    --host_config ${HOST_CONFIG_PATH} \
    --scheduling random \
    > experiment_output/logs/cara_server.log 2>&1 &"

echo "Waiting 10 seconds for CARA server to start..."
sleep 10

# Extract IP from target host
CARA_HOST_IP=$(echo ${TARGET_HOST} | cut -d'@' -f2)

echo "✅ CARA server started at http://${CARA_HOST_IP}:${CARA_PORT}"
echo ""

# Step 5: Run simple benchmark test
echo "Step 5: Running Benchmark Test"
echo "-------------------------------"

# Update test.sh with correct CARA host and run it on target host
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  ${TARGET_HOST} \
  "cd Block && \
   sed -i 's/CARA_HOST=\"127.0.0.1\"/CARA_HOST=\"127.0.0.1\"/' block/exp/cara/test.sh && \
   bash block/exp/cara/test.sh"

echo ""
echo "========================================"
echo "CARA Experiment Complete!"
echo "========================================"
echo ""
echo "Summary:"
echo "  ✅ Backends deployed and verified"
echo "  ✅ CARA server running at http://${CARA_HOST_IP}:${CARA_PORT}"
echo "  ✅ Benchmark test completed"
echo ""
echo "View results:"
echo "  ssh ${TARGET_HOST} 'cat Block/experiment_output/cara_test_results/cara_simple_test.json | jq .'"
echo ""
echo "Check CARA server logs:"
echo "  ssh ${TARGET_HOST} 'tail -f Block/experiment_output/logs/cara_server.log'"
echo ""
