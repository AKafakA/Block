#!/bin/bash
# A30 Cluster Setup Script
# For CloudLab c6525-25g nodes (1×A30 per node)
#
# Prerequisites:
#   Generate config first:
#     python block/exp/generate_config.py --user_name YOUR_USERNAME \
#         --manifest_path block/cl_manifest.xml \
#         --cluster_type a30 --tensor_parallel_size 1 --num_predictors 16
#
# Usage:
#   sh block/exp/setup.sh [HOSTS_FILE]
#
# This script installs on ALL nodes in hosts file:
#   - CUDA 12.6
#   - vLLM (block branch) with precompiled wheels
#   - PyTorch 2.6 with CUDA support (installed AFTER vLLM)
#   - Block (block branch)
#
# IMPORTANT: Installation order matters!
#   1. vLLM with VLLM_USE_PRECOMPILED=1 (downloads compatible .so files)
#   2. PyTorch 2.6.0+cu126 (after vLLM, not before)

set -e

BLOCK_GITHUB_LINK="https://github.com/AKafakA/Block.git"
VLLM_GITHUB_LINK="https://github.com/AKafakA/vllm.git"
BLOCK_BRANCH="test-a-30"
VLLM_BRANCH="block"

# Allow custom hosts file as argument
HOSTS_FILE="${1:-block/config/hosts}"

# Check if hosts file exists
if [ ! -f "$HOSTS_FILE" ]; then
    echo "=============================================="
    echo "ERROR: Hosts file not found: $HOSTS_FILE"
    echo "=============================================="
    echo ""
    echo "Please run generate_config.py first:"
    echo ""
    echo "  python block/exp/generate_config.py \\"
    echo "      --user_name YOUR_USERNAME \\"
    echo "      --manifest_path block/cl_manifest.xml \\"
    echo "      --cluster_type a30 \\"
    echo "      --tensor_parallel_size 1 \\"
    echo "      --num_predictors 16"
    echo ""
    echo "Or specify a custom hosts file:"
    echo "  sh block/exp/setup.sh /path/to/hosts"
    exit 1
fi

# Show hosts
echo "=============================================="
echo "A30 Cluster Setup"
echo "=============================================="
echo ""
echo "Hosts file: $HOSTS_FILE"
echo "Target nodes:"
cat "$HOSTS_FILE"
echo ""
echo "Block branch: $BLOCK_BRANCH"
echo "vLLM branch: $VLLM_BRANCH"
echo ""
echo "Press Enter to continue or Ctrl+C to abort..."
read

echo "=============================================="
echo "A30 Cluster Setup"
echo "=============================================="
echo "Hosts file: $HOSTS_FILE"
echo "Block branch: $BLOCK_BRANCH"
echo "vLLM branch: $VLLM_BRANCH"
echo ""

# Phase 1: System updates and build dependencies
echo "=== Phase 1: System updates ==="
parallel-ssh -t 0 -h $HOSTS_FILE "sudo apt update && sudo apt full-upgrade -y"
parallel-ssh -t 0 -h $HOSTS_FILE "sudo apt install -y python3-pip python3-venv ccache"
parallel-ssh -t 0 -h $HOSTS_FILE "pip3 install -U pip==25.0.1"

# Phase 2: CUDA 12.6 installation
echo "=== Phase 2: CUDA 12.6 installation ==="
parallel-ssh -t 0 -h $HOSTS_FILE "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600"
parallel-ssh -t 0 -h $HOSTS_FILE "wget -q https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_amd64.deb && sudo dpkg -i cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_amd64.deb"
parallel-ssh -t 0 -h $HOSTS_FILE "sudo cp /var/cuda-repo-ubuntu2004-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/ && sudo apt-get update"
parallel-ssh -t 0 -h $HOSTS_FILE "sudo dpkg --configure -a && sudo apt-get -y install cuda-toolkit-12-6 && sudo apt-get install -y nvidia-open"
parallel-ssh -t 0 -h $HOSTS_FILE "echo 'export PATH=/usr/local/cuda-12.6/bin:\$PATH' >> ~/.bashrc && echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:\$LD_LIBRARY_PATH' >> ~/.bashrc"

# Phase 3: Disable MIG mode (A30 specific - required for vLLM)
echo "=== Phase 3: Disable MIG mode ==="
parallel-ssh -t 0 -h $HOSTS_FILE "sudo nvidia-smi -mig 0 || true"

# Phase 4: Verify GPUs
echo "=== Phase 4: Verify GPUs ==="
parallel-ssh -t 0 -h $HOSTS_FILE "nvidia-smi --query-gpu=name,memory.total --format=csv"

# Phase 5: Clone and install vLLM with precompiled wheels
# IMPORTANT: vLLM must be installed BEFORE PyTorch 2.6.0+cu126
# The precompiled wheels include compatible .so files
echo "=== Phase 5: vLLM installation (precompiled, ~3 min) ==="
parallel-ssh -t 0 -h $HOSTS_FILE "rm -rf ~/vllm && git clone ${VLLM_GITHUB_LINK} ~/vllm"
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/vllm && git checkout ${VLLM_BRANCH}"
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/vllm && sudo VLLM_USE_PRECOMPILED=1 pip install --editable ."

# Add PYTHONPATH to bashrc for subprocess compatibility
parallel-ssh -t 0 -h $HOSTS_FILE "grep -q 'PYTHONPATH.*vllm' ~/.bashrc || echo 'export PYTHONPATH=\$HOME/vllm:\$PYTHONPATH' >> ~/.bashrc"

# Phase 6: Clone and install Block
echo "=== Phase 6: Block installation ==="
parallel-ssh -t 0 -h $HOSTS_FILE "rm -rf ~/Block && git clone ${BLOCK_GITHUB_LINK} ~/Block"
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/Block && git checkout ${BLOCK_BRANCH}"
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/Block && pip install -r requirements.txt"

# Phase 7: PyTorch and dependencies (MUST be after vLLM)
# Installing PyTorch after vLLM ensures ABI compatibility
echo "=== Phase 7: PyTorch and dependencies (after vLLM) ==="
parallel-ssh -t 0 -h $HOSTS_FILE "pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126"
parallel-ssh -t 0 -h $HOSTS_FILE "pip install flashinfer-python==0.2.5 triton==3.2.0"
# Fix transformers version - vLLM block branch requires 4.50.3, not 5.0+
parallel-ssh -t 0 -h $HOSTS_FILE "pip install transformers==4.50.3"

# Phase 8: Copy local configs to remote hosts
echo "=== Phase 8: Copy local configs to remote hosts ==="
CONFIG_DIR="block/config"
while IFS= read -r host; do
    echo "Copying configs to $host..."
    scp ${CONFIG_DIR}/host_configs.json ${host}:~/Block/block/config/ 2>/dev/null || true
    scp ${CONFIG_DIR}/hosts ${host}:~/Block/block/config/ 2>/dev/null || true
    scp ${CONFIG_DIR}/llama_config.json ${host}:~/Block/block/config/ 2>/dev/null || true
    scp ${CONFIG_DIR}/qwen_config.json ${host}:~/Block/block/config/ 2>/dev/null || true
done < "$HOSTS_FILE"
echo "Config files copied to all hosts."

# Phase 9: Create directories and cleanup
echo "=== Phase 9: Create directories and cleanup ==="
parallel-ssh -t 0 -h $HOSTS_FILE "mkdir -p ~/Block/experiment_output/logs ~/Block/cache"
parallel-ssh -t 0 -h $HOSTS_FILE "rm -f ~/cuda-repo-*.deb"

# Phase 10: Verify installation
echo "=== Phase 10: Verify installation ==="
parallel-ssh -t 0 -h $HOSTS_FILE "python -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}\")'"
parallel-ssh -t 0 -h $HOSTS_FILE "python -c 'from vllm.engine.async_llm_engine import AsyncLLMEngine; print(\"vLLM OK\")'"

echo ""
echo "=============================================="
echo "A30 Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Set HuggingFace token (if using gated models like Llama-2):"
echo "     export HF_TOKEN=your_token_here"
echo ""
echo "  2. Run experiments:"
echo "     sh block/exp/end_to_end_exp_scripts/main_experiments.sh"
echo ""
echo "Known issues fixed in this script:"
echo "  - VLLM_USE_V1=0 required for Block's get_scheduler_trace API"
echo "  - transformers==4.50.3 required (not 5.0+)"
echo "  - VLLM_USE_PRECOMPILED=1 for fast installation (~3 min vs 30 min)"
echo "  - PyTorch installed AFTER vLLM (order matters for ABI compatibility)"
echo "  - MIG mode disabled (required for vLLM on A30)"
echo ""
echo "IMPORTANT: Use internal network (10.x.x.x) for inter-node traffic."
echo "  Run generate_config.py to create configs with internal IPs."
echo "  See: https://docs.cloudlab.us/control-net.html"
