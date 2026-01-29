#!/bin/bash
# A100-40GB Cluster Setup Script
# For CloudLab d8545 nodes (4×A100-40GB SXM4 with NVLink)
#
# Prerequisites:
#   Generate config first:
#     python block/exp/generate_config.py --user_name asdwb \
#         --manifest_path block/a100_cl_manifest.xml \
#         --cluster_type a100 --tensor_parallel_size 4 --num_predictors 4
#
# Usage:
#   sh block/exp/setup_a100.sh [HOSTS_FILE]
#
# This script installs on ALL nodes in hosts file:
#   - CUDA 12.6
#   - PyTorch 2.6 with CUDA support
#   - vLLM (block branch) with tensor parallelism
#   - Block (a100-test branch)
#
# See: Block_paper/claude/A100_TESTING_GUIDE.md for full instructions

set -e

BLOCK_GITHUB_LINK="https://github.com/AKafakA/Block.git"
VLLM_GITHUB_LINK="https://github.com/AKafakA/vllm.git"
BLOCK_BRANCH="a100-test"
VLLM_BRANCH="block"

# Allow custom hosts file as argument
HOSTS_FILE="${1:-block/config/a100_hosts}"

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
    echo "      --manifest_path block/a100_cl_manifest.xml \\"
    echo "      --cluster_type a100 \\"
    echo "      --tensor_parallel_size 4 \\"
    echo "      --num_predictors 4"
    echo ""
    echo "Or specify a custom hosts file:"
    echo "  sh block/exp/setup_a100.sh /path/to/hosts"
    exit 1
fi

# Show hosts
echo "=============================================="
echo "A100-40GB Cluster Setup"
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
echo "A100-40GB Cluster Setup"
echo "=============================================="
echo "Hosts file: $HOSTS_FILE"
echo "Block branch: $BLOCK_BRANCH"
echo "vLLM branch: $VLLM_BRANCH"
echo ""

# Phase 1: System updates
echo "=== Phase 1: System updates ==="
parallel-ssh -t 0 -h $HOSTS_FILE "sudo apt update && sudo apt full-upgrade -y"
parallel-ssh -t 0 -h $HOSTS_FILE "sudo apt install -y python3-pip python3-venv ccache git"
parallel-ssh -t 0 -h $HOSTS_FILE "pip3 install -U pip==25.0.1"

# Phase 2: CUDA 12.6 installation
echo "=== Phase 2: CUDA 12.6 installation ==="
parallel-ssh -t 0 -h $HOSTS_FILE "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600"
parallel-ssh -t 0 -h $HOSTS_FILE "wget -q https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_amd64.deb && sudo dpkg -i cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_amd64.deb"
parallel-ssh -t 0 -h $HOSTS_FILE "sudo cp /var/cuda-repo-ubuntu2004-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/ && sudo apt-get update"
parallel-ssh -t 0 -h $HOSTS_FILE "sudo dpkg --configure -a && sudo apt-get -y install cuda-toolkit-12-6 && sudo apt-get install -y nvidia-open"
parallel-ssh -t 0 -h $HOSTS_FILE "echo 'export PATH=/usr/local/cuda-12.6/bin:\$PATH' >> ~/.bashrc && echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:\$LD_LIBRARY_PATH' >> ~/.bashrc"

# Phase 3: Verify GPUs (A100-40GB should show 4 GPUs)
echo "=== Phase 3: Verify GPUs ==="
parallel-ssh -t 0 -h $HOSTS_FILE "nvidia-smi --query-gpu=name,memory.total --format=csv"

# Phase 4: PyTorch and dependencies
echo "=== Phase 4: PyTorch and dependencies ==="
parallel-ssh -t 0 -h $HOSTS_FILE "pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126"
parallel-ssh -t 0 -h $HOSTS_FILE "pip install flashinfer-python==0.2.5 triton==3.2.0"

# Phase 5: Clone and install vLLM
echo "=== Phase 5: vLLM installation ==="
parallel-ssh -t 0 -h $HOSTS_FILE "rm -rf ~/vllm && git clone ${VLLM_GITHUB_LINK} ~/vllm"
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/vllm && git checkout ${VLLM_BRANCH}"
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/vllm && VLLM_USE_PRECOMPILED=1 pip install --editable ."

# Phase 6: Clone and install Block
echo "=== Phase 6: Block installation ==="
parallel-ssh -t 0 -h $HOSTS_FILE "rm -rf ~/Block && git clone ${BLOCK_GITHUB_LINK} ~/Block"
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/Block && git checkout ${BLOCK_BRANCH}"
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/Block && pip install -r requirements.txt"
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/Block && pip install -e ."

# Phase 7: SCP local configs to remote hosts (configs are gitignored for security)
echo "=== Phase 7: Copy local configs to remote hosts ==="
CONFIG_DIR="block/config"
while IFS= read -r host; do
    echo "Copying configs to $host..."
    # Copy A100-specific configs
    scp ${CONFIG_DIR}/a100_host_configs.json ${host}:~/Block/block/config/ 2>/dev/null || true
    scp ${CONFIG_DIR}/a100_hosts ${host}:~/Block/block/config/ 2>/dev/null || true
    scp ${CONFIG_DIR}/llama70b_a100_40gb_config.json ${host}:~/Block/block/config/ 2>/dev/null || true
    # Copy any other host configs that exist
    scp ${CONFIG_DIR}/host_configs.json ${host}:~/Block/block/config/ 2>/dev/null || true
    scp ${CONFIG_DIR}/hosts ${host}:~/Block/block/config/ 2>/dev/null || true
done < "$HOSTS_FILE"
echo "Config files copied to all hosts."

# Phase 8: Cleanup
echo "=== Phase 8: Cleanup ==="
parallel-ssh -t 0 -h $HOSTS_FILE "rm -f ~/cuda-repo-*.deb"

# Phase 9: Verify installation
echo "=== Phase 9: Verify installation ==="
parallel-ssh -t 0 -h $HOSTS_FILE "python -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}\")'"
parallel-ssh -t 0 -h $HOSTS_FILE "python -c 'import vllm; print(f\"vLLM imported successfully\")'"

echo ""
echo "=============================================="
echo "A100 Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Run Vidur profiling for A100-40GB:"
echo "     ssh to node0 and run:"
echo "     sh block/exp/end_to_end_exp_scripts/a100_40gb_profiling.sh"
echo ""
echo "  2. Download Llama-2-70B model (requires HuggingFace token)"
echo ""
echo "  3. Run experiment:"
echo "     sh block/exp/end_to_end_exp_scripts/a100_llama70b_exp.sh"
