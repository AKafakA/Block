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
#   - PyTorch 2.6 with CUDA support
#   - vLLM (block branch)
#   - Block (block branch)
#
# See: Block_paper/claude/A100_TESTING_GUIDE.md for reference

set -e

BLOCK_GITHUB_LINK="https://github.com/AKafakA/Block.git"
VLLM_GITHUB_LINK="https://github.com/AKafakA/vllm.git"
BLOCK_BRANCH="block"
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
parallel-ssh -t 0 -h $HOSTS_FILE "sudo apt install -y python3-pip python3-venv python3-dev ccache git build-essential cmake ninja-build"
parallel-ssh -t 0 -h $HOSTS_FILE "pip3 install -U pip==25.0.1 setuptools wheel"

# Phase 2: CUDA 12.6 installation
echo "=== Phase 2: CUDA 12.6 installation ==="
parallel-ssh -t 0 -h $HOSTS_FILE "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600"
parallel-ssh -t 0 -h $HOSTS_FILE "wget -q https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_amd64.deb && sudo dpkg -i cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_amd64.deb"
parallel-ssh -t 0 -h $HOSTS_FILE "sudo cp /var/cuda-repo-ubuntu2004-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/ && sudo apt-get update"
parallel-ssh -t 0 -h $HOSTS_FILE "sudo dpkg --configure -a && sudo apt-get -y install cuda-toolkit-12-6 && sudo apt-get install -y nvidia-open"
parallel-ssh -t 0 -h $HOSTS_FILE "echo 'export PATH=/usr/local/cuda-12.6/bin:\$PATH' >> ~/.bashrc && echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:\$LD_LIBRARY_PATH' >> ~/.bashrc"

# Phase 3: Verify GPUs and disable MIG mode (A30 specific)
echo "=== Phase 3: Verify GPUs ==="
parallel-ssh -t 0 -h $HOSTS_FILE "nvidia-smi --query-gpu=name,memory.total --format=csv"
# Disable MIG mode on A30 (required for vLLM)
parallel-ssh -t 0 -h $HOSTS_FILE "sudo nvidia-smi -mig 0 || true"

# Phase 4: PyTorch and dependencies (MUST be before vLLM)
echo "=== Phase 4: PyTorch and dependencies ==="
parallel-ssh -t 0 -h $HOSTS_FILE "pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126"
parallel-ssh -t 0 -h $HOSTS_FILE "pip install flashinfer-python==0.2.5 triton==3.2.0"

# Phase 5: Clone and install vLLM
echo "=== Phase 5: vLLM installation (building from source, ~20-40 min) ==="
parallel-ssh -t 0 -h $HOSTS_FILE "rm -rf ~/vllm && git clone ${VLLM_GITHUB_LINK} ~/vllm"
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/vllm && git checkout ${VLLM_BRANCH}"
# Build from source with --no-build-isolation to avoid ABI mismatch
# This ensures vLLM is compiled against the installed PyTorch version
# Clean any stale build artifacts first to avoid partial build issues
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/vllm && rm -rf build/ vllm/*.so vllm/**/*.so .deps/ 2>/dev/null || true"
# Note: pip install may fail due to optional flashmla extension not building
# We handle this by manually copying the required .so files afterward
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/vllm && export PATH=\$HOME/.local/bin:/usr/local/cuda-12.6/bin:\$PATH && export CUDA_HOME=/usr/local/cuda-12.6 && export CUDACXX=/usr/local/cuda-12.6/bin/nvcc && pip install --no-cache-dir --no-build-isolation --editable . || true"
# Workaround: manually copy required .so files if pip install failed due to optional flashmla
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/vllm && if [ ! -f vllm/_C.abi3.so ]; then cp build/lib.linux-x86_64-*/vllm/_C.abi3.so vllm/ 2>/dev/null || true; fi"
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/vllm && if [ ! -f vllm/_moe_C.abi3.so ]; then cp build/lib.linux-x86_64-*/vllm/_moe_C.abi3.so vllm/ 2>/dev/null || true; fi"
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/vllm && if [ ! -f vllm/cumem_allocator.abi3.so ]; then cp build/lib.linux-x86_64-*/vllm/cumem_allocator.abi3.so vllm/ 2>/dev/null || true; fi"
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/vllm && mkdir -p vllm/vllm_flash_attn && cp build/lib.linux-x86_64-*/vllm/vllm_flash_attn/*.so vllm/vllm_flash_attn/ 2>/dev/null || true"
# Add PYTHONPATH to bashrc for subprocess compatibility
parallel-ssh -t 0 -h $HOSTS_FILE "grep -q 'PYTHONPATH.*vllm' ~/.bashrc || echo 'export PYTHONPATH=\$HOME/vllm:\$PYTHONPATH' >> ~/.bashrc"

# Phase 6: Clone and install Block
echo "=== Phase 6: Block installation ==="
parallel-ssh -t 0 -h $HOSTS_FILE "rm -rf ~/Block && git clone ${BLOCK_GITHUB_LINK} ~/Block"
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/Block && git checkout ${BLOCK_BRANCH}"
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/Block && pip install -r requirements.txt"
# Fix transformers version - vLLM block branch requires 4.50.3, not 5.0+
parallel-ssh -t 0 -h $HOSTS_FILE "pip install transformers==4.50.3"
# Note: Block doesn't have setup.py/pyproject.toml - use PYTHONPATH=. instead

# Phase 7: SCP local configs to remote hosts (configs are gitignored for security)
echo "=== Phase 7: Copy local configs to remote hosts ==="
CONFIG_DIR="block/config"
while IFS= read -r host; do
    echo "Copying configs to $host..."
    # Copy A30-specific configs
    scp ${CONFIG_DIR}/host_configs.json ${host}:~/Block/block/config/ 2>/dev/null || true
    scp ${CONFIG_DIR}/hosts ${host}:~/Block/block/config/ 2>/dev/null || true
    scp ${CONFIG_DIR}/llama_config.json ${host}:~/Block/block/config/ 2>/dev/null || true
    scp ${CONFIG_DIR}/qwen_config.json ${host}:~/Block/block/config/ 2>/dev/null || true
done < "$HOSTS_FILE"
echo "Config files copied to all hosts."

# Phase 8: Create directories and cleanup
echo "=== Phase 8: Create directories and cleanup ==="
parallel-ssh -t 0 -h $HOSTS_FILE "mkdir -p ~/Block/experiment_output/logs ~/Block/cache"
parallel-ssh -t 0 -h $HOSTS_FILE "rm -f ~/cuda-repo-*.deb"

# Phase 9: Verify installation
echo "=== Phase 9: Verify installation ==="
parallel-ssh -t 0 -h $HOSTS_FILE "python -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}\")'"
parallel-ssh -t 0 -h $HOSTS_FILE "python -c 'import vllm; print(f\"vLLM imported successfully\")'"

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
echo "  - --no-build-isolation for ABI compatibility"
echo "  - FlashMLA optional extension may fail (handled automatically)"
echo "  - MIG mode disabled (required for vLLM on A30)"
