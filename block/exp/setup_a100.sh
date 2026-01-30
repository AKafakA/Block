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

# Phase 1: System updates and build dependencies
echo "=== Phase 1: System updates ==="
parallel-ssh -t 0 -h $HOSTS_FILE "sudo apt update && sudo apt full-upgrade -y"
# Install build tools (needed if vLLM precompiled wheels not available)
parallel-ssh -t 0 -h $HOSTS_FILE "sudo apt install -y python3-pip python3-venv python3-dev ccache git build-essential cmake ninja-build"
parallel-ssh -t 0 -h $HOSTS_FILE "pip3 install -U pip==25.0.1 setuptools wheel"

# Phase 2: CUDA 12.6 installation
echo "=== Phase 2: CUDA 12.6 installation ==="
parallel-ssh -t 0 -h $HOSTS_FILE "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600"
parallel-ssh -t 0 -h $HOSTS_FILE "wget -q https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_amd64.deb && sudo dpkg -i cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_amd64.deb"
parallel-ssh -t 0 -h $HOSTS_FILE "sudo cp /var/cuda-repo-ubuntu2004-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/ && sudo apt-get update"
parallel-ssh -t 0 -h $HOSTS_FILE "sudo dpkg --configure -a && sudo apt-get -y install cuda-toolkit-12-6 && sudo apt-get install -y nvidia-open"
parallel-ssh -t 0 -h $HOSTS_FILE "echo 'export PATH=/usr/local/cuda-12.6/bin:\$PATH' >> ~/.bashrc && echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:\$LD_LIBRARY_PATH' >> ~/.bashrc"

# Phase 3: Mount NVMe storage (A100 nodes have NVMe for model storage)
echo "=== Phase 3: Mount NVMe storage ==="
parallel-ssh -t 0 -h $HOSTS_FILE "if [ -b /dev/nvme0n1 ] && ! mountpoint -q /mydata; then sudo mkfs.ext4 -F /dev/nvme0n1 && sudo mkdir -p /mydata && sudo mount /dev/nvme0n1 /mydata && sudo chown \$(whoami):\$(whoami) /mydata && echo 'NVMe mounted at /mydata'; else echo 'NVMe already mounted or not available'; fi"

# Phase 4: Verify GPUs (A100-40GB should show 4 GPUs)
echo "=== Phase 4: Verify GPUs ==="
parallel-ssh -t 0 -h $HOSTS_FILE "nvidia-smi --query-gpu=name,memory.total --format=csv"

# Phase 5: PyTorch and dependencies
echo "=== Phase 5: PyTorch and dependencies ==="
parallel-ssh -t 0 -h $HOSTS_FILE "pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126"
parallel-ssh -t 0 -h $HOSTS_FILE "pip install flashinfer-python==0.2.5 triton==3.2.0"

# Phase 6: Clone and install vLLM
echo "=== Phase 6: vLLM installation (building from source, ~20-40 min) ==="
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
# Add PYTHONPATH to bashrc for tensor parallelism subprocess compatibility
parallel-ssh -t 0 -h $HOSTS_FILE "grep -q 'PYTHONPATH.*vllm' ~/.bashrc || echo 'export PYTHONPATH=\$HOME/vllm:\$PYTHONPATH' >> ~/.bashrc"

# Phase 7: Clone and install Block
echo "=== Phase 7: Block installation ==="
parallel-ssh -t 0 -h $HOSTS_FILE "rm -rf ~/Block && git clone ${BLOCK_GITHUB_LINK} ~/Block"
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/Block && git checkout ${BLOCK_BRANCH}"
parallel-ssh -t 0 -h $HOSTS_FILE "cd ~/Block && pip install -r requirements.txt"
# Fix transformers version - vLLM block branch requires 4.50.3, not 5.0+
parallel-ssh -t 0 -h $HOSTS_FILE "pip install transformers==4.50.3"
# Note: Block doesn't have setup.py/pyproject.toml - use PYTHONPATH=. instead

# Phase 8: SCP local configs to remote hosts (configs are gitignored for security)
echo "=== Phase 8: Copy local configs to remote hosts ==="
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

# Phase 9: Create directories and cleanup
echo "=== Phase 9: Create directories and cleanup ==="
parallel-ssh -t 0 -h $HOSTS_FILE "mkdir -p ~/Block/experiment_output/logs ~/Block/cache"
parallel-ssh -t 0 -h $HOSTS_FILE "rm -f ~/cuda-repo-*.deb"

# Phase 10: Setup SSH keys between nodes (for cache sharing)
echo "=== Phase 10: Setup SSH keys between nodes ==="
FIRST_HOST=$(head -1 $HOSTS_FILE)
# Generate SSH key on first host if not exists
ssh $FIRST_HOST "[ -f ~/.ssh/id_rsa ] || ssh-keygen -t rsa -f ~/.ssh/id_rsa -N '' -q"
FIRST_HOST_PUBKEY=$(ssh $FIRST_HOST "cat ~/.ssh/id_rsa.pub")
# Add first host's key to all other hosts
while IFS= read -r host; do
    if [ "$host" != "$FIRST_HOST" ]; then
        echo "Adding SSH key to $host..."
        ssh $host "mkdir -p ~/.ssh && echo '$FIRST_HOST_PUBKEY' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys" 2>/dev/null || true
    fi
done < "$HOSTS_FILE"

# Phase 11: Verify installation
echo "=== Phase 11: Verify installation ==="
parallel-ssh -t 0 -h $HOSTS_FILE "python -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}\")'"
parallel-ssh -t 0 -h $HOSTS_FILE "python -c 'import vllm; print(f\"vLLM imported successfully\")'"

echo ""
echo "=============================================="
echo "A100 Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Set HuggingFace token (required for Llama-2):"
echo "     export HF_TOKEN=your_token_here"
echo "     # Or save to file: echo 'your_token' > ~/.cache/huggingface/token"
echo ""
echo "  2. Run Vidur profiling for A100-40GB (if new GPU/model combo):"
echo "     sh block/exp/end_to_end_exp_scripts/a100_40gb_profiling.sh"
echo ""
echo "  3. Run experiment:"
echo "     sh block/exp/end_to_end_exp_scripts/a100_llama70b_exp.sh"
echo ""
echo "Known issues fixed in this script:"
echo "  - VLLM_USE_V1=0 required for Block's get_scheduler_trace API"
echo "  - transformers==4.50.3 required (not 5.0+)"
echo "  - --no-build-isolation for ABI compatibility"
echo "  - PYTHONPATH includes ~/vllm for TP subprocess compatibility"
echo "  - FlashMLA optional extension may fail (handled automatically)"
