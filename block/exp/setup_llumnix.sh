#!/bin/bash
# Llumnix setup for A100 nodes
# Creates a separate virtualenv with Llumnix 0.1.1 + vLLM 0.6.3.post1
HOST_FILE="block/config/a100_hosts"

# Fix Ubuntu mirror and enable universe repo
parallel-ssh -t 0 -h $HOST_FILE "sudo sed -i 's|us.archive.ubuntu.com|mirrors.edge.kernel.org|g' /etc/apt/sources.list"
parallel-ssh -t 0 -h $HOST_FILE "sudo sed -i 's|jammy restricted main|jammy restricted main universe|g; s|jammy-updates restricted main|jammy-updates restricted main universe|g; s|jammy-security restricted main|jammy-security restricted main universe|g' /etc/apt/sources.list"
parallel-ssh -t 0 -h $HOST_FILE "sudo apt update"
parallel-ssh -t 0 -h $HOST_FILE "sudo apt install -y python3-pip python3-venv ninja-build"

# CUDA setup
parallel-ssh -t 0 -h $HOST_FILE "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600"
parallel-ssh -t 0 -h $HOST_FILE "wget -q https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_amd64.deb && sudo dpkg -i cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_amd64.deb"
parallel-ssh -t 0 -h $HOST_FILE "sudo cp /var/cuda-repo-ubuntu2004-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/ && sudo apt-get update"
parallel-ssh -t 0 -h $HOST_FILE "sudo dpkg --configure -a && sudo apt-get -y install cuda-toolkit-12-6 && sudo apt-get install -y nvidia-open"
parallel-ssh -t 0 -h $HOST_FILE "echo 'export PATH=\$PATH:/usr/local/cuda-12.6/bin' >> ~/.bashrc && echo 'export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda-12.6/lib64' >> ~/.bashrc"
parallel-ssh -t 0 -h $HOST_FILE "sudo nvidia-smi -mig 0"

# NVMe mount for model storage
parallel-ssh -t 0 -h $HOST_FILE "if [ ! -d /mydata ]; then sudo mkfs.ext4 -F /dev/nvme0n1 && sudo mkdir -p /mydata && sudo mount /dev/nvme0n1 /mydata && sudo chmod 777 /mydata; fi && mkdir -p /mydata/huggingface"

# Create Llumnix virtualenv
parallel-ssh -t 0 -h $HOST_FILE "python3 -m venv /mydata/llumnix_venv"
parallel-ssh -t 0 -h $HOST_FILE "/mydata/llumnix_venv/bin/pip install --upgrade pip"
parallel-ssh -t 0 -h $HOST_FILE "/mydata/llumnix_venv/bin/pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121"
parallel-ssh -t 0 -h $HOST_FILE "/mydata/llumnix_venv/bin/pip install llumnix==0.1.1"
parallel-ssh -t 0 -h $HOST_FILE "/mydata/llumnix_venv/bin/pip install cupy-cuda12x ray==2.44.1"
parallel-ssh -t 0 -h $HOST_FILE "/mydata/llumnix_venv/bin/pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/"

# Clone Block repo for benchmark scripts
parallel-ssh -t 0 -h $HOST_FILE "git clone https://github.com/AKafakA/Block.git ~/Block 2>/dev/null || (cd ~/Block && git pull origin main)"

# Set HF_TOKEN
parallel-ssh -t 0 -h $HOST_FILE "echo 'export HF_HOME=/mydata/huggingface' >> ~/.bashrc && echo 'export VLLM_ATTENTION_BACKEND=FLASHINFER' >> ~/.bashrc"

# Create ~/.hf_token for non-interactive shells
echo -n "$(cat ~/.hf_token)" | parallel-ssh -t 0 -h $HOST_FILE -I "cat > ~/.hf_token && chmod 600 ~/.hf_token"

parallel-ssh -t 0 -h $HOST_FILE "rm -f ~/cuda-repo-*.deb"

echo "=== Llumnix Setup Complete ==="
echo "Next: deploy with block/exp/end_to_end_exp_scripts/a100_supplementary/deploy_llumnix.sh start"
