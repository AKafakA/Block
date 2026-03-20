#!/bin/bash
# A100 setup — identical to setup.sh but using a100_hosts
# IMPORTANT: Run on clean/reloaded nodes only. Do NOT run additional pip installs after.

BLOCK_GITHUB_LINK="https://github.com/AKafakA/Block.git"
VLLM_GITHUB_LINK="https://github.com/AKafakA/vllm.git"
HOST_FILE="block/config/a100_hosts"
HF_TOKEN=""

parallel-ssh -t 0 -h $HOST_FILE "sudo apt update && sudo apt full-upgrade -y"
parallel-ssh -t 0 -h $HOST_FILE "sudo apt install -y python3-pip python3-venv ccache"
parallel-ssh -t 0 -h $HOST_FILE "pip3 install -U pip==25.0.1"
parallel-ssh -t 0 -h $HOST_FILE "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600"
parallel-ssh -t 0 -h $HOST_FILE "wget -q https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_amd64.deb && sudo dpkg -i cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_amd64.deb"
parallel-ssh -t 0 -h $HOST_FILE "sudo cp /var/cuda-repo-ubuntu2004-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/ && sudo apt-get update"
parallel-ssh -t 0 -h $HOST_FILE "sudo dpkg --configure -a && sudo apt-get -y install cuda-toolkit-12-6 && sudo apt-get install -y nvidia-open"
parallel-ssh -t 0 -h $HOST_FILE "echo 'export PATH=\$PATH:/usr/local/cuda-12.6/bin' >> ~/.bashrc && echo 'export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda-12.6/lib64' >> ~/.bashrc"
parallel-ssh -t 0 -h $HOST_FILE "sudo nvidia-smi -mig 0"
parallel-ssh -t 0 -h $HOST_FILE "git clone ${VLLM_GITHUB_LINK} && cd vllm && sudo VLLM_USE_PRECOMPILED=1 pip install --editable ."
parallel-ssh -t 0 -h $HOST_FILE "git clone ${BLOCK_GITHUB_LINK} && cd Block && git checkout main && pip install -r requirements.txt"
parallel-ssh -t 0 -h $HOST_FILE "rm -f ~/cuda-repo-*.deb"
parallel-ssh -t 0 -h $HOST_FILE "pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126"
parallel-ssh -t 0 -h $HOST_FILE "pip install flashinfer-python==0.2.5 triton==3.2.0 transformers==4.50.3"

# A100 specific: NVMe + env
parallel-ssh -t 0 -h $HOST_FILE "if [ ! -d /mydata ]; then sudo mkfs.ext4 -F /dev/nvme0n1 && sudo mkdir -p /mydata && sudo mount /dev/nvme0n1 /mydata && sudo chmod 777 /mydata; fi && mkdir -p /mydata/huggingface"
parallel-ssh -t 0 -h $HOST_FILE "echo 'export HF_HOME=/mydata/huggingface' >> ~/.bashrc && echo 'export HF_TOKEN=${HF_TOKEN}' >> ~/.bashrc && echo 'export VLLM_USE_V1=0' >> ~/.bashrc"

echo "=== A100 Setup Complete ==="
