BLOCK_GITHUB_LINK="https://github.com/AKafakA/Block"
VLLM_GITHUB_LINK="https://github.com/AKafakA/vllm.git"
OLLAMA_GITHUB_LINK="https://github.com/AKafakA/ollama.git"

 general setup for all hosts
echo "Install CUDA and dependencies on all hosts..."


# also need to manually run the following command to create the directory /mydata/hf_cache on all hosts and reboot will reset
# for d8545
# sudo mkfs.ext4 -F /dev/nvme0n1 && sudo mkdir -p /mydata && sudo mount /dev/nvme0n1 /mydata && sudo chmod 777 /mydata
# for c4130
# sudo mkfs.ext4 /dev/sdb && sudo mkdir -p /mydata && sudo mount /dev/sdb /mydata && sudo chmod 777 /mydata

parallel-ssh -t 0 -h block/config/hosts "sudo apt update && sudo apt full-upgrade -y"
parallel-ssh -t 0 -h block/config/hosts "sudo apt install -y python3-pip python3-venv ccache"
parallel-ssh -t 0 -h block/config/hosts "pip install --upgrade pip"
parallel-ssh -t 0 -h block/config/hosts "pip3 install torch torchvision"
parallel-ssh -t 0 -h block/config/hosts "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600"
parallel-ssh -t 0 -h block/config/hosts "wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb && sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb"
parallel-ssh -t 0 -h block/config/hosts "sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ && sudo apt-get update"
parallel-ssh -t 0 -h block/config/hosts "sudo dpkg --configure -a && sudo apt-get -y install cuda-toolkit-12-8"
parallel-ssh -t 0 -h block/config/ampere_hosts "sudo apt-get install -y nvidia-open"
parallel-ssh -t 0 -h block/config/volta_hosts "sudo apt-get install -y cuda-drivers"
parallel-ssh -t 0 -h block/config/pascal_hosts "sudo apt-get install -y cuda-drivers"
parallel-ssh -t 0 -h block/config/ampere_hosts "pip install --upgrade torch"
parallel-ssh -t 0 -h block/config/volta_hosts "pip install --upgrade torch"
#parallel-ssh -t 0 -h block/config/hosts "pip install --upgrade flash-attn"
parallel-ssh -t 0 -h block/config/hosts "pip install --upgrade "ray[cgraph]""
parallel-ssh -t 0 -h block/config/hosts "echo 'export PATH=$PATH:/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc && echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc && source ~/.bashrc"

echo "cuda installation completed on all hosts and now tested with nvidia-smi..."
parallel-ssh -t 0 -h block/config/hosts "sudo nvidia-smi -mig 0"
parallel-ssh -t 0 -h block/config/hosts  "rm -r ~/cuda-repo-*.deb"

# For ampere hosts and volta hosts, which are able to run vllm
echo "Starting setup for vllm hosts..."
parallel-ssh -t 0 -h block/config/ampere_hosts "git clone ${VLLM_GITHUB_LINK} && cd vllm && git checkout cara_v_11"
parallel-ssh -t 0 -h block/config/ampere_hosts  "cd vllm && sudo VLLM_USE_PRECOMPILED=1 pip install --editable ."
parallel-ssh -t 0 -h block/config/ampere_hosts "git clone ${BLOCK_GITHUB_LINK} && cd Block && git checkout cara  && pip install -r requirements.txt"
parallel-ssh -t 0 -h block/config/volta_hosts "git clone ${VLLM_GITHUB_LINK} && cd vllm && git checkout cara_v_11"
parallel-ssh -t 0 -h block/config/volta_hosts  "cd vllm && sudo VLLM_USE_PRECOMPILED=1 pip install --editable ."
parallel-ssh -t 0 -h block/config/volta_hosts "git clone ${BLOCK_GITHUB_LINK} && cd Block && git checkout cara  && pip install -r requirements.txt"


# install customized ollama for pascal hosts which cannot run vllm due to older architecture
echo "Starting setup for pascal hosts..."
parallel-ssh -t 0 -h block/config/pascal_hosts "sudo apt install -y cmake"
parallel-ssh -t 0 -h block/config/pascal_hosts "wget https://go.dev/dl/go1.24.0.linux-amd64.tar.gz"
parallel-ssh -t 0 -h block/config/pascal_hosts "sudo tar -C /usr/local -xzf go1.24.0.linux-amd64.tar.gz"
parallel-ssh -t 0 -h block/config/pascal_hosts "echo 'export PATH=\$PATH:/usr/local/go/bin:/usr/local/cuda-12.8/bin:/usr/local/cuda/bin' >> ~/.bashrc"
parallel-ssh -t 0 -h block/config/pascal_hosts "echo 'export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda-12.8/lib64' >> ~/.bashrc"
parallel-ssh -t 0 -h block/config/pascal_hosts "git clone ${OLLAMA_GITHUB_LINK} && cd ollama && git checkout status-api"
# Clean any existing build artifacts first
parallel-ssh -t 0 -h block/config/pascal_hosts "cd ollama && rm -rf build CMakeCache.txt CMakeFiles build"
# Set PATH explicitly in the cmake commands so nvcc is found
parallel-ssh -t 0 -h block/config/pascal_hosts "export PATH=\$PATH:/usr/local/go/bin:/usr/local/cuda-12.8/bin:/usr/local/cuda/bin && export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda-12.8/lib64 && cd ollama && cmake -B build -DCMAKE_CUDA_ARCHITECTURES=60 ."
parallel-ssh -t 0 -h block/config/pascal_hosts "export PATH=\$PATH:/usr/local/go/bin:/usr/local/cuda-12.8/bin:/usr/local/cuda/bin && export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda-12.8/lib64 && cd ollama && cmake --build build"
