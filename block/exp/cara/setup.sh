BLOCK_GITHUB_LINK="https://github.com/AKafakA/Block"
VLLM_GITHUB_LINK="https://github.com/AKafakA/vllm.git"
OLLAMA_GITHUB_LINK="https://github.com/AKafakA/ollama.git"
BRANCH_NAME="cara-small"

# general setup for all hosts
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
parallel-ssh -t 0 -h block/config/hosts "pip install dacite"
parallel-ssh -t 0 -h block/config/hosts "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600"
parallel-ssh -t 0 -h block/config/hosts "wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb && sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb"
parallel-ssh -t 0 -h block/config/hosts "sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ && sudo apt-get update"
parallel-ssh -t 0 -h block/config/hosts "sudo dpkg --configure -a && sudo apt-get -y install cuda-toolkit-12-8"
parallel-ssh -t 0 -h block/config/ampere_hosts "sudo apt-get install -y nvidia-open"
parallel-ssh -t 0 -h block/config/volta_hosts "sudo apt-get install -y cuda-drivers"
parallel-ssh -t 0 -h block/config/pascal_hosts "sudo apt-get install -y cuda-drivers"
parallel-ssh -t 0 -h block/config/ampere_hosts "pip install --upgrade torch"
parallel-ssh -t 0 -h block/config/volta_hosts "pip install --upgrade torch"
parallel-ssh -t 0 -h block/config/hosts "pip install --upgrade "ray[cgraph]""
parallel-ssh -t 0 -h block/config/hosts "\
  echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc && \
  echo 'export PATH=\$PATH:\$CUDA_HOME/bin:/usr/local/cuda-12.8/bin' >> ~/.bashrc && \
  echo 'export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CUDA_HOME/lib64:/usr/local/cuda-12.8/lib64:\$CUDA_HOME/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu' >> ~/.bashrc && \
  echo '# Auto-discover all nvidia pip library paths' >> ~/.bashrc && \
  echo 'for dir in \$HOME/.local/lib/python3.10/site-packages/nvidia/*/lib /usr/local/lib/python3.10/dist-packages/nvidia/*/lib; do [ -d \"\$dir\" ] && export LD_LIBRARY_PATH=\$dir:\$LD_LIBRARY_PATH; done' >> ~/.bashrc && \
  source ~/.bashrc"

echo "cuda installation completed on all hosts and now tested with nvidia-smi..."
echo "If error, please consider to reboot the hosts and re-run nvidia-smi to verify cuda installation."
parallel-ssh -t 0 -h block/config/hosts "sudo nvidia-smi -mig 0"
parallel-ssh -t 0 -h block/config/hosts  "rm -r ~/cuda-repo-*.deb"

# Clone and setup Block on all hosts
parallel-ssh -t 0 -h block/config/hosts "git clone ${BLOCK_GITHUB_LINK} && cd Block && git checkout ${BRANCH_NAME}  && pip install -r requirements.txt"

# For ampere hosts and volta hosts, which are able to run vllm
echo "Starting setup for vllm hosts..."
parallel-ssh -t 0 -h block/config/ampere_hosts "git clone ${VLLM_GITHUB_LINK} && cd vllm && git checkout cara_v_11"
parallel-ssh -t 0 -h block/config/ampere_hosts  "cd vllm && sudo VLLM_USE_PRECOMPILED=1 pip install --editable ."
parallel-ssh -t 0 -h block/config/ampere_hosts "git clone ${BLOCK_GITHUB_LINK} && cd Block && git checkout cara  && pip install -r requirements.txt"

# Fix NumPy 2.x compatibility issues with sklearn, pandas, pyarrow
echo "Upgrading scikit-learn, pandas, pyarrow for NumPy 2.x compatibility..."
parallel-ssh -t 0 -h block/config/ampere_hosts "pip install --upgrade scikit-learn pandas pyarrow"

parallel-ssh -t 0 -h block/config/volta_hosts "git clone ${VLLM_GITHUB_LINK} && cd vllm && git checkout cara_v_11"
parallel-ssh -t 0 -h block/config/volta_hosts  "cd vllm && sudo VLLM_USE_PRECOMPILED=1 pip install --editable ."




# install customized vllm for pascal hosts (using the P100-compatible branch)
echo "Starting setup for pascal hosts..."
# 1. Install Build Dependencies
# vLLM requires cmake to build from source
parallel-ssh -t 0 -h block/config/pascal_hosts "sudo apt install -y cmake"
# 2. Clone vLLM and checkout your P100 branch
parallel-ssh -t 0 -h block/config/pascal_hosts "git clone ${VLLM_GITHUB_LINK} && cd vllm && git checkout cara_p100_v_6.0"
# 3. Install Xformers
# P100 cannot run FlashAttn, so we install xformers as the fallback backend
parallel-ssh -t 0 -h block/config/pascal_hosts "pip install xformers"
# 4. Build and Install vLLM
# We explicit set TORCH_CUDA_ARCH_LIST=6.0 to force the compiler to generate Pascal (sm_60) binaries.
# We pass the env var into sudo to ensure the build process sees it.
parallel-ssh -t 0 -h block/config/pascal_hosts "cd vllm && sudo CUDACXX=/usr/local/cuda-12.8/bin/nvcc TORCH_CUDA_ARCH_LIST=6.0 MAX_JOBS=7 CMAKE_BUILD_PARALLEL_LEVEL=7 pip install --editable ."
