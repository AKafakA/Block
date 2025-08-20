# Head node host for ray clusters
HEAD_NODE_HOST=""
HEAD_NODE_IP=""
HUGGINGFACE_TOKEN=""

parallel-ssh -t 0 -h block/config/hosts "sudo apt update && sudo apt full-upgrade -y"
parallel-ssh -t 0 -h block/config/hosts "pip install ray==2.44.1"
parallel-ssh -t 0 -h block/config/hosts "sudo apt-get install python3-pip -y"
parallel-ssh -t 0 -h block/config/hosts "pip install torch==2.3.0 llumnix torchvision==0.18.0 cupy-cuda12x"
parallel-ssh -t 0 -h block/config/hosts "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600"
parallel-ssh -t 0 -h block/config/hosts "wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_amd64.deb && sudo dpkg -i cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_amd64.deb"
parallel-ssh -t 0 -h block/config/hosts "sudo cp /var/cuda-repo-ubuntu2004-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/ && sudo apt-get update"
parallel-ssh -t 0 -h block/config/hosts "sudo dpkg --configure -a && sudo apt-get -y install cuda-toolkit-12-6 && sudo apt-get install -y nvidia-open"
parallel-ssh -i -t 0 -h block/config/hosts "export HEAD_NODE_IP=$HEAD_NODE_IP && export HF_TOKEN=$HUGGINGFACE_TOKEN && python -m llumnix.entrypoints.vllm.api_server --model meta-llama/Llama-2-7b-hf --port 8200 --initial-instances 1 --engine-use-ray --worker-use-ray --launch-ray-cluster "
