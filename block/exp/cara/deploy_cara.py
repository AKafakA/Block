import json
import os
import re
import sys
import argparse
import subprocess
from typing import Dict, List


# --- Helpers ---

def parse_host_file(filepath: str) -> Dict[str, List[str]]:
    node_pool = {}
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Host file '{filepath}' not found.")
        sys.exit(1)

    for line in lines:
        line = line.strip()
        if not line: continue
        # Matches patterns like: asdwb@d8545-10s10301...
        match = re.search(r'@([a-zA-Z0-9]+)-', line)
        if match:
            node_type = match.group(1)
            if node_type not in node_pool:
                node_pool[node_type] = []
            node_pool[node_type].append(line)
    return node_pool


def run_ssh_cmd(host: str, commands: List[str], description: str):
    print(f"[{description}] Connecting to {host}...")

    # --- CRITICAL FIX IS HERE ---
    # Filter out None, empty strings "", or strings with just whitespace " "
    valid_commands = [c for c in commands if c and c.strip()]

    # Now join safely
    full_command = " && ".join(valid_commands)
    # ----------------------------

    ssh_params = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        host,
        full_command
    ]

    try:
        # Timeout set to 15 mins (900s) to be safe for large model weights
        result = subprocess.run(ssh_params, capture_output=True, text=True, timeout=900)

        if result.returncode != 0:
            print(f"  [FAILED] Exit Code {result.returncode}")
            print(f"  [STDERR] {result.stderr.strip()}")
            # Print STDOUT too, as sometimes errors appear there
            if result.stdout:
                print(f"  [STDOUT] {result.stdout.strip()}")
        else:
            print(f"  [SUCCESS] {description} executed.")
            # Print stdout even on success to help with debugging
            if result.stdout and result.stdout.strip():
                print(f"  [OUTPUT] {result.stdout.strip()}")

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Command took too long (check connection or extend timeout).")
    except Exception as e:
        print(f"  [ERROR] Connection failed: {str(e)}")


def to_ollama_tag(hf_name: str) -> str:
    name = hf_name.lower()
    if "/" in name:
        name = name.split("/")[-1]
    name = name.replace("-", ":")
    return name


# --- Command Generators ---

def get_cleanup_commands(backend: str) -> List[str]:
    """Generate commands to kill existing processes before deployment"""
    if backend == "vllm":
        cmds = [
            "echo 'Cleaning up existing vLLM processes...'",
            "pkill -f 'vllm.entrypoints.openai.api_server' || echo 'No existing vLLM process found'",
            "sleep 2"
        ]
    elif backend == "ollama":
        cmds = [
            "echo 'Cleaning up existing Ollama processes...'",
            # Kill any process using port 11434 first
            "fuser -k 11434/tcp 2>/dev/null || echo 'Port 11434 not in use'",
            # Kill Go processes related to ollama
            "pkill -9 -f 'go run . serve' || echo 'No go run process found'",
            # Kill any ollama-related processes
            "pkill -9 -f 'ollama' || echo 'No ollama process found'",
            # Wait longer for port to be released
            "sleep 5",
            # Verify port is free
            "echo 'Verifying port 11434 is free...'",
            "! lsof -ti:11434 || (echo 'WARNING: Port 11434 still in use' && lsof -ti:11434 | xargs kill -9)",
            "sleep 2"
        ]
    else:
        cmds = []
    return cmds


def get_vllm_commands(model_path: str, hf_token: str, precision: str, vllm_params: Dict = None) -> List[str]:
    dtype_flag = "float16" if precision == "fp16" else "auto"

    # Ensure token is not None to prevent Python crash
    token_str = hf_token if hf_token else ""

    # Check if we should use custom HF cache
    use_hf_cache = vllm_params.get("use_hf_cache", False) if vllm_params else False

    # Build vLLM command with parameters
    vllm_cmd_parts = [
        "nohup python3 -u -m vllm.entrypoints.openai.api_server",
        f"--model {model_path}",
        f"--dtype {dtype_flag}",
        "--tensor-parallel-size $GPU_COUNT",
        "--trust-remote-code"
    ]

    # Add optional vLLM parameters from config (skip use_hf_cache as it's not a vLLM param)
    if vllm_params:
        if "gpu-memory-utilization" in vllm_params:
            vllm_cmd_parts.append(f"--gpu-memory-utilization {vllm_params['gpu-memory-utilization']}")
        if "max-model-len" in vllm_params:
            vllm_cmd_parts.append(f"--max-model-len {vllm_params['max-model-len']}")
        if vllm_params.get("enforce-eager"):
            vllm_cmd_parts.append("--enforce-eager")

    vllm_cmd_str = " ".join(vllm_cmd_parts)

    cmds = [
        "echo 'Step 1: Checking vllm directory...'",
        "if [ ! -d ~/vllm ]; then echo 'ERROR: vllm directory not found'; exit 1; fi",
        "echo 'Step 2: Changing to vllm directory...'",
        "cd ~/vllm",
        "echo 'Step 3: Setting environment variables...'",
        f"export HF_TOKEN={token_str}",
    ]

    # Only export HF_HOME if use_hf_cache is true
    if use_hf_cache:
        cmds.append("export HF_HOME=/mydata/hf_cache")
        cmds.append("echo 'Using custom HF cache at /mydata/hf_cache'")
    else:
        cmds.append("echo 'Using system default HF cache'")

    cmds.extend([
        "export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/nvshmem/lib:$LD_LIBRARY_PATH",
        "echo 'Step 4: Detecting GPU count...'",
        "export GPU_COUNT=$(python3 -c 'import torch; print(torch.cuda.device_count())')",
        "echo \"GPU_COUNT=$GPU_COUNT\"",
        "echo 'Step 5: Launching vLLM server in background...'",
        f"sh -c 'cd ~/vllm && {vllm_cmd_str} > vllm_server.log 2>&1 < /dev/null &'",
        "sleep 2",
        "echo 'Deployment completed!'",
        "exit 0"
    ])
    return cmds


def get_ollama_commands(hf_name: str, num_parallel: int = 4) -> List[str]:
    ollama_tag = to_ollama_tag(hf_name)

    cmds = [
        "if [ ! -d ~/ollama ]; then echo 'ERROR: ollama directory not found'; exit 1; fi",
        "cd ~/ollama",
        "echo 'Setting environment variables...'",
        # Export PATH to include Go and CUDA (same as in setup.sh)
        "export PATH=$PATH:/usr/local/go/bin:/usr/local/cuda-12.8/bin:/usr/local/cuda/bin",
        "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.8/lib64",
        # CRITICAL: Set OLLAMA_HOST to listen on all interfaces, not just localhost
        "export OLLAMA_HOST=0.0.0.0:11434",
        # Enable parallel request processing to maximize GPU utilization
        f"export OLLAMA_NUM_PARALLEL={num_parallel}",
        "export OLLAMA_MAX_LOADED_MODELS=1",
        f"echo 'Ollama server starting with {num_parallel} parallel requests (listening on 0.0.0.0:11434)...'",
        # Start server with go run wrapped in sh -c - environment variables will be inherited
        "sh -c 'cd ~/ollama && go run . serve > ollama_server.log 2>&1 < /dev/null &'",
        "sleep 5",
        f"echo 'Pulling {ollama_tag} via REST API (synchronous)...'",
        # Pull synchronously to wait for completion before warmup
        f'curl -s http://localhost:11434/api/pull -d \'{{\"model\": \"{ollama_tag}\"}}\' > ollama_pull.log 2>&1',
        f"echo 'Model pulled successfully. Warming up {ollama_tag} to load into GPU...'",
        # Warmup inference to load model into GPU memory
        f'curl -s http://localhost:11434/api/generate -d \'{{\"model\": \"{ollama_tag}\", \"prompt\": \"Hello\", \"stream\": false}}\' > ollama_warmup.log 2>&1',
        "echo 'Warmup completed. Model loaded into GPU memory.'",
        "echo 'Ollama deployment completed!'",
        "exit 0"
    ]
    return cmds


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Model Deployment Script v4")

    parser.add_argument("--hosts", help="Host file path",
                        default="block/config/hosts")
    parser.add_argument("--config", help="Model config JSON",
                        default="block/config/cara/model_config_template.json")

    # NOTE: Hardcoded for local testing. Remove before committing to public repo.
    parser.add_argument("--hf-token", help="Hugging Face Token",
                        default="")

    parser.add_argument("--output", default="block/config/cara/model_deployment.json",
                        help="Output config path")

    parser.add_argument("--models", type=str, default="Qwen-2.5-3B",
                        help="Comma-separated list of models to deploy (e.g., 'Qwen-2.5-3B' or 'Qwen-2.5-3B,Qwen-2.5-7B'). Deploy all if not specified.")

    parser.add_argument("--ollama-num-parallel", type=int, default=4,
                        help="Number of parallel requests Ollama can handle (default: 4). Higher values increase GPU utilization.")

    args = parser.parse_args()

    # 1. Load Data
    available_nodes = parse_host_file(args.hosts)
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)

    # Load existing deployment config if doing selective deployment
    final_config = {}
    if args.models and os.path.exists(args.output):
        try:
            with open(args.output, 'r') as f:
                final_config = json.load(f)
            print(f"--- Loaded existing deployment config from {args.output} ---")
        except:
            print(f"--- Could not load existing config, starting fresh ---")

    print(f"--- Loaded {sum(len(v) for v in available_nodes.values())} nodes ---")

    # Filter models if --models specified
    models_to_deploy = None
    if args.models:
        models_to_deploy = [m.strip() for m in args.models.split(',')]
        print(f"--- Deploying only specified models: {models_to_deploy} ---")
        print(f"--- Existing models will be preserved in output config ---")
    else:
        print(f"--- Deploying all models ---")

    # 2. Allocation & Deployment
    for model_key, details in config.items():
        # Skip if not in the list of models to deploy
        if models_to_deploy and model_key not in models_to_deploy:
            print(f"\nSkipping: {model_key} (not in deployment list)")
            # Still add to final config if it exists in output file
            continue
        print(f"\nProcessing: {model_key}...")

        # --- Allocation Logic ---
        new_entry = details.copy()
        assigned_hosts = []
        vllm_params = {}  # Store vLLM-specific parameters

        if 'node_type' in details:
            for ntype, node_config in details['node_type'].items():
                # Support both old format (just count) and new format (dict with count + params)
                if isinstance(node_config, int):
                    # Old format: "d8545": 2
                    count = node_config
                else:
                    # New format: "d8545": {"count": 2, "gpu-memory-utilization": 0.95, ...}
                    count = node_config.get('count', 1)
                    # Extract vLLM parameters (everything except 'count')
                    vllm_params = {k: v for k, v in node_config.items() if k != 'count'}

                if ntype not in available_nodes or len(available_nodes[ntype]) < count:
                    print(f"CRITICAL ERROR: Not enough nodes of type {ntype} for {model_key}")
                    # In a real script you might want to continue, but exiting is safer here
                    sys.exit(1)

                nodes = available_nodes[ntype][:count]
                # Remove used nodes from pool
                available_nodes[ntype] = available_nodes[ntype][count:]
                assigned_hosts.extend(nodes)

            del new_entry['node_type']
            new_entry['node_hosts'] = assigned_hosts
        else:
            assigned_hosts = details.get('node_hosts', [])

        final_config[model_key] = new_entry

        # --- SSH Deployment Logic ---
        backend = details.get('backend', 'vllm')
        hf_name = details.get('hf_model_name')
        precision = details.get('precision', 'fp16')

        for host in assigned_hosts:
            # First, cleanup existing processes
            cleanup_cmds = get_cleanup_commands(backend)
            if cleanup_cmds:
                run_ssh_cmd(host, cleanup_cmds, f"Cleanup ({model_key})")

            # Then deploy
            if backend == "vllm":
                if vllm_params:
                    print(f"  vLLM params: {vllm_params}")
                cmds = get_vllm_commands(hf_name, args.hf_token, precision, vllm_params)
                run_ssh_cmd(host, cmds, f"Deploying vLLM ({model_key})")
            elif backend == "ollama":
                print(f"  Ollama parallel requests: {args.ollama_num_parallel}")
                cmds = get_ollama_commands(hf_name, num_parallel=args.ollama_num_parallel)
                run_ssh_cmd(host, cmds, f"Deploying Ollama ({model_key})")

    # 3. Save Config
    with open(args.output, 'w') as f:
        json.dump(final_config, f, indent=2)
    print(f"\nDone! Final configuration saved to {args.output}")


if __name__ == "__main__":
    main()