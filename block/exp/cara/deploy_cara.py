import json
import os
import re
import sys
import argparse
import subprocess
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed


# --- Constants ---
# Sleep times for deployment (in seconds)
FRESH_DEPLOY_SLEEP_SECONDS = 600  # Time to download models from HuggingFace
NORMAL_DEPLOY_SLEEP_SECONDS = 10  # Time to start already-cached models


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
    """Execute SSH command synchronously (blocking). Used for sequential deployment."""
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


def run_ssh_cmd_async(host: str, commands: List[str], description: str) -> Dict:
    """Execute SSH command and return result dict (for parallel deployment)."""
    valid_commands = [c for c in commands if c and c.strip()]
    full_command = " && ".join(valid_commands)

    ssh_params = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        host,
        full_command
    ]

    try:
        result = subprocess.run(ssh_params, capture_output=True, text=True, timeout=900)
        return {
            "host": host,
            "description": description,
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout.strip() if result.stdout else "",
            "stderr": result.stderr.strip() if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "host": host,
            "description": description,
            "success": False,
            "error": "TIMEOUT: Command took too long (>15 minutes)"
        }
    except Exception as e:
        return {
            "host": host,
            "description": description,
            "success": False,
            "error": f"Connection failed: {str(e)}"
        }


def deploy_parallel(tasks: List[tuple], max_workers: int = 20) -> bool:
    """
    Deploy to multiple hosts in parallel.

    Args:
        tasks: List of (host, commands, description) tuples
        max_workers: Maximum parallel SSH connections

    Returns:
        True if all deployments succeeded, False otherwise
    """
    if not tasks:
        return True

    print(f"\n{'='*60}")
    print(f"Starting parallel deployment to {len(tasks)} hosts...")
    print(f"Max parallel connections: {max_workers}")
    print(f"{'='*60}\n")

    all_success = True
    completed = 0
    total = len(tasks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_ssh_cmd_async, host, commands, description): (host, description)
            for host, commands, description in tasks
        }

        # Process results as they complete
        for future in as_completed(future_to_task):
            host, description = future_to_task[future]
            completed += 1

            try:
                result = future.result()
                if result["success"]:
                    print(f"[{completed}/{total}] ✓ {host} - {description}")
                    # Print important output (condensed)
                    if result.get("stdout") and "Deployment completed" in result["stdout"]:
                        print(f"         → Deployment completed successfully")
                else:
                    all_success = False
                    print(f"[{completed}/{total}] ✗ {host} - {description}")
                    if "error" in result:
                        print(f"         → Error: {result['error']}")
                    else:
                        print(f"         → Exit code: {result.get('returncode', 'unknown')}")
                        if result.get("stderr"):
                            # Print first 200 chars of stderr
                            stderr_preview = result["stderr"][:200]
                            print(f"         → {stderr_preview}...")
            except Exception as e:
                all_success = False
                print(f"[{completed}/{total}] ✗ {host} - {description}")
                print(f"         → Exception: {str(e)}")

    print(f"\n{'='*60}")
    if all_success:
        print(f"✓ All {total} deployments completed successfully!")
    else:
        print(f"✗ Some deployments failed. Check logs above.")
    print(f"{'='*60}\n")

    return all_success


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
            # Anchor to python so we don't kill the current remote shell running 'sh -c "... pkill ..."'
            "pkill -f '^python.*vllm\\.entrypoints\\.openai\\.api_server' || echo 'No existing vLLM process found'",
            "sleep 2"
        ]
    elif backend == "ollama":
        cmds = [
            "echo 'Cleaning up existing Ollama processes...'",
            # Kill any process using port 11434 first
            "fuser -k 11434/tcp 2>/dev/null || echo 'Port 11434 not in use'",
            # Kill Go processes related to ollama
            # Anchor to 'go' to avoid matching the current 'sh -c' command string
            "pkill -9 -f '^go .*run .*\\. .*serve' || echo 'No go run process found'",
            # Kill any ollama-related processes
            # Anchor to 'ollama' binary name if present
            "pkill -9 -f '^ollama( |$)' || echo 'No ollama process found'",
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


def get_vllm_commands(model_path: str, hf_token: str, precision: str, vllm_params: Dict = None,
                      fresh_deploy: bool = False) -> List[str]:
    # Use dtype from vllm_params if present (for old vLLM v0 versions), otherwise derive from precision
    if vllm_params and "dtype" in vllm_params:
        dtype_flag = vllm_params["dtype"]
    else:
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

    # Add optional vLLM parameters from config
    # Filter out non-vLLM params: use_hf_cache, serve_with_v0, dtype (already used above), attention_backend (set as env var)
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

    # Set attention backend as environment variable for old vLLM v0
    if vllm_params and "attention_backend" in vllm_params:
        backend_value = vllm_params["attention_backend"].upper()
        cmds.append(f"export VLLM_ATTENTION_BACKEND={backend_value}")
        cmds.append(f"echo 'Using attention backend: {backend_value}'")

    sleep_time = FRESH_DEPLOY_SLEEP_SECONDS if fresh_deploy else NORMAL_DEPLOY_SLEEP_SECONDS

    cmds.extend([
        # Ensure CUDA env and common library locations (covers CUDA 12.x + pip-installed NVIDIA libs)
        "export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}",
        # Dynamically add all nvidia lib paths from user's pip site-packages
        (
            "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:" \
            "$CUDA_HOME/lib64:" \
            "$CUDA_HOME/targets/x86_64-linux/lib:" \
            "/usr/local/cuda-12.8/lib64:" \
            "/usr/lib/x86_64-linux-gnu:" \
            "$(python3 -c 'import site; print(site.getusersitepackages())')/nvidia/cudnn/lib:" \
            "$(python3 -c 'import site; print(site.getusersitepackages())')/nvidia/cusparselt/lib:" \
            "$(python3 -c 'import site; print(site.getusersitepackages())')/nvidia/nccl/lib:" \
            "$(python3 -c 'import site; print(site.getusersitepackages())')/nvidia/nvjitlink/lib:" \
            "$(python3 -c 'import site; print(site.getusersitepackages())')/nvidia/nvshmem/lib:" \
            "$(python3 -c 'import site; print(site.getusersitepackages())')/nvidia/cublas/lib:" \
            "$(python3 -c 'import site; print(site.getusersitepackages())')/nvidia/cuda_runtime/lib:" \
            "/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:" \
            "/usr/local/lib/python3.10/dist-packages/nvidia/cusparselt/lib:" \
            "/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:" \
            "/usr/local/lib/python3.10/dist-packages/nvidia/nvshmem/lib"
        ),
        "export PATH=$PATH:$CUDA_HOME/bin:/usr/local/cuda-12.8/bin",
        "echo 'Step 4: Detecting GPU count...'",
        "export GPU_COUNT=$(python3 -c 'import torch; print(torch.cuda.device_count())')",
        "echo \"GPU_COUNT=$GPU_COUNT\"",
        "echo 'Step 5: Launching vLLM server in background...'",
        f"sh -c 'cd ~/vllm && {vllm_cmd_str} > vllm_server.log 2>&1 < /dev/null &'",
        f"sleep {sleep_time}",
        "echo 'Deployment completed!'",
        "exit 0"
    ])
    return cmds


def get_predictor_deployment_commands(
    hostname: str,
    backend_port: int,
    predictor_config: Dict,
    host_config: Dict
) -> List[str]:
    """Generate commands to deploy CARA predictors on a host.

    Args:
        hostname: The hostname (without user@)
        backend_port: Backend port from host_config
        predictor_config: Predictor deployment configuration
        host_config: Host configuration containing predictor_ports

    Returns:
        List of shell commands to deploy predictors
    """
    predictor_type = predictor_config.get("predictor_type", "dummy")
    predictor_ports = host_config[hostname]["predictor_ports"]
    # Avoid port collision with backend by skipping backend_port if present
    predictor_ports = [p for p in predictor_ports if p != backend_port]

    # Get data output directory for creating the directory structure
    # (Other config settings like enable_data_collection, sample_rate are read directly by predictor from config file)
    data_output_dir = predictor_config.get("data_output_dir", "./training_data/cara")

    # Build a single aggregated command string to avoid "& &&" join issues.
    header_parts = [
        "echo 'Deploying CARA Predictors...'",
        # Anchor to python to avoid killing the remote 'sh -c' that contains this string in its command line
        "pkill -f '^python.*block\\.predictor\\.cara\\.cara_predictor_api_server' || echo 'No existing predictors'",
        "sleep 2",
        "mkdir -p Block/experiment_output/logs",
        f"mkdir -p Block/{data_output_dir}",
    ]
    header_cmd = " && ".join(header_parts)

    # Start all predictors in background within one grouped command after cd
    bg_cmds = []
    for predictor_port in predictor_ports:
        bg_cmds.append(
            (
                f"nohup $PYTHON_BIN -u -m block.predictor.cara.cara_predictor_api_server "
                f"--host 0.0.0.0 "
                f"--port {predictor_port} "
                f"--backend-port {backend_port} "
                f"--hostname {hostname} "
                f"--config-path block/config/cara/predictor_deployment_config.json "
                f"> experiment_output/logs/predictor_{predictor_port}.log 2>&1 < /dev/null &"
            )
        )

    # Group background launches so the outer command does not end with '&'.
    # Detect usable python binary inside the group.
    group_cmd = (
        "cd Block && ( "
        "export PYTHONUNBUFFERED=1; "
        "PYTHON_BIN=${PREDICTOR_PYTHON_BIN:-$(command -v python3 || command -v python)}; "
        + " ".join(bg_cmds) +
        " true )"
    )

    # Verify processes are up; fail if any did not start. Print concise summary only.
    ports_list = " ".join(str(p) for p in predictor_ports)
    verify_cmd = (
        "sleep 5 && "
        "fail=0; failed_ports=''; "
        f"for p in {ports_list}; do "
        "if pgrep -f \"block.predictor.cara.cara_predictor_api_server.*--port $p\" >/dev/null; then :; "
        "else echo \"Predictor failed on port $p\" 1>&2; failed_ports=\"$failed_ports $p\"; fail=1; fi; done; "
        "if [ $fail -ne 0 ]; then echo \"Failed predictor ports:$failed_ports\" 1>&2; exit 1; fi"
    )

    # Final echo (concise)
    tail_cmd = f"echo 'Predictors OK: {len(predictor_ports)}/{len(predictor_ports)}'"

    combined = " && ".join([header_cmd, group_cmd, verify_cmd, tail_cmd])
    return [combined]


def get_ollama_commands(hf_name: str, num_parallel: int = 4, fresh_deploy: bool = False) -> List[str]:
    ollama_tag = to_ollama_tag(hf_name)

    sleeping_time = FRESH_DEPLOY_SLEEP_SECONDS if fresh_deploy else NORMAL_DEPLOY_SLEEP_SECONDS

    cmds = [
        "if [ ! -d ~/ollama ]; then echo 'ERROR: ollama directory not found'; exit 1; fi",
        "cd ~/ollama",
        "echo 'Setting environment variables...'",
        # Export PATH to include Go and CUDA (same as in setup.sh)
        "export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}",
        "export PATH=$PATH:/usr/local/go/bin:$CUDA_HOME/bin:/usr/local/cuda-12.8/bin",
        (
            "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:" \
            "$CUDA_HOME/lib64:" \
            "$CUDA_HOME/targets/x86_64-linux/lib:" \
            "/usr/local/cuda-12.8/lib64:" \
            "/usr/lib/x86_64-linux-gnu:" \
            "/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:" \
            "/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:" \
            "/usr/local/lib/python3.10/dist-packages/cusparselt/lib"
        ),
        # CRITICAL: Set OLLAMA_HOST to listen on all interfaces, not just localhost
        "export OLLAMA_HOST=0.0.0.0:11434",
        # Enable parallel request processing to maximize GPU utilization
        f"export OLLAMA_NUM_PARALLEL={num_parallel}",
        "export OLLAMA_MAX_LOADED_MODELS=1",
        f"echo 'Ollama server starting with {num_parallel} parallel requests (listening on 0.0.0.0:11434)...'",
        # Start server with go run wrapped in sh -c - environment variables will be inherited
        "sh -c 'cd ~/ollama && go run . serve > ollama_server.log 2>&1 < /dev/null &'",
        f"sleep {sleeping_time}",
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

    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated list of models to deploy (e.g., 'Qwen-2.5-3B' or 'Qwen-2.5-3B,Qwen-2.5-7B'). Deploy all if not specified.")

    parser.add_argument("--ollama-num-parallel", type=int, default=4,
                        help="Number of parallel requests Ollama can handle (default: 4). Higher values increase GPU utilization.")

    parser.add_argument("--fresh-deploy", action="store_true",
                        help="If set, it usually need much longer time to download models from HF and golang packages for Ollama.")

    parser.add_argument("--no-deploy-predictors", dest="deploy_predictors", action="store_false", default=True,
                        help="Skip deploying CARA predictors (default: predictors ARE deployed)")
    parser.add_argument("--predictor-config", type=str,
                        default="block/config/cara/predictor_deployment_config.json",
                        help="Path to predictor deployment config")
    parser.add_argument("--host-config", type=str,
                        default="block/config/host_configs.json",
                        help="Path to host config file (contains backend_port and predictor_ports)")
    parser.add_argument(
        "-d", "--deploy-services",
        nargs="+",
        default=["model_instance", "predictor"],
        help=(
            "Services to deploy as a list: model_instance predictor. "
            "Default: model_instance predictor"
        )
    )

    parser.add_argument("--no-distribute-config", dest="distribute_config", action="store_false", default=True,
                        help="Skip distributing config to remote nodes (default: config IS distributed)")

    parser.add_argument("--sequential", dest="parallel", action="store_false", default=True,
                        help="Deploy to nodes sequentially (slower, useful for debugging). Default: parallel deployment")
    parser.add_argument("--max-parallel-workers", type=int, default=20,
                        help="Maximum number of parallel SSH connections (default: 20)")

    args = parser.parse_args()

    # 1. Load Data
    available_nodes = parse_host_file(args.hosts)
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)

    # Determine which services to deploy (list of strings)
    deploy_services = [s.lower() for s in (args.deploy_services or [])]
    deploy_model_instances = "model_instance" in deploy_services
    deploy_predictors_flag = ("predictor" in deploy_services) and args.deploy_predictors

    # Load host config and predictor config if deploying predictors
    host_config = None
    predictor_config = None
    if deploy_predictors_flag:
        try:
            with open(args.host_config, 'r') as f:
                host_config = json.load(f)
            print(f"--- Loaded host config from {args.host_config} ---")
        except FileNotFoundError:
            print(f"Error: Host config file '{args.host_config}' not found.")
            sys.exit(1)

        try:
            with open(args.predictor_config, 'r') as f:
                predictor_config = json.load(f)
            print(f"--- Loaded predictor config from {args.predictor_config} ---")
        except FileNotFoundError:
            print(f"Error: Predictor config file '{args.predictor_config}' not found.")
            sys.exit(1)

    # Load existing deployment config if doing selective deployment
    final_config = {}
    if args.models and os.path.exists(args.output):
        try:
            with open(args.output, 'r') as f:
                final_config = json.load(f)
            print(f"--- Loaded existing deployment config from {args.output} ---")
        except (json.JSONDecodeError, IOError) as e:
            print(f"--- Could not load existing config ({e}), starting fresh ---")

    print(f"--- Loaded {sum(len(v) for v in available_nodes.values())} nodes ---")

    # Filter models if --models specified
    models_to_deploy = None
    if args.models:
        models_to_deploy = [m.strip() for m in args.models.split(',')]
        print(f"--- Deploying only specified models: {models_to_deploy} ---")
        print(f"--- Existing models will be preserved in output config ---")
    else:
        print(f"--- Deploying all models ---")

    # Deployment mode
    if args.parallel:
        print(f"--- Using PARALLEL deployment (max {args.max_parallel_workers} workers) ---")
    else:
        print(f"--- Using SEQUENTIAL deployment (slower, for debugging) ---")

    # 2. Allocation & Deployment
    # Collect all deployment tasks for parallel execution
    deployment_tasks = []
    for model_key, details in config.items():
        # Skip if not in the list of models to deploy
        if models_to_deploy and model_key not in models_to_deploy:
            print(f"\nSkipping: {model_key} (not in deployment list)")
            continue

        # When --models is used with an existing deployment config, use the
        # already-allocated node_hosts instead of re-running allocation.
        # This avoids the bug where re-allocation grabs nodes belonging to other models.
        if models_to_deploy and model_key in final_config and 'node_hosts' in final_config[model_key]:
            existing_hosts = final_config[model_key]['node_hosts']
            print(f"\nRe-deploying: {model_key} to {len(existing_hosts)} existing hosts...")

            # Build node_type -> hosts mapping from existing config to get correct vllm_params
            if 'node_type' in details:
                for ntype, node_config in details['node_type'].items():
                    if isinstance(node_config, int):
                        vllm_params = {}
                    else:
                        vllm_params = {k: v for k, v in node_config.items() if k != 'count'}
                    # Find which existing hosts match this node type
                    ntype_hosts = [h for h in existing_hosts if ntype in h.split('@')[-1]]
                    if ntype_hosts:
                        _deploy_hosts(ntype_hosts, vllm_params, ntype_label=ntype)
            else:
                _deploy_hosts(existing_hosts, {})

            # Preserve existing config entry
            final_config[model_key] = final_config[model_key]
            continue

        print(f"\nProcessing: {model_key}...")

        # --- Allocation Logic ---
        new_entry = details.copy()
        assigned_hosts = []  # Accumulate all hosts for config output

        # --- SSH Deployment Logic (common fields) ---
        backend = details.get('backend', 'vllm')
        hf_name = details.get('hf_model_name')
        precision = details.get('precision', 'fp16')

        # Validate required fields before deployment
        if not hf_name:
            print(f"ERROR: Missing 'hf_model_name' for {model_key}")
            sys.exit(1)
        if backend not in ["vllm", "ollama"]:
            print(f"ERROR: Invalid backend '{backend}' for {model_key}. Must be 'vllm' or 'ollama'")
            sys.exit(1)

        def _deploy_hosts(hosts: List[str], vllm_params: Dict, ntype_label: str = ""):
            """Deploy model instances and predictors to a group of hosts with shared vllm_params."""
            label_suffix = f" [{ntype_label}]" if ntype_label else ""
            for host in hosts:
                # Deploy model instances
                if deploy_model_instances:
                    # First, cleanup existing processes
                    cleanup_cmds = get_cleanup_commands(backend)
                    if cleanup_cmds:
                        if args.parallel:
                            deployment_tasks.append((host, cleanup_cmds, f"Cleanup ({model_key}{label_suffix})"))
                        else:
                            run_ssh_cmd(host, cleanup_cmds, f"Cleanup ({model_key}{label_suffix})")

                    # Then deploy backend
                    if backend == "vllm":
                        if vllm_params:
                            print(f"  vLLM params for {ntype_label or 'default'}: {vllm_params}")
                        cmds = get_vllm_commands(hf_name, args.hf_token, precision, vllm_params,
                                                 fresh_deploy=args.fresh_deploy)
                        if args.parallel:
                            deployment_tasks.append((host, cmds, f"Deploy vLLM ({model_key}{label_suffix})"))
                        else:
                            run_ssh_cmd(host, cmds, f"Deploying vLLM ({model_key}{label_suffix})")
                    elif backend == "ollama":
                        print(f"  Ollama parallel requests: {args.ollama_num_parallel}")
                        cmds = get_ollama_commands(hf_name, num_parallel=args.ollama_num_parallel,
                                                   fresh_deploy=args.fresh_deploy)
                        if args.parallel:
                            deployment_tasks.append((host, cmds, f"Deploy Ollama ({model_key}{label_suffix})"))
                        else:
                            run_ssh_cmd(host, cmds, f"Deploying Ollama ({model_key}{label_suffix})")

                # Deploy predictors if requested
                if deploy_predictors_flag:
                    # Extract hostname from "user@hostname" format
                    hostname = host.split("@")[-1] if "@" in host else host
                    # Validate hostname exists in host_config
                    if hostname not in host_config:
                        print(f"ERROR: Hostname '{hostname}' not found in host_config ({args.host_config})")
                        print(f"Available hosts: {list(host_config.keys())}")
                        sys.exit(1)
                    # Get backend_port from host_config (single source of truth!)
                    backend_port = host_config[hostname]["backend_port"]
                    predictor_cmds = get_predictor_deployment_commands(
                        hostname=hostname,
                        backend_port=backend_port,
                        predictor_config=predictor_config,
                        host_config=host_config
                    )
                    if args.parallel:
                        deployment_tasks.append((host, predictor_cmds, f"Deploy Predictors ({model_key}{label_suffix})"))
                    else:
                        run_ssh_cmd(host, predictor_cmds, f"Deploying Predictors ({model_key}{label_suffix})")

        if 'node_type' in details:
            for ntype, node_config in details['node_type'].items():
                # Support both old format (just count) and new format (dict with count + params)
                if isinstance(node_config, int):
                    # Old format: "d8545": 2
                    count = node_config
                    vllm_params = {}
                else:
                    # New format: "d8545": {"count": 2, "gpu-memory-utilization": 0.95, ...}
                    count = node_config.get('count', 0)
                    # Extract vLLM parameters (everything except 'count')
                    vllm_params = {k: v for k, v in node_config.items() if k != 'count'}

                if (ntype not in available_nodes and count > 0) or len(available_nodes.get(ntype, [])) < count:
                    print(f"CRITICAL ERROR: Not enough nodes of type {ntype} for {model_key} (requires {count}, available {len(available_nodes.get(ntype, []))})")
                    sys.exit(1)

                if count == 0:
                    continue
                nodes = available_nodes[ntype][:count]
                # Remove used nodes from pool
                available_nodes[ntype] = available_nodes[ntype][count:]
                assigned_hosts.extend(nodes)

                # Deploy this group of nodes with their specific vllm_params
                _deploy_hosts(nodes, vllm_params, ntype_label=ntype)

            del new_entry['node_type']
            new_entry['node_hosts'] = assigned_hosts
        else:
            assigned_hosts = details.get('node_hosts', [])
            # Deploy with default (empty) vllm_params
            _deploy_hosts(assigned_hosts, {})

        final_config[model_key] = new_entry

    # Execute all deployment tasks in parallel (if parallel mode enabled)
    if args.parallel and deployment_tasks:
        success = deploy_parallel(deployment_tasks, max_workers=args.max_parallel_workers)
        if not success:
            print("\n⚠ WARNING: Some deployments failed. Check logs above.")
            print("Continuing with config generation and distribution...\n")

    # 3. Save Config
    with open(args.output, 'w') as f:
        json.dump(final_config, f, indent=2)
    print(f"\nDone! Final configuration saved to {args.output}")

    # 4. Distribute config to all nodes (so scheduler can access it)
    if args.distribute_config:
        print(f"\n--- Distributing config to remote nodes ---")
        all_hosts = []
        for hosts in available_nodes.values():
            all_hosts.extend(hosts)

        # Add back the hosts we used for deployment
        for model_key, details in final_config.items():
            if 'node_hosts' in details:
                all_hosts.extend(details['node_hosts'])

        # Remove duplicates
        all_hosts = list(set(all_hosts))

        if not all_hosts:
            print("Warning: No hosts found for config distribution")
        else:
            print(f"Distributing {args.output} to {len(all_hosts)} nodes...")
            for host in all_hosts:
                try:
                    # Create remote directory if needed and copy config
                    remote_path = f"~/Block/{args.output}"
                    remote_dir = os.path.dirname(remote_path)

                    scp_cmd = [
                        "scp",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "UserKnownHostsFile=/dev/null",
                        args.output,
                        f"{host}:{remote_path}"
                    ]

                    result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        print(f"  ✓ {host}")
                    else:
                        print(f"  ✗ {host}: {result.stderr.strip()}")
                except Exception as e:
                    print(f"  ✗ {host}: {str(e)}")

            print(f"Config distribution complete!")


if __name__ == "__main__":
    main()
