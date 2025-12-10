import json
import argparse
import subprocess
import sys
import time
from typing import Dict, List

# --- Configuration ---
# Success signatures to look for in the logs
LOG_PATTERNS = {
    "vllm": {
        "file": "~/vllm/vllm_server.log",
        "success_marker": "Uvicorn running on"
    },
    "ollama": {
        "file": "~/ollama/ollama_server.log",
        "success_marker": "Listening on"
    }
}


def check_server_log(host: str, backend: str) -> bool:
    """
    Phase 1: Scans the remote log file for a specific 'success' string.
    Returns True if the server seems ready.
    """
    print(f"  [1/2] Scanning logs on {host}...", end=" ", flush=True)

    config = LOG_PATTERNS.get(backend)
    if not config:
        print("skipped (unknown backend).")
        return False

    # Command: grep the log file. If found, exit 0. If not, exit 1.
    # We use 'tail' to avoid grepping massive history if the file is old,
    # but scanning the whole file is safer for initialization checks.
    check_cmd = f"grep -q '{config['success_marker']}' {config['file']}"

    ssh_params = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        host, check_cmd
    ]

    try:
        result = subprocess.run(ssh_params, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            print("✅ Log OK (Server Started).")
            return True
        else:
            print("⚠️  Log Warning (Marker not found).")
            # Optional: Print the last few lines to see what's wrong
            tail_cmd = f"tail -n 3 {config['file']}"
            tail_res = subprocess.run(ssh_params[:-1] + [tail_cmd], capture_output=True, text=True, timeout=5)
            print(f"        Last log lines: {tail_res.stdout.strip()}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ SSH Timeout during log check.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def check_inference_api(host: str, backend: str, model_name: str) -> bool:
    """
    Phase 2: Sends a real inference request via Curl.
    """
    print(f"  [2/2] Verifying API...", end=" ", flush=True)

    if backend == "vllm":
        payload = json.dumps({
            "model": model_name,
            "prompt": "Test.",
            "max_tokens": 1
        }).replace('"', '\\"')
        cmd = f"curl -s -X POST http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d \"{payload}\""
        success_key = "choices"

    elif backend == "ollama":
        ollama_tag = model_name.split('/')[-1].lower().replace("-", ":")
        payload = json.dumps({
            "model": ollama_tag,
            "prompt": "Test.",
            "stream": False
        }).replace('"', '\\"')
        cmd = f"curl -s -X POST http://localhost:11434/api/generate -d \"{payload}\""
        success_key = "response"

    else:
        return False

    ssh_params = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        host, cmd
    ]

    try:
        # Give it 30s because the first inference might trigger compilation/loading
        result = subprocess.run(ssh_params, capture_output=True, text=True, timeout=30)

        if result.returncode == 0 and success_key in result.stdout.lower():
            print("✅ API Responding.")
            return True
        else:
            print(f"❌ API Fail. Response: {result.stdout[:50]}...")
            return False

    except subprocess.TimeoutExpired:
        print("❌ API Timeout.")
        return False


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Cara Feasibility")
    parser.add_argument("--config", default="block/config/cara/model_deployment.json",
                        help="Path to deployment config")
    parser.add_argument("--output", default="block/config/cara/verified_hosts.json", help="Output file")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            deployment = json.load(f)
    except FileNotFoundError:
        print("Error: Config file not found.")
        sys.exit(1)

    successful_hosts = {}

    print("--- Starting Two-Stage Verification ---\n")

    for model_key, details in deployment.items():
        backend = details.get('backend', 'vllm')
        model_name = details.get('hf_model_name')
        hosts = details.get('node_hosts', [])

        verified_list = []

        print(f"Checking Group: {model_key} ({backend})")

        for host in hosts:
            # Stage 1: Log Check
            if check_server_log(host, backend):
                # Stage 2: Inference Check (Only if logs are good)
                if check_inference_api(host, backend, model_name):
                    verified_list.append(host)
            else:
                print(f"  Skipping API check for {host} due to log failure.")

        if verified_list:
            successful_hosts[model_key] = verified_list
        print("-" * 40)

    print(f"\nVerification Complete. {sum(len(v) for v in successful_hosts.values())} nodes fully operational.")

    with open(args.output, 'w') as f:
        json.dump(successful_hosts, f, indent=2)


if __name__ == "__main__":
    main()