import argparse
import os
import xml.etree.ElementTree as ET
from collections import OrderedDict
import json

def generate_config(ip_address, predictor_port, backend_port, tensor_parallel_size=1):
    config = {
        "ip_address": ip_address,
        "predictor_ports": predictor_port,
        "backend_port": backend_port,
        "tensor_parallel_size": tensor_parallel_size
    }
    return config


def generate_configs(num_predictors, backend_port,
                     manifest_path,
                     config_output_path,
                     user_name,
                     tensor_parallel_size=1,
                     cluster_type="a30"):
    """Generate cluster configuration from CloudLab manifest.

    Args:
        num_predictors: Number of predictor workers per node
        backend_port: vLLM backend port (default 8000)
        manifest_path: Path to CloudLab manifest XML
        config_output_path: Output directory for config files
        user_name: SSH username for CloudLab nodes
        tensor_parallel_size: GPUs per model instance (1 for A30, 4 for A100-70B)
        cluster_type: "a30" (single GPU nodes) or "a100" (multi-GPU nodes)
    """
    # Generate predictor ports: 8100, 8300, 8400, 8500, ...
    # Skip 8200 as it's used by global scheduler client API
    predictor_ports = []
    port = backend_port + 100  # Start at 8100
    while len(predictor_ports) < num_predictors:
        if port != backend_port + 200:  # Skip 8200
            predictor_ports.append(port)
        port += 100

    tree = ET.parse(manifest_path)
    # get root element
    nodes = {}
    root = tree.getroot()

    for child in root:
        if "node" in child.tag:
            node_info = {}
            node_name = child.get("client_id")
            nodes[node_name] = node_info
            for subchild in child:
                if "host" in subchild.tag:
                    ip_address = subchild.get("ipv4")
                    node_info["ip_adresses"] = ip_address
                if "services" in subchild.tag:
                    host_name = subchild[0].get("hostname")
                    node_info["hostname"] = host_name

    nodes = OrderedDict(sorted(nodes.items()))

    # Output file names based on cluster type
    if cluster_type == "a100":
        host_config_files = os.path.join(config_output_path, "a100_host_configs.json")
        host_files = os.path.join(config_output_path, "a100_hosts")
    else:
        host_config_files = os.path.join(config_output_path, "host_configs.json")
        host_files = os.path.join(config_output_path, "hosts")

    host_names = []
    with open(host_config_files, "w+") as f, open(host_files, "w+") as n :
        configs = {}
        for node in nodes:
            node_info = nodes[node]
            host_names.append(user_name + "@" + node_info["hostname"])
            config = generate_config(
                node_info["ip_adresses"],
                predictor_ports[:num_predictors],
                backend_port,
                tensor_parallel_size
            )
            configs[node_info["hostname"]] = config
        json.dump(configs, f, sort_keys=True, indent=4)
        for host in host_names:
            n.write(host + "\n")

    print(f"Generated config for {cluster_type} cluster:")
    print(f"  - Host config: {host_config_files}")
    print(f"  - Hosts file: {host_files}")
    print(f"  - Nodes: {len(nodes)}")
    print(f"  - Tensor parallel size: {tensor_parallel_size}")
    print(f"  - Predictors per node: {num_predictors}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Block cluster configuration from CloudLab manifest")
    parser.add_argument("--user_name", type=str, required=True,
                        help="SSH username for CloudLab nodes")
    parser.add_argument("--num_predictors", type=int, default=16,
                        help="Number of predictor workers per node")
    parser.add_argument("--backend_port", type=int, default=8000,
                        help="vLLM backend port number")
    parser.add_argument("--host_config_files", type=str, default="block/config",
                        help="Output directory for config files")
    parser.add_argument("--manifest_path", type=str, default="block/cl_manifest.xml",
                        help="Path to CloudLab manifest XML file")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="GPUs per model instance (1 for A30/7B, 4 for A100/70B)")
    parser.add_argument("--cluster_type", type=str, default="a30", choices=["a30", "a100"],
                        help="Cluster type: a30 (single GPU) or a100 (multi-GPU)")
    args = parser.parse_args()

    generate_configs(
        args.num_predictors,
        args.backend_port,
        args.manifest_path,
        args.host_config_files,
        args.user_name,
        args.tensor_parallel_size,
        args.cluster_type
    )