import argparse
import os
import xml.etree.ElementTree as ET
from collections import OrderedDict
import json

# manifest_path = "block/cl_manifest.xml"
# config_output_path = "block/config"
# user_name = ""


def generate_config(ip_address, predictor_port, backend_port):
    config = {
        "ip_address": ip_address,
        "predictor_ports": predictor_port,
        "backend_port": backend_port
    }
    return config


def generate_configs(num_predictors, backend_port,
                     manifest_path,
                     config_output_path,
                     user_name):
    predictor_ports = []
    for i in range(num_predictors):
        predictor_ports.append(backend_port + 100 * i)

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
    host_config_files = os.path.join(config_output_path, "host_configs.json")
    host_files = os.path.join(config_output_path, "hosts")

    host_names = []
    with open(host_config_files, "w+") as f, open(host_files, "w+") as n:
        configs = {}
        for node in nodes:
            node_info = nodes[node]
            host_names.append(user_name + node_info["hostname"])
            config = generate_config(node_info["ip_adresses"], predictor_ports[:num_predictors], backend_port)
            configs[node_info["hostname"]] = config
        json.dump(configs, f, sort_keys=True, indent=4)
        for host in host_names:
            n.write(host + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_name", type=str, help="User name to ssh to cloud lab hostnames",
                        default="asdwb")
    parser.add_argument("--num_predictors", type=int, default=16, help="Number of predictor nodes to use")
    parser.add_argument("--backend_port", type=int, default=8000, help="Backend port number")
    parser.add_argument("--host_config_files", type=str, help="Path to output host config files",
                        default="block/config")
    parser.add_argument("--manifest_path", type=str, help="Path to cloud lab manifest xml file",
                        default="block/cl_manifest.xml")
    args = parser.parse_args()
    generate_configs(args.num_predictors, args.backend_port,
                     args.manifest_path,
                     args.host_config_files,
                     args.user_name)
