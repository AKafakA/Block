import argparse
import os
import xml.etree.ElementTree as ET
from collections import OrderedDict
import json

manifest_path = "block/cl_manifest.xml"
config_output_path = "block/config"
user_name = ""


def generate_config(ip_address, predictor_port, backend_port):
    config = {
        "ip_address": ip_address,
        "predictor_ports": predictor_port,
        "backend_port": backend_port
    }
    return config


predictor_ports = [8100, 8300, 8400, 8500, 8600, 8700, 8800, 8900, 9000, 9100, 9200, 9300, 9400, 9500, 9600, 9700]
backend_port = 8000

tree = ET.parse(manifest_path)
num_schedulers = 1
num_predictors = 16
# get root element
nodes = {}
root = tree.getroot()
upload = True

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
    j = 0
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
    parser.add_argument("--user_name", type=str, required=True, help="User name to ssh to cloud lab hostnames")
    args = parser.parse_args()
    user_name = args.user_name
