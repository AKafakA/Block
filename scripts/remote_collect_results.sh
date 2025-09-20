#!/bin/bash
# Fetch experiment outputs from remote hosts into the local simulation_analysis tree.
#
# Example:
#   HOSTS_FILE=hosts.txt ./scripts/remote_collect_results.sh \
#       --remote-path Block/simulation_analysis/full_runs/qps32_remote \
#       --local-root simulation_analysis/remote_runs/qps32

set -euo pipefail

HOSTS_FILE=${HOSTS_FILE:-hosts.txt}
REMOTE_PATH=""
LOCAL_ROOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote-path)
            REMOTE_PATH=$2
            shift 2
            ;;
        --local-root)
            LOCAL_ROOT=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${REMOTE_PATH}" || -z "${LOCAL_ROOT}" ]]; then
    echo "Usage: $0 --remote-path <dir_on_remote> --local-root <local_dir>" >&2
    exit 1
fi

if [[ ! -f "${HOSTS_FILE}" ]]; then
    echo "Hosts file ${HOSTS_FILE} not found" >&2
    exit 1
fi

mkdir -p "${LOCAL_ROOT}"

parallel-rsync -a -h "${HOSTS_FILE}" \
    ":${REMOTE_PATH}/" "${LOCAL_ROOT}/{#}/"

echo "Results copied to ${LOCAL_ROOT}" 

