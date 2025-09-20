#!/bin/bash
# Launch experiment suites on dedicated remote hosts.
#
# Expected environment variables:
#   NORMAL_SCALE_SIMULATION_HOST (e.g., user@caelum-104)
#   LARGE_SCALE_SIMULATION_HOST  (e.g., user@caelum-105)
# Optional overrides:
#   REMOTE_ROOT (default: ~/Block)
#   VENV_DIR    (default: $REMOTE_ROOT/venv)
#   PYTHON_BIN  (default: python3)
#   REPO_URL    (default: current repo origin)
#   REMOTE_EXTRA_ARGS (additional args passed to the suite script)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <run_qps32|run_large>" >&2
    exit 1
fi

NORMAL_SCALE_SIMULATION_HOST="wd312@caelum-104"
LARGE_SCALE_SIMULATION_HOST="wd312@caelum-105"

ACTION=$1

REMOTE_ROOT=${REMOTE_ROOT:-~/Block}
VENV_DIR=${VENV_DIR:-$REMOTE_ROOT/venv}
PYTHON_BIN=${PYTHON_BIN:-python3}
REPO_URL=${REPO_URL:-$(git config --get remote.origin.url)}
REMOTE_EXTRA_ARGS=${REMOTE_EXTRA_ARGS:-}

case "${ACTION}" in
    run_qps32)
        HOST=${NORMAL_SCALE_SIMULATION_HOST:-}
        REMOTE_SUITE="python simulation_analysis/run_qps32_suite.sh ${REMOTE_EXTRA_ARGS}"
        ;;
    run_large)
        HOST=${LARGE_SCALE_SIMULATION_HOST:-}
        REMOTE_SUITE="python simulation_analysis/run_large_scale_suite.sh ${REMOTE_EXTRA_ARGS}"
        ;;
    *)
        echo "Unknown action: ${ACTION}" >&2
        exit 1
        ;;
esac

if [[ -z "${HOST}" ]]; then
    echo "Missing host variable for ${ACTION}. Set NORMAL_SCALE_SIMULATION_HOST or LARGE_SCALE_SIMULATION_HOST." >&2
    exit 1
fi

ssh "${HOST}" "set -euo pipefail; \
    if [[ ! -d ${REMOTE_ROOT} ]]; then git clone ${REPO_URL} ${REMOTE_ROOT}; fi; \
    cd ${REMOTE_ROOT}; \
    git pull --ff-only; \
    if [[ ! -d ${VENV_DIR} ]]; then ${PYTHON_BIN} -m venv ${VENV_DIR}; fi; \
    source ${VENV_DIR}/bin/activate; \
    pip install --upgrade pip >/dev/null; \
    pip install -r requirements.txt >/dev/null; \
    PYTHONPATH=${REMOTE_ROOT} ${REMOTE_SUITE}" || {
        echo "Remote execution failed" >&2
        exit 1
    }

