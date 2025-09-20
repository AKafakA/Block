#!/bin/bash
# Remote simulator suite helper
#
# Splits the workflow into:
#   1) setup   – clone/pull repo, checkout branch, create venv, install deps
#   2) run_*   – launch experiment suite in the background (non-blocking)
#   3) tail_*  – tail the latest background run log
#
# Expected environment variables (with defaults):
#   NORMAL_SCALE_SIMULATION_HOST (default: wd312@caelum-104)
#   LARGE_SCALE_SIMULATION_HOST  (default: wd312@caelum-105)
# Optional overrides:
#   REMOTE_ROOT       (default: ~/Block)
#   PYTHON_BIN        (default: python3)
#   REPO_URL          (default: current repo origin)
#   REMOTE_BRANCH     (default: simulator)
#   REMOTE_EXTRA_ARGS (additional args passed to the suite script)

set -euo pipefail

usage() {
  cat >&2 <<EOF
Usage:
  $0 setup [qps32|large]
  $0 run_qps32
  $0 run_large
  $0 tail_qps32
  $0 tail_large

Env:
  NORMAL_SCALE_SIMULATION_HOST   Remote host for QPS=32 suite (default: wd312@caelum-104)
  LARGE_SCALE_SIMULATION_HOST    Remote host for large-scale suite (default: wd312@caelum-105)
  REMOTE_BRANCH                  Branch to checkout remotely (default: simulator)
  REMOTE_ROOT                    Remote repo root (default: ~/Block)
  REPO_URL                       Git URL (default: origin of local repo)
  REMOTE_EXTRA_ARGS              Extra args passed to run_* suite scripts
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

NORMAL_SCALE_SIMULATION_HOST=${NORMAL_SCALE_SIMULATION_HOST:-wd312@caelum-104}
LARGE_SCALE_SIMULATION_HOST=${LARGE_SCALE_SIMULATION_HOST:-wd312@caelum-214}

REMOTE_ROOT=${REMOTE_ROOT:-~/Block}
PYTHON_BIN=${PYTHON_BIN:-python3}
REPO_URL=${REPO_URL:-$(git config --get remote.origin.url)}
REMOTE_EXTRA_ARGS=${REMOTE_EXTRA_ARGS:-}
REMOTE_BRANCH=${REMOTE_BRANCH:-simulator}

ACTION=$1; shift || true

pick_host_and_suite() {
  local mode="$1"
  case "$mode" in
    qps32)
      HOST=${NORMAL_SCALE_SIMULATION_HOST}
      REMOTE_SUITE="bash simulation_analysis/run_qps32_suite.sh ${REMOTE_EXTRA_ARGS}"
      RUN_DIR_NAME="qps32"
      REMOTE_OUTPUT_DIR="${REMOTE_ROOT}/simulation_analysis/full_runs/qps32_remote"
      ;;
    large)
      HOST=${LARGE_SCALE_SIMULATION_HOST}
      REMOTE_SUITE="bash simulation_analysis/run_large_scale_suite.sh ${REMOTE_EXTRA_ARGS}"
      RUN_DIR_NAME="large"
      REMOTE_OUTPUT_DIR="${REMOTE_ROOT}/simulation_analysis/large_scale/remote"
      ;;
    *)
      echo "Unknown mode: $mode" >&2
      exit 1
      ;;
  esac
}

remote_setup() {
  pick_host_and_suite "$1"
  ssh "$HOST" "set -euo pipefail; \
    if [[ ! -d ${REMOTE_ROOT} ]]; then git clone ${REPO_URL} ${REMOTE_ROOT}; fi; \
    cd ${REMOTE_ROOT}; \
    git fetch --all --prune; \
    if git rev-parse --verify ${REMOTE_BRANCH} >/dev/null 2>&1; then \
      git checkout ${REMOTE_BRANCH}; \
    else \
      git checkout -b ${REMOTE_BRANCH} origin/${REMOTE_BRANCH} || git checkout ${REMOTE_BRANCH}; \
    fi; \
    git pull --ff-only origin ${REMOTE_BRANCH} || true; \
    ${PYTHON_BIN} -m pip install --user --upgrade pip setuptools wheel >/dev/null 2>&1 || true; \
    if ! ${PYTHON_BIN} -m pip install --user -r requirements-sim.txt >/dev/null 2>&1; then \
      echo '[remote_setup] requirements install failed; relaxing numpy pin' >&2; \
    fi"
}

remote_run_detached() {
  pick_host_and_suite "$1"
  # Timestamped log under simulation_analysis/remote_runs/<mode>/
  local ts
  ts=$(date -u +%Y%m%d_%H%M%SZ)
  local log_dir="${REMOTE_ROOT}/simulation_analysis/remote_runs/${RUN_DIR_NAME}"
  local log_file="${log_dir}/run_${ts}.log"

  # Launch in background with nohup; persist PID and print log path
  ssh "$HOST" "set -euo pipefail; \
    mkdir -p ${log_dir}; \
    cd ${REMOTE_ROOT}; \
    pyver=\$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo 3.10); \
    site=${REMOTE_ROOT}/.pyuser/lib/python\${pyver}/site-packages; \
    nohup bash -lc 'export PYTHONUSERBASE=${REMOTE_ROOT}/.pyuser; export PYTHONPATH=${REMOTE_ROOT}:'"\${site}"':\$PYTHONPATH; ${REMOTE_SUITE}' \
      > ${log_file} 2>&1 & echo \$! > ${log_dir}/last.pid; echo ${log_file}"
}

remote_tail_latest() {
  pick_host_and_suite "$1"
  local log_dir="${REMOTE_ROOT}/simulation_analysis/remote_runs/${RUN_DIR_NAME}"
  ssh -t "$HOST" "set -euo pipefail; \
    mkdir -p ${log_dir}; \
    latest=\$(ls -1t ${log_dir}/run_*.log 2>/dev/null | head -n1); \
    if [[ -z \"\${latest}\" ]]; then echo 'No logs found in ' ${log_dir} >&2; exit 1; fi; \
    echo 'Tailing: ' \"\${latest}\"; \
    tail -f \"\${latest}\""
}

# Collect logs and outputs from the remote to a local folder.
# Usage: LOCAL_ROOT=simulation_analysis/remote_collected/qps32 collect qps32
remote_collect() {
  pick_host_and_suite "$1"
  local mode="$1"
  local local_root="${LOCAL_ROOT:-simulation_analysis/remote_collected/${mode}}"
  local remote_log_dir="${REMOTE_ROOT}/simulation_analysis/remote_runs/${RUN_DIR_NAME}"

  mkdir -p "${local_root}/logs" "${local_root}/outputs"

  if command -v rsync >/dev/null 2>&1; then
    rsync -av --rsh=ssh "${HOST}:${remote_log_dir}/" "${local_root}/logs/"
    rsync -av --rsh=ssh "${HOST}:${REMOTE_OUTPUT_DIR}/" "${local_root}/outputs/" || true
  else
    echo "rsync not found; falling back to scp (slower)" >&2
    scp -r "${HOST}:${remote_log_dir}" "${local_root}/" || true
    scp -r "${HOST}:${REMOTE_OUTPUT_DIR}" "${local_root}/outputs" || true
  fi

  echo "Collected to ${local_root}"
}

case "$ACTION" in
  setup)
    mode=${1:-qps32}
    remote_setup "$mode"
    echo "Setup completed on [$mode] host."
    ;;
  run_qps32)
    path=$(remote_run_detached qps32)
    echo "Remote run started (QPS=32). Log: $path"
    ;;
  run_large)
    path=$(remote_run_detached large)
    echo "Remote run started (large-scale). Log: $path"
    ;;
  one_click_qps32)
    remote_setup qps32
    path=$(remote_run_detached qps32)
    echo "One-click: setup+run (QPS=32). Log: $path"
    ;;
  one_click_large)
    remote_setup large
    path=$(remote_run_detached large)
    echo "One-click: setup+run (large-scale). Log: $path"
    ;;
  one_click_both)
    remote_setup qps32
    remote_setup large
    path1=$(remote_run_detached qps32)
    path2=$(remote_run_detached large)
    echo "One-click: setup+run both. Logs: QPS32=$path1 LARGE=$path2"
    ;;
  tail_qps32)
    remote_tail_latest qps32
    ;;
  tail_large)
    remote_tail_latest large
    ;;
  collect_qps32)
    remote_collect qps32
    ;;
  collect_large)
    remote_collect large
    ;;
  collect_all)
    remote_collect qps32
    remote_collect large
    ;;
  *)
    usage
    exit 1
    ;;
esac
