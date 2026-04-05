#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AREAL_DIR="${ROOT_DIR}/AReaL"

CONFIG_PATH="${1:-${ROOT_DIR}/ASearcher/configs/asearcher_local_light_qwen3.yaml}"
DATASET_PATH="${2:-}"
MODEL_PATH="${3:-Qwen/Qwen3-1.7B}"
EXPERIMENT_NAME="${4:-asearcher-light-local}"
TRIAL_NAME="${5:-run1}"

if [[ -z "${DATASET_PATH}" ]]; then
  echo "Usage: $0 <config_path> <dataset_path> [model_path] [experiment_name] [trial_name]" >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}:${AREAL_DIR}:${PYTHONPATH:-}"
export RAG_SERVER_ADDR_DIR="${RAG_SERVER_ADDR_DIR:-/tmp/areal/rag_server_addrs}"

# Local lightweight runs only talk to on-box services and local model files.
# Clearing proxy envs prevents localhost / container IP traffic from being
# accidentally routed through SSH-forwarded proxies.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
export no_proxy="${no_proxy:-${NO_PROXY}}"

cd "${AREAL_DIR}"

python3 -m areal.infra.launcher.local "${ROOT_DIR}/ASearcher/train/asearcher_light.py" \
  --config "${CONFIG_PATH}" \
  actor.path="${MODEL_PATH}" \
  train_dataset.path="${DATASET_PATH}" \
  experiment_name="${EXPERIMENT_NAME}" \
  trial_name="${TRIAL_NAME}"
