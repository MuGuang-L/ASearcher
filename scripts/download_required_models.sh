#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${1:-${ROOT_DIR}/models}"

QWEN_REPO="${QWEN_REPO:-Qwen/Qwen3-1.7B}"
E5_REPO="${E5_REPO:-intfloat/e5-base-v2}"

QWEN_DIR="${MODEL_ROOT}/Qwen3-1.7B"
E5_DIR="${MODEL_ROOT}/e5-base-v2"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd hf

mkdir -p "${MODEL_ROOT}"

export HF_HUB_ENABLE_HF_TRANSFER=1

echo "[1/2] Downloading ${QWEN_REPO} -> ${QWEN_DIR}"
hf download "${QWEN_REPO}" \
  --local-dir "${QWEN_DIR}"

echo "[2/2] Downloading ${E5_REPO} -> ${E5_DIR}"
hf download "${E5_REPO}" \
  --local-dir "${E5_DIR}"

cat <<EOF

Download complete.

Model paths:
  Qwen:
    ${QWEN_DIR}
  E5:
    ${E5_DIR}

Suggested next commands:
  RETRIEVER_MODEL=${E5_DIR} WIKI2018_WORK_DIR=${ROOT_DIR}/data/wiki2018_smoke bash scripts/build_index.sh

  RETRIEVER_MODEL=${E5_DIR} WIKI2018_WORK_DIR=${ROOT_DIR}/data/wiki2018_smoke bash scripts/launch_local_server.sh 8766 /tmp/areal/rag_server_addrs

  bash scripts/run_light_local.sh \\
    ${ROOT_DIR}/ASearcher/configs/asearcher_local_light_qwen3.yaml \\
    ${ROOT_DIR}/data/train_data/ASearcher-Base-35k.sample_10000.jsonl \\
    ${QWEN_DIR}
EOF
