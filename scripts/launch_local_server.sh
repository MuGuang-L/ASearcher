#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8766}"
SAVE_ADDR_DIR="${2:-${RAG_SERVER_ADDR_DIR:-/tmp/areal/rag_server_addrs}}"
WIKI2018_WORK_DIR="${WIKI2018_WORK_DIR:-${3:-}}"
HOST="${HOST:-127.0.0.1}"

# This server is only meant to be reached locally.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
export no_proxy="${no_proxy:-${NO_PROXY}}"

if [[ -z "${WIKI2018_WORK_DIR}" ]]; then
  echo "Usage: bash scripts/launch_local_server.sh <port> <save_addr_dir> <wiki2018_work_dir>" >&2
  echo "   or: WIKI2018_WORK_DIR=/path/to/wiki2018 bash scripts/launch_local_server.sh" >&2
  exit 1
fi

INDEX_FILE="${INDEX_FILE:-${WIKI2018_WORK_DIR}/e5.index/e5_Flat.index}"
CORPUS_FILE="${CORPUS_FILE:-${WIKI2018_WORK_DIR}/wiki_corpus.jsonl}"
PAGES_FILE="${PAGES_FILE:-${WIKI2018_WORK_DIR}/wiki_webpages.jsonl}"
TOPK="${TOPK:-3}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat__e5-base-v2}"
FAISS_GPU="${FAISS_GPU:-0}"

mkdir -p "${SAVE_ADDR_DIR}"

ARGS=(
  --index_path "${INDEX_FILE}"
  --corpus_path "${CORPUS_FILE}"
  --pages_path "${PAGES_FILE}"
  --topk "${TOPK}"
  --retriever_name "${RETRIEVER_NAME}"
  --retriever_model "${RETRIEVER_MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --save-address-to "${SAVE_ADDR_DIR}"
)

if [[ "${FAISS_GPU}" == "1" ]]; then
  ARGS+=(--faiss_gpu)
fi

python3 "${ROOT_DIR}/tools/local_retrieval_server.py" \
  "${ARGS[@]}"
