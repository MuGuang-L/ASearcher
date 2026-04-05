#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-${ROOT_DIR}/data}"
TRAIN_SAMPLE_LINES="${TRAIN_SAMPLE_LINES:-3500}"
LOCAL_KNOWLEDGE_LINES="${LOCAL_KNOWLEDGE_LINES:-10000}"

TEST_DIR="${DATA_ROOT}/test_data"
TRAIN_DIR="${DATA_ROOT}/train_data"
WIKI_DIR="${DATA_ROOT}/wiki2018_smoke"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd hf
require_cmd curl
require_cmd python3

mkdir -p "${TEST_DIR}" "${TRAIN_DIR}" "${WIKI_DIR}"

echo "[1/4] Downloading ASearcher test data (small, download full set)..."
hf download inclusionAI/ASearcher-test-data \
  --repo-type dataset \
  --local-dir "${TEST_DIR}" \
  --quiet

echo "[2/4] Downloading ASearcher base training data..."
hf download inclusionAI/ASearcher-train-data \
  ASearcher-Base-35k.jsonl \
  --repo-type dataset \
  --local-dir "${TRAIN_DIR}" \
  --quiet

TRAIN_FULL="${TRAIN_DIR}/ASearcher-Base-35k.jsonl"
TRAIN_SAMPLE="${TRAIN_DIR}/ASearcher-Base-35k.sample_${TRAIN_SAMPLE_LINES}.jsonl"

echo "[3/4] Creating a small training sample: ${TRAIN_SAMPLE_LINES} lines..."
python3 - "${TRAIN_FULL}" "${TRAIN_SAMPLE}" "${TRAIN_SAMPLE_LINES}" <<'PY'
import sys

src, dst, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        if i >= limit:
            break
        fout.write(line)
PY

echo "[4/4] Streaming a smoke-test subset of local RAG knowledge..."
CORPUS_URL="https://huggingface.co/datasets/inclusionAI/ASearcher-Local-Knowledge/resolve/main/wiki_corpus.jsonl"
PAGES_URL="https://huggingface.co/datasets/inclusionAI/ASearcher-Local-Knowledge/resolve/main/wiki_webpages.jsonl"

CORPUS_SAMPLE="${WIKI_DIR}/wiki_corpus.jsonl"
PAGES_SAMPLE="${WIKI_DIR}/wiki_webpages.jsonl"

if [[ ! -s "${CORPUS_SAMPLE}" ]]; then
  curl -L "${CORPUS_URL}" | sed -n "1,${LOCAL_KNOWLEDGE_LINES}p" > "${CORPUS_SAMPLE}"
fi

if [[ ! -s "${PAGES_SAMPLE}" ]]; then
  curl -L "${PAGES_URL}" | sed -n "1,${LOCAL_KNOWLEDGE_LINES}p" > "${PAGES_SAMPLE}"
fi

cat <<EOF

Download complete.

Data root:
  ${DATA_ROOT}

Useful paths:
  Test data dir:
    ${TEST_DIR}
  Full train file:
    ${TRAIN_FULL}
  Small train sample:
    ${TRAIN_SAMPLE}
  Smoke local knowledge dir:
    ${WIKI_DIR}

Next:
  WIKI2018_WORK_DIR=${WIKI_DIR} bash scripts/build_index.sh
  WIKI2018_WORK_DIR=${WIKI_DIR} bash scripts/launch_local_server.sh 8766 /tmp/areal/rag_server_addrs
  bash scripts/run_light_local.sh \\
    ${ROOT_DIR}/ASearcher/configs/asearcher_local_light_qwen3.yaml \\
    ${TRAIN_SAMPLE}
EOF
