#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-${ROOT_DIR}/data}"
TRAIN_LINES="${TRAIN_LINES:-10000}"
WIKI_LINES="${WIKI_LINES:-10000}"

TRAIN_DIR="${DATA_ROOT}/train_data"
WIKI_DIR="${DATA_ROOT}/wiki2018_smoke"

TRAIN_FULL="${TRAIN_DIR}/ASearcher-Base-35k.jsonl"
TRAIN_SAMPLE="${TRAIN_DIR}/ASearcher-Base-35k.sample_${TRAIN_LINES}.jsonl"
CORPUS_FILE="${WIKI_DIR}/wiki_corpus.jsonl"
PAGES_FILE="${WIKI_DIR}/wiki_webpages.jsonl"

if [[ ! -f "${TRAIN_FULL}" ]]; then
  echo "Missing train file: ${TRAIN_FULL}" >&2
  exit 1
fi

if [[ ! -f "${CORPUS_FILE}" ]]; then
  echo "Missing corpus file: ${CORPUS_FILE}" >&2
  exit 1
fi

mkdir -p "${TRAIN_DIR}" "${WIKI_DIR}"

echo "[1/2] Creating ${TRAIN_LINES}-line training sample..."
python3 - "${TRAIN_FULL}" "${TRAIN_SAMPLE}" "${TRAIN_LINES}" <<'PY'
import sys

src, dst, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        if i >= limit:
            break
        fout.write(line)
PY

echo "[2/2] Materializing a lightweight pages file from corpus for local access..."
python3 - "${CORPUS_FILE}" "${PAGES_FILE}" "${WIKI_LINES}" <<'PY'
import json
import sys

src, dst, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
count = 0
with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line in fin:
        if count >= limit:
            break
        obj = json.loads(line)
        fout.write(json.dumps({
            "url": obj.get("url"),
            "title": obj.get("wikipedia_title", ""),
            "contents": obj.get("contents", ""),
        }, ensure_ascii=False) + "\n")
        count += 1
PY

cat <<EOF

Prepared local smoke data:

  Train sample:
    ${TRAIN_SAMPLE}
  Corpus:
    ${CORPUS_FILE}
  Pages:
    ${PAGES_FILE}

Next commands:
  WIKI2018_WORK_DIR=${WIKI_DIR} bash scripts/build_index.sh
  WIKI2018_WORK_DIR=${WIKI_DIR} bash scripts/launch_local_server.sh 8766 /tmp/areal/rag_server_addrs
  bash scripts/run_light_local.sh \\
    ${ROOT_DIR}/ASearcher/configs/asearcher_local_light_qwen3.yaml \\
    ${TRAIN_SAMPLE}
EOF
