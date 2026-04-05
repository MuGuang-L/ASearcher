#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WIKI2018_WORK_DIR="${WIKI2018_WORK_DIR:-${1:-}}"

if [[ -z "${WIKI2018_WORK_DIR}" ]]; then
  echo "Usage: WIKI2018_WORK_DIR=/path/to/wiki2018 bash scripts/build_index.sh" >&2
  echo "   or: bash scripts/build_index.sh /path/to/wiki2018" >&2
  exit 1
fi

CORPUS_FILE="${CORPUS_FILE:-${WIKI2018_WORK_DIR}/wiki_corpus.jsonl}"
SAVE_DIR="${SAVE_DIR:-${WIKI2018_WORK_DIR}/e5.index}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat__e5-base-v2}"
FAISS_TYPE="${FAISS_TYPE:-Flat}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MAX_LENGTH="${MAX_LENGTH:-256}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

python3 "${ROOT_DIR}/utils/index_builder.py" \
  --retrieval_method "${RETRIEVER_NAME}" \
  --model_path "${RETRIEVER_MODEL}" \
  --corpus_path "${CORPUS_FILE}" \
  --save_dir "${SAVE_DIR}" \
  --use_fp16 \
  --max_length "${MAX_LENGTH}" \
  --batch_size "${BATCH_SIZE}" \
  --pooling_method mean \
  --faiss_type "${FAISS_TYPE}" \
  --save_embedding
