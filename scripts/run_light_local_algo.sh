#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ALGO="${1:-}"
BASE_CONFIG="${2:-${ROOT_DIR}/ASearcher/configs/asearcher_local_light_qwen3.yaml}"
DATASET_PATH="${3:-}"
MODEL_PATH="${4:-/workspace/ASearcher/models/Qwen3-1.7B}"
EXPERIMENT_NAME="${5:-asearcher-light-qwen3}"
TRIAL_NAME="${6:-run1}"

if [[ -z "${ALGO}" || -z "${DATASET_PATH}" ]]; then
  echo "Usage: $0 <algo> [base_config] <dataset_path> [model_path] [experiment_name] [trial_name]" >&2
  echo "Example: $0 grpo ${ROOT_DIR}/ASearcher/configs/asearcher_local_light_qwen3.yaml ${ROOT_DIR}/data/train_data/ASearcher-Base-35k.sample_10000.jsonl /workspace/ASearcher/models/Qwen3-1.7B asearcher-light-qwen3 run-grpo" >&2
  exit 1
fi

RENDERED_CONFIG="/tmp/areal/light-config-${ALGO}-${TRIAL_NAME}.yaml"
python3 "${ROOT_DIR}/scripts/render_light_algo_config.py" --base "${BASE_CONFIG}" --algo "${ALGO}" --out "${RENDERED_CONFIG}"

exec bash "${ROOT_DIR}/scripts/run_light_local.sh" "${RENDERED_CONFIG}" "${DATASET_PATH}" "${MODEL_PATH}" "${EXPERIMENT_NAME}" "${TRIAL_NAME}"
