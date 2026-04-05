#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-ghcr.mirrorify.net/inclusionai/areal-runtime:v1.0.2-sglang}"
CONTAINER_NAME="${CONTAINER_NAME:-asearcher-areal}"
NETWORK_MODE="${NETWORK_MODE:-host}"
CONTAINER_PYTHONPATH="/workspace/ASearcher:/workspace/ASearcher/AReaL"
CODEX_HOME_DIR="${CODEX_HOME_DIR:-${ROOT_DIR}/.codex-home}"

resolve_proxy_var() {
  local var_name="$1"
  local current_value="${!var_name:-}"
  if [[ -n "${current_value}" ]]; then
    printf '%s' "${current_value}"
    return
  fi

  local sudo_user_name="${SUDO_USER:-}"
  if [[ -n "${sudo_user_name}" ]]; then
    local user_home
    user_home="$(eval echo "~${sudo_user_name}")"
    local bashrc_path="${user_home}/.bashrc"
    if [[ -f "${bashrc_path}" ]]; then
      local line
      line="$(grep -E "^export ${var_name}=" "${bashrc_path}" | tail -n 1 || true)"
      if [[ -n "${line}" ]]; then
        printf '%s' "${line#*=}" | sed 's/^"//; s/"$//'
        return
      fi
    fi
  fi

  printf ''
}

HTTP_PROXY_VALUE="$(resolve_proxy_var HTTP_PROXY)"
HTTPS_PROXY_VALUE="$(resolve_proxy_var HTTPS_PROXY)"
ALL_PROXY_VALUE="$(resolve_proxy_var ALL_PROXY)"
NO_PROXY_VALUE="$(resolve_proxy_var NO_PROXY)"
http_proxy_value="$(resolve_proxy_var http_proxy)"
https_proxy_value="$(resolve_proxy_var https_proxy)"
all_proxy_value="$(resolve_proxy_var all_proxy)"
no_proxy_value="$(resolve_proxy_var no_proxy)"

if [[ -z "${HTTP_PROXY_VALUE}" && -n "${http_proxy_value}" ]]; then
  HTTP_PROXY_VALUE="${http_proxy_value}"
fi
if [[ -z "${HTTPS_PROXY_VALUE}" && -n "${https_proxy_value}" ]]; then
  HTTPS_PROXY_VALUE="${https_proxy_value}"
fi
if [[ -z "${ALL_PROXY_VALUE}" && -n "${all_proxy_value}" ]]; then
  ALL_PROXY_VALUE="${all_proxy_value}"
fi
if [[ -z "${NO_PROXY_VALUE}" && -n "${no_proxy_value}" ]]; then
  NO_PROXY_VALUE="${no_proxy_value}"
fi
if [[ -z "${http_proxy_value}" && -n "${HTTP_PROXY_VALUE}" ]]; then
  http_proxy_value="${HTTP_PROXY_VALUE}"
fi
if [[ -z "${https_proxy_value}" && -n "${HTTPS_PROXY_VALUE}" ]]; then
  https_proxy_value="${HTTPS_PROXY_VALUE}"
fi
if [[ -z "${all_proxy_value}" && -n "${ALL_PROXY_VALUE}" ]]; then
  all_proxy_value="${ALL_PROXY_VALUE}"
fi
if [[ -z "${no_proxy_value}" && -n "${NO_PROXY_VALUE}" ]]; then
  no_proxy_value="${NO_PROXY_VALUE}"
fi

mkdir -p "${CODEX_HOME_DIR}"

docker run --gpus all --rm -it \
  --name "${CONTAINER_NAME}" \
  --network "${NETWORK_MODE}" \
  --shm-size 32g \
  -e HTTP_PROXY="${HTTP_PROXY_VALUE}" \
  -e HTTPS_PROXY="${HTTPS_PROXY_VALUE}" \
  -e ALL_PROXY="${ALL_PROXY_VALUE}" \
  -e NO_PROXY="${NO_PROXY_VALUE}" \
  -e http_proxy="${http_proxy_value}" \
  -e https_proxy="${https_proxy_value}" \
  -e all_proxy="${all_proxy_value}" \
  -e no_proxy="${no_proxy_value}" \
  -e NPM_CONFIG_PROXY="${http_proxy_value}" \
  -e NPM_CONFIG_HTTPS_PROXY="${https_proxy_value}" \
  -e PYTHONPATH="${CONTAINER_PYTHONPATH}" \
  -v "${ROOT_DIR}:/workspace/ASearcher" \
  -v "${CODEX_HOME_DIR}:/root/.codex" \
  -w /workspace/ASearcher \
  "${IMAGE_TAG}" \
  bash
