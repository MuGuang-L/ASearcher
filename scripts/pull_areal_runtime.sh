#!/bin/bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-ghcr.mirrorify.net/inclusionai/areal-runtime:v1.0.2-sglang}"

echo "Pulling ${IMAGE_TAG}"
docker pull "${IMAGE_TAG}"
