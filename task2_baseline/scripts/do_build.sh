#!/usr/bin/env bash

set -euo pipefail  # stricter safety

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
TASK_ROOT="$(realpath "${SCRIPT_DIR}/..")"
DOCKER_IMAGE_TAG="chimera-task2-classification"

echo "[INFO] Building Docker image for CHIMERA Task 2..."
echo "  Project root:     ${PROJECT_ROOT}"
echo "  Dockerfile path:  ${TASK_ROOT}/Dockerfile"
echo "  Image tag:        ${DOCKER_IMAGE_TAG}"

docker build \
  --platform=linux/amd64 \
  --tag "${DOCKER_IMAGE_TAG}" \
  --file "${TASK_ROOT}/Dockerfile" \
  "${PROJECT_ROOT}"


