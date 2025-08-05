#!/usr/bin/env bash

# Stop at first error
set -e

# SCRIPT_DIR will be .../CHIMERA/task2_baseline/scripts
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Go up two directories to get the project root (.../CHIMERA)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
# Go up one directory to get the task root (.../CHIMERA/task2_baseline)
TASK_ROOT=$(realpath "${SCRIPT_DIR}/..")

# Change the image tag for Task 2
DOCKER_IMAGE_TAG="chimera-task2-classification"

echo "Project Root (Build Context): ${PROJECT_ROOT}"
echo "Dockerfile Location: ${TASK_ROOT}/Dockerfile"

# The final argument ("$PROJECT_ROOT") sets the build context.
# The -f flag explicitly points to the Dockerfile's location within that context.
docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG" \
  --file "${TASK_ROOT}/Dockerfile" \
  "$PROJECT_ROOT" 2>&1

