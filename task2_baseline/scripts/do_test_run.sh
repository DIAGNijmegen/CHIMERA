#!/usr/bin/env bash

# Stop at first error
set -e

# SCRIPT_DIR will be .../CHIMERA/task2_baseline/scripts
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Go up one directory to get the task root (.../CHIMERA/task2_baseline)
TASK_ROOT=$(realpath "${SCRIPT_DIR}/..")
# Go up two directories to get the project root (.../CHIMERA)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")

# Task 2 classification container tag
DOCKER_IMAGE_TAG="chimera-task2-classification"
DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

# Define INPUT and OUTPUT directories relative to the TASK_ROOT
INPUT_DIR="${TASK_ROOT}/test/input"
OUTPUT_DIR="${TASK_ROOT}/test/output"
# Define MODEL directory relative to the PROJECT_ROOT
MODEL_DIR=MODEL_DIR="/path/to/my_task2_model_weights"


echo "=+= (Re)building the container with the correct context..."
# This sources the updated build script from its location
source "${SCRIPT_DIR}/do_build.sh"

cleanup() {
    echo "=+= Cleaning permissions ..."
    docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      "$DOCKER_IMAGE_TAG" \
      -c "chmod -R -f o+rwX /output/* || true"

    docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null
}

# Allow read access to inputs and model dir
chmod -R -f o+rX "$INPUT_DIR" "$MODEL_DIR"

# Clean output directories
for i in {0..9}; do
  if [ -d "${OUTPUT_DIR}/interface_${i}" ]; then
    chmod -f o+rwX "${OUTPUT_DIR}/interface_${i}"
    docker run --rm --quiet --platform=linux/amd64 \
      --volume "${OUTPUT_DIR}/interface_${i}":/output \
      --entrypoint /bin/sh "$DOCKER_IMAGE_TAG" \
      -c "rm -rf /output/* || true"
  else
    mkdir -p -m o+rwX "${OUTPUT_DIR}/interface_${i}"
  fi
done

docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null

trap cleanup EXIT

run_docker_forward_pass() {
    local interface_dir="$1"

    echo "=+= Doing a forward pass on ${interface_dir}"

    docker run --rm \
        --platform=linux/amd64 \
        --network none \
        --gpus all \
        --volume "${INPUT_DIR}/${interface_dir}":/input:ro \
        --volume "${OUTPUT_DIR}/${interface_dir}":/output \
        --volume "$DOCKER_NOOP_VOLUME":/tmp \
        --volume "${MODEL_DIR}":/opt/ml/model:ro \
        "$DOCKER_IMAGE_TAG"

    echo "=+= Wrote results to ${OUTPUT_DIR}/${interface_dir}"
}

# Run inference for all 10 GC interfaces (0 to 9)
for i in {0..9}; do
  run_docker_forward_pass "interface_${i}"
done

echo "=+= Save this image for uploading via ./do_save.sh"
