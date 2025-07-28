#!/usr/bin/env bash

# Stop at first error
set -e

# SCRIPT_DIR will be .../CHIMERA/task1_baseline/scripts
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Go up one directory to get the task root (.../CHIMERA/task1_baseline)
TASK_ROOT=$(realpath "${SCRIPT_DIR}/..")
# Go up two directories to get the project root (.../CHIMERA)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")


DOCKER_IMAGE_TAG="example-algorithm-prostate-cancer-biochemical-recurrence-prediction"
DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

# Define INPUT and OUTPUT directories relative to the TASK_ROOT
INPUT_DIR="${TASK_ROOT}/test/input"
OUTPUT_DIR="${TASK_ROOT}/test/output"
# Define MODEL directory relative to the PROJECT_ROOT
MODEL_DIR="${PROJECT_ROOT}/common/model"

echo "=+= (Re)building the container with the correct context..."
# This sources the updated build script from its location
source "${SCRIPT_DIR}/do_build.sh"

cleanup() {
    echo "=+= Cleaning permissions ..."
    # Ensure permissions are set correctly on the output
    docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      "$DOCKER_IMAGE_TAG" \
      -c "chmod -R -f o+rwX /output/* || true"

    # Ensure volume is removed
    docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null
}

# This allows for the Docker user to read from the correct model and input locations
chmod -R -f o+rX "$INPUT_DIR" "$MODEL_DIR"

# The rest of this section cleans up output directories.
# It is verbose but functionally correct. No path changes needed here
# as it uses the corrected $OUTPUT_DIR variable.
if [ -d "${OUTPUT_DIR}/interface_0" ]; then
  chmod -f o+rwX "${OUTPUT_DIR}/interface_0"
  docker run --rm --quiet --platform=linux/amd64 --volume "${OUTPUT_DIR}/interface_0":/output --entrypoint /bin/sh "$DOCKER_IMAGE_TAG" -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwX "${OUTPUT_DIR}/interface_0"
fi
if [ -d "${OUTPUT_DIR}/interface_1" ]; then
  chmod -f o+rwX "${OUTPUT_DIR}/interface_1"
  docker run --rm --quiet --platform=linux/amd64 --volume "${OUTPUT_DIR}/interface_1":/output --entrypoint /bin/sh "$DOCKER_IMAGE_TAG" -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwX "${OUTPUT_DIR}/interface_1"
fi
if [ -d "${OUTPUT_DIR}/interface_2" ]; then
  chmod -f o+rwX "${OUTPUT_DIR}/interface_2"
  docker run --rm --quiet --platform=linux/amd64 --volume "${OUTPUT_DIR}/interface_2":/output --entrypoint /bin/sh "$DOCKER_IMAGE_TAG" -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwX "${OUTPUT_DIR}/interface_2"
fi
if [ -d "${OUTPUT_DIR}/interface_3" ]; then
  chmod -f o+rwX "${OUTPUT_DIR}/interface_3"
  docker run --rm --quiet --platform=linux/amd64 --volume "${OUTPUT_DIR}/interface_3":/output --entrypoint /bin/sh "$DOCKER_IMAGE_TAG" -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwX "${OUTPUT_DIR}/interface_3"
fi
if [ -d "${OUTPUT_DIR}/interface_4" ]; then
  chmod -f o+rwX "${OUTPUT_DIR}/interface_4"
  docker run --rm --quiet --platform=linux/amd64 --volume "${OUTPUT_DIR}/interface_4":/output --entrypoint /bin/sh "$DOCKER_IMAGE_TAG" -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwX "${OUTPUT_DIR}/interface_4"
fi
if [ -d "${OUTPUT_DIR}/interface_5" ]; then
  chmod -f o+rwX "${OUTPUT_DIR}/interface_5"
  docker run --rm --quiet --platform=linux/amd64 --volume "${OUTPUT_DIR}/interface_5":/output --entrypoint /bin/sh "$DOCKER_IMAGE_TAG" -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwX "${OUTPUT_DIR}/interface_5"
fi
if [ -d "${OUTPUT_DIR}/interface_6" ]; then
  chmod -f o+rwX "${OUTPUT_DIR}/interface_6"
  docker run --rm --quiet --platform=linux/amd64 --volume "${OUTPUT_DIR}/interface_6":/output --entrypoint /bin/sh "$DOCKER_IMAGE_TAG" -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwX "${OUTPUT_DIR}/interface_6"
fi
if [ -d "${OUTPUT_DIR}/interface_7" ]; then
  chmod -f o+rwX "${OUTPUT_DIR}/interface_7"
  docker run --rm --quiet --platform=linux/amd64 --volume "${OUTPUT_DIR}/interface_7":/output --entrypoint /bin/sh "$DOCKER_IMAGE_TAG" -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwX "${OUTPUT_DIR}/interface_7"
fi
if [ -d "${OUTPUT_DIR}/interface_8" ]; then
  chmod -f o+rwX "${OUTPUT_DIR}/interface_8"
  docker run --rm --quiet --platform=linux/amd64 --volume "${OUTPUT_DIR}/interface_8":/output --entrypoint /bin/sh "$DOCKER_IMAGE_TAG" -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwX "${OUTPUT_DIR}/interface_8"
fi
if [ -d "${OUTPUT_DIR}/interface_9" ]; then
  chmod -f o+rwX "${OUTPUT_DIR}/interface_9"
  docker run --rm --quiet --platform=linux/amd64 --volume "${OUTPUT_DIR}/interface_9":/output --entrypoint /bin/sh "$DOCKER_IMAGE_TAG" -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwX "${OUTPUT_DIR}/interface_9"
fi


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


run_docker_forward_pass "interface_0"
run_docker_forward_pass "interface_1"
run_docker_forward_pass "interface_2"
run_docker_forward_pass "interface_3"
run_docker_forward_pass "interface_4"
run_docker_forward_pass "interface_5"
run_docker_forward_pass "interface_6"
run_docker_forward_pass "interface_7"
run_docker_forward_pass "interface_8"
run_docker_forward_pass "interface_9"

echo "=+= Save this image for uploading via ./do_save.sh"