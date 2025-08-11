#!/usr/bin/env bash

# Stop at first error
set -e

# SCRIPT_DIR will be .../CHIMERA/task1_baseline/scripts
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Go up two directories to get the project root (.../CHIMERA)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")

# Set default container name
DOCKER_IMAGE_TAG="example-algorithm-prostate-cancer-biochemical-recurrence-prediction"

echo "=+= (Re)building the container with the correct context..."
# This sources the updated build script from its location
source "${SCRIPT_DIR}/do_build.sh"

# Get the build information from the Docker image tag
build_timestamp=$( docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG")

if [ -z "$build_timestamp" ]; then
    echo "Error: Failed to retrieve build information for container $DOCKER_IMAGE_TAG"
    exit 1
fi

# Format the build information to remove special characters
formatted_build_info=$(echo "$build_timestamp" | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g')

# Set the output filename. The file will be saved in the scripts folder.
output_filename="${SCRIPT_DIR}/${DOCKER_IMAGE_TAG}_${formatted_build_info}.tar.gz"

# Save the Docker-container image and gzip it
echo "==+=="
echo "Saving the container image as ${output_filename}. This can take a while."
docker save "$DOCKER_IMAGE_TAG" | gzip -c > "$output_filename"
echo "Container image saved as ${output_filename}"
echo "==+=="

# Create the optional model tarball if the model directory exists
MODEL_DIR="${PROJECT_ROOT}/common/model"
if [ -d "$MODEL_DIR" ]; then
    echo "==+=="
    output_tarball_name="${SCRIPT_DIR}/model.tar.gz"
    echo "Creating the optional model tarball as ${output_tarball_name}."
    # Create a tarball of the contents of the model directory
    tar -czf "$output_tarball_name" -C "$MODEL_DIR" .
    echo "(Optional) Uploadable model tarball was created as ${output_tarball_name}"
    echo "==+=="
else
    echo "==+=="
    echo "Skipping model tarball creation: Directory ${MODEL_DIR} not found."
    echo "==+=="
fi