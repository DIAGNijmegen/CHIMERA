#!/bin/bash
set -e

# =========================================================
# CHIMERA Task 2 - Local Docker Test Script
# =========================================================
# This simulates how Grand Challenge will run your container
# =========================================================

# --- Paths ---
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"

# Local test input (must follow GC interface structure)
INPUT_DIR="${PROJECT_ROOT}/test/input"

# Local output folder (will be cleared each run)
OUTPUT_DIR="${PROJECT_ROOT}/test/output"

# Your trained model weights (local folder)
# Must contain:
#   best_model.pth
#   config.json
#   clinical_data.csv
#   pathology_features/CASE_ID.pt files
#   clinical_processor.pkl (optional)
MODEL_DIR="/home/maryam/my_task2_model_weights"

# Docker image name (built with do_build.sh)
DOCKER_IMAGE="chimera-task2-prediction"

# --- Checks ---
if [ ! -d "${INPUT_DIR}" ]; then
    echo "[ERROR] INPUT_DIR not found: ${INPUT_DIR}"
    exit 1
fi
if [ ! -d "${MODEL_DIR}" ]; then
    echo "[ERROR] MODEL_DIR not found: ${MODEL_DIR}"
    exit 1
fi

# --- Prepare output folder ---
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# --- Run container ---
echo "[INFO] Running Task 2 container..."
docker run --rm \
    --runtime=nvidia \
    --gpus all \
    --volume "${INPUT_DIR}":/input:ro \
    --volume "${OUTPUT_DIR}":/output:rw \
    --volume "${MODEL_DIR}":/opt/ml/model:ro \
    ${DOCKER_IMAGE}

echo "[INFO] Inference complete. Predictions saved to: ${OUTPUT_DIR}"
