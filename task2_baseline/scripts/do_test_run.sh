#!/bin/bash
set -e

# ================================================
# CHIMERA Task 2 - Local Docker Test Script
# ================================================
# This script will:
# 1. Build or use your Task 2 Docker container
# 2. Mount local input/output/model directories
# 3. Run inference using the GC-style pipeline
# ================================================

# --- SETTINGS ---
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"

# Path to your local test input data (must match GC interface format)
INPUT_DIR="${PROJECT_ROOT}/test/input"

# Path where predictions will be saved
OUTPUT_DIR="${PROJECT_ROOT}/test/output"

# Path to your trained Task 2 model weights (local machine)
# This folder must contain:
#   best_model.pth
#   config.json
#   clinical_processor.pkl (optional)
MODEL_DIR="/home/maryam/my_task2_model_weights"

# Docker image name
DOCKER_IMAGE="chimera-task2-prediction"

# --- CHECKS ---
if [ ! -d "${INPUT_DIR}" ]; then
    echo "[ERROR] INPUT_DIR not found: ${INPUT_DIR}"
    exit 1
fi
if [ ! -d "${MODEL_DIR}" ]; then
    echo "[ERROR] MODEL_DIR not found: ${MODEL_DIR}"
    exit 1
fi

# --- CLEAN OUTPUT DIR ---
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# --- RUN CONTAINER ---
echo "[INFO] Running Task 2 container..."
docker run --rm \
    --runtime=nvidia \
    --gpus all \
    --volume "${INPUT_DIR}":/input:ro \
    --volume "${OUTPUT_DIR}":/output:rw \
    --volume "${MODEL_DIR}":/opt/ml/model:ro \
    ${DOCKER_IMAGE}

echo "[INFO] Inference complete. Results saved to: ${OUTPUT_DIR}"
