#!/bin/bash
set -e

INPUT_DIR=/path/to/local/input
OUTPUT_DIR=/path/to/local/output

echo "🚀 Running container..."
docker run --rm \
    -v $INPUT_DIR:/input \
    -v $OUTPUT_DIR:/output \
    task2_baseline
echo "✅ Finished. Check $OUTPUT_DIR/predictions.csv"
