#!/bin/bash
set -e
echo "🚀 Building Task 2 classification container..."
docker build -t task2_baseline .
echo "✅ Build complete."
