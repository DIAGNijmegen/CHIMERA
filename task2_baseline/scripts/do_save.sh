#!/bin/bash
set -e
echo "💾 Saving Docker image..."
docker save task2_baseline | gzip > task2_baseline.tar.gz
echo "✅ Saved as task2_baseline.tar.gz"
