#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
while pgrep -f "run_bp.py dev2" > /dev/null; do sleep 20; done
ls results/bp_dist_dev2.parquet || { echo "dev2 output missing"; exit 1; }
./scripts/post_dev2.sh
