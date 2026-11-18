#!/bin/bash
# Start the PathSnap API for the evaluation (Linux/macOS).  Usage: scripts/run_api.sh [trips_folder] [port]
# The trips folder may be the folder with the trip sub-folders, or its parent (a wrapper folder is fine).
cd "$(dirname "$0")/.."
export PATHSNAP_EVAL_DIR="${1:-data/eval_trips}"
PORT="${2:-8000}"
echo "Serving trips from: $PATHSNAP_EVAL_DIR   on port $PORT"
python3 -m uvicorn src.api.app:app --host 0.0.0.0 --port "$PORT"
