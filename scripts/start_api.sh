#!/bin/bash
# Start the PathSnap API (Linux/macOS).  Usage: scripts/start_api.sh [test|eval|/path/to/trips] [port]
cd "$(dirname "$0")/.."
MODE="${1:-test}"; PORT="${2:-8000}"
case "$MODE" in test) TRIPS="data/student_test"; [ -d "$TRIPS" ] || TRIPS="data/Student folder/student_test";; eval) TRIPS="data/eval_trips";; *) TRIPS="$MODE";; esac
[ -d "$TRIPS" ] || { echo "Folder not found: $TRIPS"; exit 1; }
echo "Mode: $MODE | trips folder: $TRIPS | sub-folders: $(ls "$TRIPS" | wc -l) | port: $PORT"
echo "Give Laptop 2 this API address:  http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT"
export PATHSNAP_EVAL_DIR="$TRIPS"
python3 -m uvicorn src.api.app:app --host 0.0.0.0 --port "$PORT"
