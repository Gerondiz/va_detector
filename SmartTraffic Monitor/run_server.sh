#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PYTHONPATH"
while true; do
    echo "[$(date)] Starting SmartTraffic server..."
    python3 -m uvicorn traffic_backend.main:app --host 0.0.0.0 --port 8001 --log-level warning
    EXIT_CODE=$?
    echo "[$(date)] Server exited with code $EXIT_CODE, restarting in 3s..."
    sleep 3
done
