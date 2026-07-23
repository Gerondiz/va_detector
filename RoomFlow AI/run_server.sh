#!/bin/bash
cd "$(dirname "$0")"
while true; do
    echo "[$(date)] Starting server..."
    python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --log-level warning
    EXIT_CODE=$?
    echo "[$(date)] Server exited with code $EXIT_CODE, restarting in 3s..."
    sleep 3
done
