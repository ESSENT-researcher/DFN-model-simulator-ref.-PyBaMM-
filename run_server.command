#!/bin/bash
# DFN Model Simulator — Server Launcher
# Double-click this file to start the server and open the web interface.

cd "$(dirname "$0")"

# Kill any existing process on port 5001
existing=$(lsof -ti :5001)
if [ -n "$existing" ]; then
    echo "기존 서버 종료 중 (PID: $existing)..."
    kill "$existing" 2>/dev/null
    sleep 1
fi

# Start server in background and save PID
python3 run_server.py &
SERVER_PID=$!

# When terminal closes, kill the server
trap "echo '서버 종료 중...'; kill $SERVER_PID 2>/dev/null; exit" SIGINT SIGTERM SIGHUP EXIT

wait $SERVER_PID
