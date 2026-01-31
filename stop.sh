#!/bin/bash

# Stop script for Acoustic Anomaly Detection Application
# This script stops the FastAPI server running on port 8000

echo "🛑 Stopping Acoustic Anomaly Detection server..."

# Method 1: Kill process on port 8000
if lsof -ti:8000 > /dev/null 2>&1; then
    echo "   Found server running on port 8000"
    lsof -ti:8000 | xargs kill -9
    echo "✅ Server stopped successfully"
else
    echo "   No server found running on port 8000"
fi

# Method 2: Also kill any uvicorn processes related to inference_api
if pgrep -f "uvicorn inference_api" > /dev/null; then
    echo "   Found uvicorn processes"
    pkill -f "uvicorn inference_api"
    echo "✅ All uvicorn processes stopped"
fi

# Verify
sleep 1
if ! lsof -ti:8000 > /dev/null 2>&1; then
    echo ""
    echo "✅ Server is completely stopped"
    echo "   Port 8000 is now free"
else
    echo ""
    echo "⚠️  Warning: Some processes may still be running"
    echo "   Try running: lsof -ti:8000 | xargs kill -9"
fi


