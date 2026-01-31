#!/bin/bash

# Acoustic Anomaly Detection - Quick Start Script for macOS
# This script sets up and runs the application

set -e  # Exit on error

PROJECT_DIR="/Users/akanksha/Downloads/final dataset-4"
VENV_DIR="$PROJECT_DIR/venv_mac"

echo "🚀 Starting Acoustic Anomaly Detection Application..."
echo ""

# Navigate to project directory
cd "$PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv_mac
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv_mac/bin/activate

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models/*.joblib 2>/dev/null)" ]; then
    echo "⚠️  Warning: No model files found in models/ directory"
    echo "   Make sure you have trained the models first"
fi

# Start the server
echo ""
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "   Press Ctrl+C to stop the server"
echo ""

# Start server in background
uvicorn inference_api:app --reload --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Wait a moment for server to start
sleep 3

# Check if server started successfully
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Server is running!"
    echo ""
    echo "📱 Opening frontend in browser..."
    open "http://localhost:8000"
    echo ""
    echo "🎉 Application is ready!"
    echo "   - Frontend: http://localhost:8000 (opens automatically)"
    echo "   - API: http://localhost:8000"
    echo "   - API Docs: http://localhost:8000/docs"
    echo "   - Health Check: http://localhost:8000/health"
    echo ""
    echo "Press Ctrl+C to stop the server"
    
    # Wait for user to stop the server
    wait $SERVER_PID
else
    echo "❌ Server failed to start. Please check the error messages above."
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

