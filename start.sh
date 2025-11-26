#!/bin/bash
# PropelAI Startup Script
# Runs the FastAPI backend with the web UI

echo "🚀 Starting PropelAI..."
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check dependencies
echo "📦 Checking dependencies..."
pip install -q fastapi uvicorn python-multipart openpyxl python-docx pypdf 2>/dev/null

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Start server
echo ""
echo "✅ Starting PropelAI API server..."
echo "   Open http://localhost:8000 in your browser"
echo ""
echo "   Press Ctrl+C to stop"
echo ""

cd "$SCRIPT_DIR"
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
