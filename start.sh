#!/bin/bash
# PropelAI Startup Script
# Runs the FastAPI backend with the web UI

echo "🚀 Starting PropelAI..."
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Load environment variables from .env if it exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "📋 Loading environment from .env..."
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check dependencies
echo "📦 Checking dependencies..."
pip install -q fastapi uvicorn python-multipart openpyxl python-docx pypdf tensorlake 2>/dev/null

# Check Tensorlake configuration
if [ -n "$TENSORLAKE_API_KEY" ]; then
    echo "✅ Tensorlake API key configured (Gemini 3 OCR enabled)"
else
    echo "⚠️  Tensorlake API key not set (using pypdf fallback)"
    echo "   Set TENSORLAKE_API_KEY in .env for enhanced PDF extraction"
fi

# Start server
echo ""
echo "✅ Starting PropelAI API server..."
echo "   Open http://localhost:8000 in your browser"
echo ""
echo "   Press Ctrl+C to stop"
echo ""

cd "$SCRIPT_DIR"
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
