#!/usr/bin/env bash
# Combined script to run backend (FastAPI) and frontend (Streamlit) for Pneumonia Detection

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Optional: activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating venv..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating .venv..."
    source .venv/bin/activate
fi

# Cleanup: kill backend when this script exits (e.g. Ctrl+C)
BACKEND_PID=""
cleanup() {
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo ""
        echo "Stopping backend (PID $BACKEND_PID)..."
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Start backend in background
echo "Starting backend (FastAPI) on http://localhost:7779 ..."
python main.py &
BACKEND_PID=$!

# Give backend time to start (and load model)
echo "Waiting for backend to be ready..."
for i in {1..30}; do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:7779/health 2>/dev/null | grep -q 200; then
        echo "Backend is up."
        break
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "Backend process exited. Check logs."
        exit 1
    fi
    sleep 1
done

# Start frontend (foreground so you see logs and Ctrl+C stops this script → cleanup runs)
echo "Starting frontend (Streamlit) on http://localhost:8501 ..."
streamlit run frontend/app.py
