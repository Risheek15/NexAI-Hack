#!/usr/bin/env bash
# Maintenance Copilot - One-Click Launcher for Linux/macOS
set -e

echo "=========================================================="
echo "  Starting Maintenance Copilot (FastAPI + Next.js UI)    "
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtualenv if present
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 1. Start FastAPI Backend on port 8000
echo "[1/2] Starting FastAPI Backend on http://127.0.0.1:8000..."
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# 2. Start Next.js Frontend on port 3000
echo "[2/2] Starting Next.js Frontend on http://localhost:3000..."
cd "$SCRIPT_DIR/frontend"
npm run start &
FRONTEND_PID=$!

cd "$SCRIPT_DIR"

cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "Servers stopped."
}
trap cleanup EXIT INT TERM

echo ""
echo "=========================================================="
echo "  Maintenance Copilot is Live!                            "
echo "  - Frontend UI:  http://localhost:3000                   "
echo "  - Backend API:  http://127.0.0.1:8000                   "
echo "  - API Swagger:  http://127.0.0.1:8000/docs              "
echo "=========================================================="
echo "Press Ctrl+C to terminate both servers."
echo ""

wait
