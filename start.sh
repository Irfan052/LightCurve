#!/bin/bash
# AstroExo Hackathon Startup Script (Linux/Mac)

echo "🚀 Starting AstroExo Pipeline Setup..."

# 1. Install Backend Dependencies
echo "📦 Installing Python backend dependencies..."
pip install -r requirements.txt

# 2. Install Frontend Dependencies
echo "📦 Installing Node.js frontend dependencies..."
cd frontend
npm install
cd ..

# 3. Start the Backend Server (Background)
echo "🌐 Starting FastAPI Backend on port 8000..."
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# 4. Start the Frontend Server
echo "🌐 Starting Vite Frontend on port 5173..."
cd frontend
npm run dev &
FRONTEND_PID=$!

# Handle exit
cleanup() {
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "✅ AstroExo is running!"
echo "➡️ Frontend: http://localhost:5173"
echo "➡️ Backend API: http://localhost:8000/docs"
echo "Press Ctrl+C to stop both servers."

wait