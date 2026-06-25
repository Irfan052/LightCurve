@echo off
title AstroExo Pipeline Startup

echo ===================================================
echo   AstroExo Hackathon Pipeline Setup
echo ===================================================

echo [1/4] Installing Python backend dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing python dependencies.
    pause
    exit /b %errorlevel%
)

echo [2/4] Installing Node.js frontend dependencies...
cd frontend
call npm install
if %errorlevel% neq 0 (
    echo Error installing frontend dependencies.
    cd ..
    pause
    exit /b %errorlevel%
)
cd ..

echo [3/4] Starting FastAPI Backend on port 8000...
start "AstroExo Backend" cmd /k "python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000"

echo [4/4] Starting Vite Frontend on port 5173...
start "AstroExo Frontend" cmd /k "cd frontend && npm run dev"

echo ===================================================
echo   AstroExo is running!
echo   Frontend: http://localhost:5173
echo   Backend API: http://localhost:8000/docs
echo ===================================================
echo Keep this window open. Close the popup command windows to shut down.
pause
