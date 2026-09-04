@echo off
title IBVAP - Tactical Border Surveillance C2
echo ====================================================================
echo   IBVAP - Intelligent Border Video Analytics Platform
echo   Sashastra Seema Bal (SSB) Command Grid
echo ====================================================================
echo.

:: Check and automatically free port 8000 if occupied by an older process
for /f "tokens=5" %%a in ('netstat -aon ^| findstr /r ":8000.*LISTENING"') do (
    echo [INFO] Freeing occupied port 8000 (killing previous PID %%a)...
    taskkill /F /PID %%a >nul 2>&1
    timeout /t 1 /nobreak >nul
)

echo Starting backend server on http://127.0.0.1:8000 ...
echo Opening tactical dashboard in your default browser...
echo.

cd /d "%~dp0backend"

:: Wait 2 seconds then launch browser directly to explicit HTTP
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"

:: Start Uvicorn
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Server exited with code %ERRORLEVEL%
    pause
)
