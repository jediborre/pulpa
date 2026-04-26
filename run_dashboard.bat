@echo off
setlocal
title Pulpa Betting System

echo ==================================================
echo        PULPA BETTING DASHBOARD - STARTUP
echo ==================================================
echo.

:: Detectar .venv
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] No se encontro el entorno virtual en .venv
    echo Por favor, asegurate de tener la carpeta .venv en la raiz.
    pause
    exit /b
)

:: 1. Iniciar API Backend en una nueva ventana
echo [+] Iniciando API Python (FastAPI)...
start "Pulpa API Backend" cmd /c "call .venv\Scripts\activate && python api.py"

:: Esperar un momento a que el API arranque
timeout /t 3 /nobreak > nul

:: 2. Iniciar Dashboard Frontend
echo [+] Iniciando Dashboard (Vite/React)...
cd v13_dashboard
npm run dev

endlocal
pause
