@echo off
setlocal
title SISTEMA PULPA - Inicia Todo

echo ==================================================
echo         SISTEMA INTEGRADO PULPA (ALL-IN-ONE)
echo ==================================================
echo.

:: Detectar .venv
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] No se encontro el entorno virtual en .venv
    pause
    exit /b
)

:: 1. Iniciar API Backend
echo [+] Iniciando Backend API...
start "Pulpa API Backend" cmd /c "call .venv\Scripts\activate && python api.py"

:: 2. Iniciar Bot de Telegram
echo [+] Iniciando Telegram Bot...
start "Pulpa Telegram Bot" cmd /c "call .venv\Scripts\activate && python match/telegram_bot.py"

:: 3. Iniciar Frontend Dashboard
echo [+] Iniciando Dashboard Web...
timeout /t 2 /nobreak > nul
cd v13_dashboard
npm run dev

endlocal
pause
