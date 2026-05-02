@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

:: ── Verificar .venv ───────────────────────────────
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Entorno virtual no encontrado. Ejecuta instalar.bat primero.
    pause
    exit /b 1
)

:MENU
cls
echo.
echo  ██████╗ ██╗   ██╗██╗     ██████╗  █████╗
echo  ██╔══██╗██║   ██║██║     ██╔══██╗██╔══██╗
echo  ██████╔╝██║   ██║██║     ██████╔╝███████║
echo  ██╔═══╝ ██║   ██║██║     ██╔═══╝ ██╔══██║
echo  ██║     ╚██████╔╝███████╗██║     ██║  ██║
echo  ╚═╝      ╚═════╝ ╚══════╝╚═╝     ╚═╝  ╚═╝
echo.
echo ==================================================
echo              MENU PRINCIPAL
echo ==================================================
echo.
echo   1) Correr Bot de Telegram
echo   2) Correr API Backend
echo   3) Correr Dashboard (API + Frontend)
echo   4) Correr Todo  (Bot + API + Dashboard)
echo   5) Traer fecha nueva  (option 15 de cli.py)
echo   6) Entrenar modelo V2
echo   7) Entrenar modelo V6
echo   8) Entrenar V2 + V6  (en orden)
echo   9) Instalar / actualizar dependencias
echo   10) V6.2: Solo entrenar
echo   11) V6.2: Solo reporte Q4 ROI
echo   12) V6.2: Entrenar + reporte
echo   0) Salir
echo.
set /p OPT="  Selecciona: "

if "%OPT%"=="0" goto FIN
if "%OPT%"=="1" goto BOT
if "%OPT%"=="2" goto API
if "%OPT%"=="3" goto DASHBOARD
if "%OPT%"=="4" goto TODO
if "%OPT%"=="5" goto FETCH_DATE
if "%OPT%"=="6" goto TRAIN_V2
if "%OPT%"=="7" goto TRAIN_V6
if "%OPT%"=="8" goto TRAIN_ALL
if "%OPT%"=="9" goto INSTALAR
if "%OPT%"=="10" goto TRAIN_REPORT_ONLY
if "%OPT%"=="11" goto REPORT_ONLY
if "%OPT%"=="12" goto TRAIN_AND_REPORT

echo [ERROR] Opcion invalida.
timeout /t 2 /nobreak >nul
goto MENU

:: ─────────────────────────────────────────────────
:BOT
cls
echo [+] Iniciando Telegram Bot...
start "Pulpa - Telegram Bot" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && python match\telegram_bot.py"
goto MENU

:: ─────────────────────────────────────────────────
:API
cls
echo [+] Iniciando API Backend...
start "Pulpa - API Backend" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && python api.py"
goto MENU

:: ─────────────────────────────────────────────────
:DASHBOARD
cls
echo [+] Iniciando API Backend...
start "Pulpa - API Backend" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && python api.py"
timeout /t 3 /nobreak >nul
echo [+] Iniciando Dashboard (Vite)...
start "Pulpa - Dashboard" cmd /k "cd /d %~dp0\v13_dashboard && npm run dev"
goto MENU

:: ─────────────────────────────────────────────────
:TODO
cls
echo [+] Iniciando Bot + API + Dashboard...
start "Pulpa - Telegram Bot"  cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && python match\telegram_bot.py"
start "Pulpa - API Backend"   cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && python api.py"
timeout /t 3 /nobreak >nul
start "Pulpa - Dashboard"     cmd /k "cd /d %~dp0\v13_dashboard && npm run dev"
goto MENU

:: ─────────────────────────────────────────────────
:FETCH_DATE
cls
echo [+] Traer fecha nueva (selector directo de fechas faltantes)...
echo.
call .venv\Scripts\activate && python match\cli.py fetch-date-menu
pause
goto MENU

:: ─────────────────────────────────────────────────
:TRAIN_V2
cls
echo [+] Entrenando modelo V2...
call .venv\Scripts\activate
python match\training\train_q3_q4_models_v2.py
if errorlevel 1 (
    echo [ERROR] Entrenamiento V2 fallo.
) else (
    echo [OK] V2 entrenado correctamente.
)
pause
goto MENU

:: ─────────────────────────────────────────────────
:TRAIN_V6
cls
echo [+] Entrenando modelo V6...
call .venv\Scripts\activate
python match\training\train_q3_q4_models_v6.py
if errorlevel 1 (
    echo [ERROR] Entrenamiento V6 fallo.
) else (
    echo [OK] V6 entrenado correctamente.
)
pause
goto MENU

:: ─────────────────────────────────────────────────
:TRAIN_ALL
cls
echo [+] Entrenando V2...
call .venv\Scripts\activate
python match\training\train_q3_q4_models_v2.py
if errorlevel 1 (
    echo [ERROR] V2 fallo. Abortando.
    pause
    goto MENU
)
echo [OK] V2 completado.
echo.
echo [+] Entrenando V6...
python match\training\train_q3_q4_models_v6.py
if errorlevel 1 (
    echo [ERROR] V6 fallo.
) else (
    echo [OK] V6 completado.
)
pause
goto MENU

:: ─────────────────────────────────────────────────
:TRAIN_REPORT
cls
:TRAIN_REPORT_ONLY
cls
echo [+] V6.2: Solo entrenar...
call train_and_report.bat train
pause
goto MENU

:: ─────────────────────────────────────────────────
:REPORT_ONLY
cls
echo [+] V6.2: Solo generar reporte Q4 ROI...
call train_and_report.bat report
pause
goto MENU

:: ─────────────────────────────────────────────────
:TRAIN_AND_REPORT
cls
echo [+] V6.2: Entrenar + reporte...
call train_and_report.bat both
pause
goto MENU

:: ─────────────────────────────────────────────────
:INSTALAR
cls
call instalar.bat
goto MENU

:: ─────────────────────────────────────────────────
:FIN
endlocal
exit /b 0
