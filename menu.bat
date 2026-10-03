@echo off
setlocal enabledelayedexpansion
title SISTEMA PULPA - Centro de Control
cd /d "%~dp0"

:: ── Verificar .venv ───────────────────────────────
if not exist ".venv\Scripts\activate.bat" (
    echo [AVISO] Entorno virtual .venv no encontrado.
    echo         Puedes crearlo seleccionando la opcion 24 de instalacion.
    echo.
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
echo ==================================================================
echo                  CENTRO DE CONTROL PRINCIPAL
echo ==================================================================
echo.
echo  [1] OPERACION EN VIVO Y SERVICIOS
echo    1) Iniciar Monitoreo V2 (Modular interactivo)
echo    2) Iniciar Monitoreo V2 (Modo CDP directo)
echo    3) Iniciar Bot de Telegram
echo    4) Iniciar API Backend (FastAPI)
echo    5) Iniciar Dashboard Web (API + Frontend Vite)
echo    6) Iniciar Todo (All-in-One: Bot + API + Dashboard)
echo.
echo  [2] ANALISIS, ESTADISTICAS Y CONSENSO
echo    7) Estadisticas de Modelos / Fusion Consensus / Excel (CLI)
echo    8) M27_V3: Reporte ROI y Yield (Modelo Campeon con H2H)
echo.
echo  [3] INGESTA, SCRAPING Y BACKFILL DE DATOS
echo    9) Traer fecha nueva / descargar dias faltantes
echo   10) Backfill historico general (matches.db)
echo   11) Backfill masivo H2H (SofaScore - priorizado por ligas)
echo   12) Comparar scraper tradicional vs obscura
echo.
echo  [4] MODELOS MACHINE LEARNING (ENTRENAMIENTO Y REPORTES)
echo   13) M27_V1: Entrenar modelo
echo   14) M27_V1: Solo reporte ROI
echo   15) M27_V1: Solo probe
echo   16) V6.2: Entrenar modelo
echo   17) V6.2: Generar reporte Q4 ROI
echo   18) V6.2: Entrenar + Reporte completo
echo   19) V6.3: Menu de reportes (interactivo, m27, m30, probe)
echo   20) Entrenar V2 (clasificador base)
echo   21) Entrenar V6 (clasificador base)
echo   22) Entrenar V2 + V6 (en orden)
echo.
echo  [5] MANTENIMIENTO, OBSCURA Y SISTEMA
echo   23) Menu Obscura (Iniciar / Apagar / Instalar / Estado)
echo   24) Instalar / Reparar dependencias (.venv, pip, playwright, npm)
echo.
echo    0) Salir
echo ==================================================================
set /p OPT="  Selecciona una opcion: "

:: Salida
if "%OPT%"=="0" goto FIN

:: [1] Servicios en Vivo
if "%OPT%"=="1" goto RUN_MONITOR_V2
if "%OPT%"=="2" goto RUN_MONITOR_CDP
if "%OPT%"=="3" goto BOT
if "%OPT%"=="4" goto API
if "%OPT%"=="30" goto API
if "%OPT%"=="5" goto DASHBOARD
if "%OPT%"=="32" goto DASHBOARD
if "%OPT%"=="6" goto TODO

:: [2] Analisis y Consenso
if "%OPT%"=="7" goto VIEW_STATS_CLI
if "%OPT%"=="8" goto REPORT_M27_V3_ONLY
if "%OPT%"=="31" goto REPORT_M27_V3_ONLY

:: [3] Ingesta y Backfill
if "%OPT%"=="9" goto FETCH_DATE
if "%OPT%"=="10" goto BACKFILL
if "%OPT%"=="11" goto BACKFILL_H2H_MASIVO
if "%OPT%"=="33" goto BACKFILL_H2H_MASIVO
if "%OPT%"=="12" goto COMPARE_SCRAPER
if "%OPT%"=="25" goto COMPARE_SCRAPER

:: [4] Machine Learning
if "%OPT%"=="13" goto TRAIN_M27_V1
if "%OPT%"=="14" goto REPORT_M27_V1_ONLY
if "%OPT%"=="15" goto REPORT_M27_V1_PROBE
if "%OPT%"=="16" goto TRAIN_V62_ONLY
if "%OPT%"=="17" goto REPORT_V62_ONLY
if "%OPT%"=="18" goto TRAIN_AND_REPORT_V62
if "%OPT%"=="19" goto MENU_V63
if "%OPT%"=="20" goto TRAIN_V2
if "%OPT%"=="28" goto TRAIN_V2
if "%OPT%"=="21" goto TRAIN_V6
if "%OPT%"=="22" goto TRAIN_ALL

:: [5] Mantenimiento y Sistema
if "%OPT%"=="23" goto MENU_OBSCURA
if "%OPT%"=="26" goto MENU_OBSCURA
if "%OPT%"=="27" goto START_OBSCURA_DIRECT
if "%OPT%"=="29" goto STOP_OBSCURA_DIRECT
if "%OPT%"=="24" goto INSTALAR
if "%OPT%"=="99" goto INSTALAR

echo [ERROR] Opcion invalida.
timeout /t 2 /nobreak >nul
goto MENU

:: ─────────────────────────────────────────────────
:: [1] OPERACION EN VIVO Y SERVICIOS
:: ─────────────────────────────────────────────────

:RUN_MONITOR_V2
cls
echo.
echo ===============================================
echo  Iniciando Monitoreo V2 (Interactivo)
echo ===============================================
echo.
echo [1/3] Matando procesos residuales de Chrome...
taskkill /IM chrome.exe /F >nul 2>&1
timeout /t 2 /nobreak >nul

echo [2/3] Iniciando Monitoreo V2...
echo.
echo  NOTA: Cada scrape lanza Chrome temporal (se cierra al terminar).
echo  Si ves demora, es normal: Playwright gestiona los procesos.
echo.
start "Pulpa - Monitoreo V2" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && python bet_monitor_v2\main.py"
goto MENU

:RUN_MONITOR_CDP
cls
echo.
echo ===============================================
echo  Iniciando Monitoreo V2 (Modo CDP Forzado)
echo ===============================================
echo.
set "SSL_CERT_FILE=%~dp0.venv\Lib\site-packages\certifi\cacert.pem"
set "OBSCURA_CDP_URL=http://127.0.0.1:9222"
set "SOFASCORE_SCRAPER_BACKEND=obscura"
set "SOFASCORE_SCRAPER_BACKEND_PROBE=obscura"
set "SOFASCORE_SCRAPER_BACKEND_LIVE=obscura"
set "SOFASCORE_SCRAPER_BACKEND_FT=obscura"
start "Pulpa - Monitoreo V2 (CDP)" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && python bet_monitor_v2\main.py"
goto MENU

:BOT
cls
echo [+] Iniciando Telegram Bot...
start "Pulpa - Telegram Bot" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && python match\telegram_bot.py"
goto MENU

:API
cls
echo [+] Iniciando API Backend (FastAPI)...
start "Pulpa - API Backend" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && python api.py"
goto MENU

:DASHBOARD
cls
echo [+] Iniciando API Backend...
start "Pulpa - API Backend" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && python api.py"
timeout /t 3 /nobreak >nul
echo [+] Iniciando Dashboard Web (Vite/React)...
start "Pulpa - Dashboard" cmd /k "cd /d %~dp0\dashboard && npm run dev"
goto MENU

:TODO
cls
echo [+] Iniciando Suite Completa (Bot + API + Dashboard)...
start "Pulpa - Telegram Bot"  cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && python match\telegram_bot.py"
start "Pulpa - API Backend"   cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && python api.py"
timeout /t 3 /nobreak >nul
start "Pulpa - Dashboard"     cmd /k "cd /d %~dp0\dashboard && npm run dev"
goto MENU

:: ─────────────────────────────────────────────────
:: [2] ANALISIS, ESTADISTICAS Y CONSENSO
:: ─────────────────────────────────────────────────

:VIEW_STATS_CLI
cls
call .venv\Scripts\activate
python tools\stats_cli.py
pause
goto MENU

:REPORT_M27_V3_ONLY
cls
echo [+] M27_V3: Generando reporte de ROI y Yield (Campeon H2H)...
call .venv\Scripts\activate
python match\training\report_m_v1_roi.py --only-m27-v3
pause
goto MENU

:: ─────────────────────────────────────────────────
:: [3] INGESTA, SCRAPING Y BACKFILL DE DATOS
:: ─────────────────────────────────────────────────

:FETCH_DATE
cls
echo [+] Traer fecha nueva (selector directo de fechas faltantes)...
echo.
set "FORCE_FLAG="
set /p REDOWNLOAD="  Forzar redescarga de partidos ya completos? [s/N]: "
if /I "%REDOWNLOAD%"=="s" set "FORCE_FLAG= --force-redownload"
if /I "%REDOWNLOAD%"=="si" set "FORCE_FLAG= --force-redownload"
if /I "%REDOWNLOAD%"=="y" set "FORCE_FLAG= --force-redownload"
call .venv\Scripts\activate && python match\cli.py fetch-date-menu!FORCE_FLAG!
pause
goto MENU

:BACKFILL
cls
echo [+] Backfill historico de matches (matches.db)...
call .venv\Scripts\activate
python match\scripts\backfill.py match\matches.db --all --backend chrome --session-rotate 20
pause
goto MENU

:BACKFILL_H2H_MASIVO
cls
echo.
echo ===============================================
echo  Backfill Masivo H2H (SofaScore)
echo ===============================================
echo.
echo  Descarga H2H para los ~23k partidos restantes.
echo  Prioriza ligas con mas datos, excluye mujeres.
echo  Espera: 40s + jitter entre partidos.
echo.
echo  [!] Esto tomara varias horas/dias.
echo  [!] Si ves errores 403 consecutivos, el script
echo      se pausara para que reinicies internet.
echo.
set /p CONFIRM="  Continuar? [s/N]: "
if /I "%CONFIRM%"=="n" goto MENU
if /I "%CONFIRM%"=="no" goto MENU
if not defined CONFIRM goto MENU
if /I not "%CONFIRM%"=="s" if /I not "%CONFIRM%"=="si" if /I not "%CONFIRM%"=="y" if /I not "%CONFIRM%"=="yes" goto MENU

call .venv\Scripts\activate
python temp_scripts\backfill_h2h_masivo.py
pause
goto MENU

:COMPARE_SCRAPER
cls
echo [+] Comparando scraper tradicional vs obscura...
call .venv\Scripts\activate
python match\scripts\compare_scrapers.py --all --limit 5
pause
goto MENU

:: ─────────────────────────────────────────────────
:: [4] MODELOS MACHINE LEARNING (ENTRENAMIENTO Y REPORTES)
:: ─────────────────────────────────────────────────

:TRAIN_M27_V1
cls
echo [+] M27_V1: Entrenando modelo...
call .venv\Scripts\activate
python match\training\train_q4_m27_v1.py
pause
goto MENU

:REPORT_M27_V1_ONLY
cls
echo [+] M27_V1: Generando reporte ROI...
call .venv\Scripts\activate
python match\training\report_v63_q4_roi.py --only-m27-v1
pause
goto MENU

:REPORT_M27_V1_PROBE
cls
echo [+] M27_V1: Modo probe (sin filtros live)...
call .venv\Scripts\activate
python match\training\report_v63_q4_roi.py --no-v62 --no-raw --no-m27 --no-m30 --no-m27-probe --no-m30-probe --no-m27-v1 --with-m27-v1-probe
pause
goto MENU

:TRAIN_V62_ONLY
cls
echo [+] V6.2: Entrenando modelo con poda de ligas...
call .venv\Scripts\activate
python match\training\train_q3_q4_models_v6_2.py
pause
goto MENU

:REPORT_V62_ONLY
cls
echo [+] V6.2: Generando reporte Q4 ROI...
call .venv\Scripts\activate
python match\training\report_v62_q4_roi.py
pause
goto MENU

:TRAIN_AND_REPORT_V62
cls
echo [+] V6.2: Entrenando y generando reporte completo...
call .venv\Scripts\activate
python match\training\train_q3_q4_models_v6_2.py
if errorlevel 1 (
    echo [ERROR] Entrenamiento V6.2 fallo.
    pause
    goto MENU
)
python match\training\report_v62_q4_roi.py
pause
goto MENU

:MENU_V63
cls
echo.
echo ==================================================
echo           V6.3 - SUITE DE REPORTES ROI
echo ==================================================
echo.
echo   1) Reporte interactivo (elige bloques)
echo   2) Solo reporte m27 (rapido)
echo   3) Solo reporte raw monitor
echo   4) Solo reporte m30
echo   5) Reporte m27 + m30 (sin raw)
echo   6) Solo m27 (rebuild caches)
echo   7) m27 probe (sin post-filtros live)
echo   8) m30 probe (sin post-filtros live)
echo   0) Volver al menu principal
echo.
set /p VOPT="  Selecciona: "

if "%VOPT%"=="1" (
    call .venv\Scripts\activate
    python match\training\report_v63_q4_roi.py
    pause
    goto MENU_V63
)
if "%VOPT%"=="2" (
    call .venv\Scripts\activate
    python match\training\report_v63_q4_roi.py --only-m27
    pause
    goto MENU_V63
)
if "%VOPT%"=="3" (
    call .venv\Scripts\activate
    python match\training\report_v63_q4_roi.py --no-v62 --with-raw --no-m27 --no-m30
    pause
    goto MENU_V63
)
if "%VOPT%"=="4" (
    call .venv\Scripts\activate
    python match\training\report_v63_q4_roi.py --no-v62 --no-raw --no-m27 --with-m30
    pause
    goto MENU_V63
)
if "%VOPT%"=="5" (
    call .venv\Scripts\activate
    python match\training\report_v63_q4_roi.py --no-v62 --no-raw --with-m27 --with-m30
    pause
    goto MENU_V63
)
if "%VOPT%"=="6" (
    call .venv\Scripts\activate
    python match\training\report_v63_q4_roi.py --only-m27 --rebuild-pred-cache --rebuild-splits-cache
    pause
    goto MENU_V63
)
if "%VOPT%"=="7" (
    call .venv\Scripts\activate
    python match\training\report_v63_q4_roi.py --no-v62 --no-raw --no-m27 --no-m30 --with-m27-probe
    pause
    goto MENU_V63
)
if "%VOPT%"=="8" (
    call .venv\Scripts\activate
    python match\training\report_v63_q4_roi.py --no-v62 --no-raw --no-m27 --no-m30 --with-m30-probe
    pause
    goto MENU_V63
)
if "%VOPT%"=="0" goto MENU

echo [ERROR] Opcion invalida.
timeout /t 2 /nobreak >nul
goto MENU_V63

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
:: [5] MANTENIMIENTO, OBSCURA Y SISTEMA
:: ─────────────────────────────────────────────────

:MENU_OBSCURA
call menu_obscura.bat
goto MENU

:START_OBSCURA_DIRECT
cls
call menu_obscura.bat start
pause
goto MENU

:STOP_OBSCURA_DIRECT
cls
call menu_obscura.bat stop
pause
goto MENU

:INSTALAR
cls
echo.
echo ==================================================
echo        PULPA - INSTALACION DE DEPENDENCIAS
echo ==================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado. Instalalo desde https://python.org
    pause
    goto MENU
)

if not exist ".venv\Scripts\activate.bat" (
    echo [+] Creando entorno virtual .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual.
        pause
        goto MENU
    )
    echo [OK] Entorno virtual creado.
) else (
    echo [OK] Entorno virtual ya existe.
)

call .venv\Scripts\activate.bat

echo.
echo [+] Actualizando pip...
python -m pip install --upgrade pip --quiet

echo.
echo [+] Instalando dependencias Python (match/requirements.txt)...
pip install -r match\requirements.txt
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de dependencias Python.
    pause
    goto MENU
)
echo [OK] Dependencias Python instaladas.

echo.
echo [+] Instalando navegadores Playwright...
python -m playwright install chromium
if errorlevel 1 (
    echo [AVISO] Playwright install fallo. Puede continuar si ya estaban instalados.
)
echo [OK] Playwright listo.

echo.
echo [+] Instalando dependencias Node.js (dashboard)...
where npm >nul 2>&1
if errorlevel 1 (
    echo [AVISO] npm no encontrado. Omitiendo instalacion del dashboard.
) else (
    cd dashboard
    call npm install
    if errorlevel 1 (
        echo [AVISO] npm install fallo en dashboard.
    ) else (
        echo [OK] Dependencias Node instaladas.
    )
    cd ..
)

echo.
echo ==================================================
echo   Instalacion completada con exito.
echo ==================================================
pause
goto MENU

:: ─────────────────────────────────────────────────
:FIN
endlocal
exit /b 0
