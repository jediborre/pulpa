@echo off
setlocal enabledelayedexpansion
title SISTEMA PULPA - Centro de Control
cd /d "%~dp0"

:: ── Variables Globales y Obscura ───────────────────
set "OBSCURA_DIR=%~dp0tools\obscura\v0.1.5"
set "OBSCURA_EXE=%OBSCURA_DIR%\obscura.exe"
set "SSL_CERT_FILE=%~dp0.venv\Lib\site-packages\certifi\cacert.pem"
set "OBSCURA_VERSION=v0.1.5"
set "OBSCURA_ZIP=%OBSCURA_DIR%\obscura-x86_64-windows.zip"
set "OBSCURA_URL=https://github.com/h4ckf0r0day/obscura/releases/download/%OBSCURA_VERSION%/obscura-x86_64-windows.zip"

:: ── Parametros directos por linea de comandos (para scripts y automatizacion) ──
if /i "%~1"=="help" goto CLI_HELP
if /i "%~1"=="/?" goto CLI_HELP
if /i "%~1"=="-h" goto CLI_HELP
if /i "%~1"=="--help" goto CLI_HELP
if /i "%~1"=="v1" goto RUN_MONITOR_V1_CLI
if /i "%~1"=="monitor_v1" goto RUN_MONITOR_V1_CLI
if /i "%~1"=="v2" goto RUN_MONITOR_V2_CLI
if /i "%~1"=="monitor_v2" goto RUN_MONITOR_V2_CLI
if /i "%~1"=="v2_cdp" goto RUN_MONITOR_CDP_CLI
if /i "%~1"=="monitor_v2_cdp" goto RUN_MONITOR_CDP_CLI
if /i "%~1"=="todo_v1" goto DO_TODO_V1_CLI
if /i "%~1"=="todo_v2" goto DO_TODO_V2_CLI
if /i "%~1"=="obscura" goto MENU_OBSCURA
if /i "%~1"=="obscura_start" goto DO_START_OBSCURA_CLI
if /i "%~1"=="obscura_stop" goto DO_STOP_OBSCURA_CLI
if /i "%~1"=="obscura_install" goto DO_INSTALL_OBSCURA_CLI
if /i "%~1"=="obscura_status" goto DO_STATUS_OBSCURA_CLI
if /i "%~1"=="start" goto DO_START_OBSCURA_CLI
if /i "%~1"=="stop" goto DO_STOP_OBSCURA_CLI
if /i "%~1"=="install" goto DO_INSTALL_OBSCURA_CLI
if /i "%~1"=="status" goto DO_STATUS_OBSCURA_CLI

:: ── Verificar .venv ───────────────────────────────
if not exist ".venv\Scripts\activate.bat" (
    echo [AVISO] Entorno virtual .venv no encontrado.
    echo         Puedes crearlo seleccionando la opcion 25 de instalacion.
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
echo    1) Iniciar Monitor V1 (Telegram Bot + Bet Monitor)
echo    2) Iniciar Monitor V2 (Daemon Asincrono - Interactivo)
echo    3) Iniciar Monitor V2 (Modo CDP directo)
echo    4) Iniciar API Backend (FastAPI)
echo    5) Iniciar Dashboard Web (API + Frontend Vite)
echo    6) Iniciar Todo con V1 (All-in-One: Monitor V1 + API + Dashboard)
echo    7) Iniciar Todo con V2 (All-in-One: Monitor V2 + API + Dashboard)
echo.
echo  [2] ANALISIS, ESTADISTICAS Y CONSENSO
echo    8) Estadisticas de Modelos / Fusion Consensus / Excel (CLI)
echo    9) M27_V3: Reporte ROI y Yield (Modelo Campeon con H2H)
echo.
echo  [3] INGESTA, SCRAPING Y BACKFILL DE DATOS
echo   10) Traer fecha nueva / descargar dias faltantes
echo   11) Backfill historico general (matches.db)
echo   12) Backfill masivo H2H (SofaScore - priorizado por ligas)
echo   13) Comparar scraper tradicional vs obscura
echo.
echo  [4] MODELOS MACHINE LEARNING (ENTRENAMIENTO Y REPORTES)
echo   14) M27_V1: Entrenar modelo
echo   15) M27_V1: Solo reporte ROI
echo   16) M27_V1: Solo probe
echo   17) V6.2: Entrenar modelo
echo   18) V6.2: Generar reporte Q4 ROI
echo   19) V6.2: Entrenar + Reporte completo
echo   20) V6.3: Menu de reportes (interactivo, m27, m30, probe)
echo   21) Entrenar V2 (clasificador base)
echo   22) Entrenar V6 (clasificador base)
echo   23) Entrenar V2 + V6 (en orden)
echo.
echo  [5] MANTENIMIENTO, OBSCURA Y SISTEMA
echo   24) Menu Obscura (Iniciar / Apagar / Instalar / Estado)
echo   25) Instalar / Reparar dependencias (.venv, pip, playwright, npm)
echo.
echo    0) Salir
echo ==================================================================
set /p OPT="  Selecciona una opcion: "

:: Salida
if "%OPT%"=="0" goto FIN

:: [1] Servicios en Vivo
if "%OPT%"=="1" goto RUN_MONITOR_V1
if /i "%OPT%"=="v1" goto RUN_MONITOR_V1
if "%OPT%"=="2" goto RUN_MONITOR_V2
if /i "%OPT%"=="v2" goto RUN_MONITOR_V2
if "%OPT%"=="3" goto RUN_MONITOR_CDP
if "%OPT%"=="4" goto API
if "%OPT%"=="30" goto API
if "%OPT%"=="5" goto DASHBOARD
if "%OPT%"=="32" goto DASHBOARD
if "%OPT%"=="6" goto TODO_V1
if "%OPT%"=="7" goto TODO_V2

:: [2] Analisis y Consenso
if "%OPT%"=="8" goto VIEW_STATS_CLI
if "%OPT%"=="9" goto REPORT_M27_V3_ONLY
if "%OPT%"=="31" goto REPORT_M27_V3_ONLY

:: [3] Ingesta y Backfill
if "%OPT%"=="10" goto FETCH_DATE
if "%OPT%"=="11" goto BACKFILL
if "%OPT%"=="12" goto BACKFILL_H2H_MASIVO
if "%OPT%"=="33" goto BACKFILL_H2H_MASIVO
if "%OPT%"=="13" goto COMPARE_SCRAPER

:: [4] Machine Learning
if "%OPT%"=="14" goto TRAIN_M27_V1
if "%OPT%"=="15" goto REPORT_M27_V1_ONLY
if "%OPT%"=="16" goto REPORT_M27_V1_PROBE
if "%OPT%"=="17" goto TRAIN_V62_ONLY
if "%OPT%"=="18" goto REPORT_V62_ONLY
if "%OPT%"=="19" goto TRAIN_AND_REPORT_V62
if "%OPT%"=="20" goto MENU_V63
if "%OPT%"=="21" goto TRAIN_V2
if "%OPT%"=="28" goto TRAIN_V2
if "%OPT%"=="22" goto TRAIN_V6
if "%OPT%"=="23" goto TRAIN_ALL

:: [5] Mantenimiento y Sistema
if "%OPT%"=="24" goto MENU_OBSCURA
if "%OPT%"=="26" goto MENU_OBSCURA
if "%OPT%"=="27" goto START_OBSCURA_DIRECT
if "%OPT%"=="29" goto STOP_OBSCURA_DIRECT
if "%OPT%"=="25" goto INSTALAR
if "%OPT%"=="99" goto INSTALAR

echo [ERROR] Opcion invalida.
timeout /t 2 /nobreak >nul
goto MENU

:: ─────────────────────────────────────────────────
:: [1] OPERACION EN VIVO Y SERVICIOS
:: ─────────────────────────────────────────────────

:RUN_MONITOR_V1
cls
echo.
echo ===============================================
echo  Iniciando Monitor V1 (Telegram Bot + Bet Monitor)
echo ===============================================
echo.
echo [+] Levantando Monitor V1 en una nueva ventana...
start "Pulpa - Monitor V1" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && python monitor_v1\main.py"
goto MENU

:BOT
goto RUN_MONITOR_V1

:RUN_MONITOR_V2
cls
echo.
echo ===============================================
echo  Iniciando Monitor V2 (Interactivo)
echo ===============================================
echo.
echo [1/3] Matando procesos residuales de Chrome...
taskkill /IM chrome.exe /F >nul 2>&1
timeout /t 2 /nobreak >nul

echo [2/3] Iniciando Monitor V2 en una nueva ventana...
echo.
echo  NOTA: Cada scrape lanza Chrome temporal (se cierra al terminar).
echo  Si ves demora, es normal: Playwright gestiona los procesos.
echo.
start "Pulpa - Monitoreo V2" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && python monitor_v2\main.py"
goto MENU

:RUN_MONITOR_CDP
cls
echo ===============================================
echo  Iniciando Monitor V2 (Modo CDP Forzado)
echo ===============================================
echo.
set "SSL_CERT_FILE=%~dp0.venv\Lib\site-packages\certifi\cacert.pem"
set "OBSCURA_CDP_URL=http://127.0.0.1:9222"
set "SOFASCORE_SCRAPER_BACKEND=obscura"
set "SOFASCORE_SCRAPER_BACKEND_PROBE=obscura"
set "SOFASCORE_SCRAPER_BACKEND_LIVE=obscura"
set "SOFASCORE_SCRAPER_BACKEND_FT=obscura"
start "Pulpa - Monitoreo V2 (CDP)" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && python monitor_v2\main.py"
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

:TODO_V1
cls
echo.
echo =========================================================
echo  Iniciando Suite Completa con Monitor V1 (All-in-One)
echo  (Monitor V1 + API Backend + Dashboard Web)
echo =========================================================
echo.
echo [1/3] Iniciando Monitor V1 (Telegram Bot)...
start "Pulpa - Monitor V1" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && python monitor_v1\main.py"
echo [2/3] Iniciando API Backend (FastAPI)...
start "Pulpa - API Backend" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && python api.py"
timeout /t 3 /nobreak >nul
echo [3/3] Iniciando Dashboard Web (Vite/React)...
start "Pulpa - Dashboard" cmd /k "cd /d %~dp0\dashboard && npm run dev"
goto MENU

:TODO_V2
cls
echo.
echo =========================================================
echo  Iniciando Suite Completa con Monitor V2 (All-in-One)
echo  (Monitor V2 + API Backend + Dashboard Web)
echo =========================================================
echo.
echo [1/4] Matando procesos residuales de Chrome...
taskkill /IM chrome.exe /F >nul 2>&1
timeout /t 2 /nobreak >nul
echo [2/4] Iniciando Monitor V2 (Daemon Asincrono)...
start "Pulpa - Monitoreo V2" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && python monitor_v2\main.py"
echo [3/4] Iniciando API Backend (FastAPI)...
start "Pulpa - API Backend" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate && python api.py"
timeout /t 3 /nobreak >nul
echo [4/4] Iniciando Dashboard Web (Vite/React)...
start "Pulpa - Dashboard" cmd /k "cd /d %~dp0\dashboard && npm run dev"
goto MENU

:TODO
goto TODO_V1

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
python match\scripts\backfill.py matches.db --all --backend chrome --session-rotate 20
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
python tmp\backfill_h2h_masivo.py
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
cls
echo.
echo ==================================================================
echo                  CONTROL DE OBSCURA (%OBSCURA_VERSION%)
echo ==================================================================
echo.
echo   1) Iniciar Obscura (CDP en 127.0.0.1:9222 con stealth)
echo   2) Detener Obscura (Cerrar procesos activos)
echo   3) Instalar / Reinstalar Obscura (%OBSCURA_VERSION%)
echo   4) Verificar Estado del Puerto (127.0.0.1:9222)
echo   0) Volver al Menu Principal
echo.
echo ==================================================================
set /p OBS_OPT="  Selecciona una opcion: "

if "%OBS_OPT%"=="1" (
    call :DO_START_OBSCURA
    pause
    goto MENU_OBSCURA
)
if "%OBS_OPT%"=="2" (
    call :DO_STOP_OBSCURA
    pause
    goto MENU_OBSCURA
)
if "%OBS_OPT%"=="3" (
    call :DO_INSTALL_OBSCURA
    pause
    goto MENU_OBSCURA
)
if "%OBS_OPT%"=="4" (
    call :DO_STATUS_OBSCURA
    pause
    goto MENU_OBSCURA
)
if "%OBS_OPT%"=="0" goto MENU

echo [ERROR] Opcion invalida.
timeout /t 2 /nobreak >nul
goto MENU_OBSCURA

:START_OBSCURA_DIRECT
call :DO_START_OBSCURA
pause
goto MENU

:STOP_OBSCURA_DIRECT
call :DO_STOP_OBSCURA
pause
goto MENU

:: ── Rutas CLI de Monitores y Servicios (retorno directo) ──
:RUN_MONITOR_V1_CLI
echo [+] Iniciando Monitor V1 (Telegram Bot + Bet Monitor) desde CLI...
call .venv\Scripts\activate
python monitor_v1\main.py
exit /b %errorlevel%

:RUN_MONITOR_V2_CLI
echo [+] Iniciando Monitor V2 desde CLI...
call .venv\Scripts\activate
python monitor_v2\main.py
exit /b %errorlevel%

:RUN_MONITOR_CDP_CLI
echo [+] Iniciando Monitor V2 (Modo CDP Forzado) desde CLI...
set "SSL_CERT_FILE=%~dp0.venv\Lib\site-packages\certifi\cacert.pem"
set "OBSCURA_CDP_URL=http://127.0.0.1:9222"
set "SOFASCORE_SCRAPER_BACKEND=obscura"
set "SOFASCORE_SCRAPER_BACKEND_PROBE=obscura"
set "SOFASCORE_SCRAPER_BACKEND_LIVE=obscura"
set "SOFASCORE_SCRAPER_BACKEND_FT=obscura"
call .venv\Scripts\activate
python monitor_v2\main.py
exit /b %errorlevel%

:DO_TODO_V1_CLI
call :TODO_V1
exit /b 0

:DO_TODO_V2_CLI
call :TODO_V2
exit /b 0

:CLI_HELP
echo ==================================================================
echo                      SISTEMA PULPA - AYUDA CLI
echo ==================================================================
echo Uso: menu.bat [comando]
echo.
echo Comandos disponibles:
echo   v1 / monitor_v1         Inicia Monitor V1 (Telegram Bot + Bet Monitor)
echo   v2 / monitor_v2         Inicia Monitor V2 (Daemon Asincrono interactivo)
echo   v2_cdp / monitor_v2_cdp Inicia Monitor V2 en modo CDP directo
echo   todo_v1                 Inicia Todo con Monitor V1 (Bot + API + Dashboard)
echo   todo_v2                 Inicia Todo con Monitor V2 (Daemon + API + Dashboard)
echo   obscura_start / start   Inicia el servicio Obscura en puerto 9222
echo   obscura_stop / stop     Detiene el servicio Obscura
echo   obscura_status / status Verifica estado de Obscura
echo   obscura_install         Instala o actualiza Obscura v0.1.5
echo.
echo Sin argumentos abre el Centro de Control interactivo.
echo ==================================================================
exit /b 0

:: ── Rutas CLI Obscura (retorno directo sin pause) ───
:DO_START_OBSCURA_CLI
call :DO_START_OBSCURA
exit /b %errorlevel%

:DO_STOP_OBSCURA_CLI
call :DO_STOP_OBSCURA
exit /b %errorlevel%

:DO_INSTALL_OBSCURA_CLI
call :DO_INSTALL_OBSCURA
exit /b %errorlevel%

:DO_STATUS_OBSCURA_CLI
call :DO_STATUS_OBSCURA
exit /b %errorlevel%

:: ── Funciones Core de Obscura ────────────────────────
:DO_START_OBSCURA
echo.
echo [+] Verificando instalacion de Obscura...
if not exist "%OBSCURA_EXE%" (
    echo [ERROR] Obscura no esta instalado en: %OBSCURA_EXE%
    echo         Selecciona la opcion 3 en el menu de Obscura para instalarlo.
    exit /b 1
)

echo [+] Verificando si el puerto 9222 ya esta en uso...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $c = New-Object Net.Sockets.TcpClient('127.0.0.1', 9222); $c.Close(); exit 0 } catch { exit 1 }"
if not errorlevel 1 (
    echo [OK] Obscura ya esta corriendo en 127.0.0.1:9222
    exit /b 0
)

echo [+] Iniciando Obscura CDP en puerto 9222 con stealth y SSL cert...
powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command "$env:SSL_CERT_FILE='%SSL_CERT_FILE%'; Start-Process -WindowStyle Hidden -FilePath '%OBSCURA_EXE%' -ArgumentList @('serve','--port','9222','--stealth') -WorkingDirectory '%OBSCURA_DIR%'"

:: Breve comprobacion
timeout /t 2 /nobreak >nul
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $c = New-Object Net.Sockets.TcpClient('127.0.0.1', 9222); $c.Close(); exit 0 } catch { exit 1 }"
if not errorlevel 1 (
    echo [OK] Obscura iniciado exitosamente en 127.0.0.1:9222
    exit /b 0
) else (
    echo [ADVERTENCIA] El proceso fue lanzado, pero el puerto 9222 aun no responde.
    exit /b 0
)

:DO_STOP_OBSCURA
echo.
echo [+] Deteniendo proceso obscura.exe...
taskkill /IM obscura.exe /F >nul 2>&1
if errorlevel 1 (
    echo [OK] No habia ningun proceso de Obscura activo.
) else (
    echo [OK] Proceso obscura.exe detenido exitosamente.
)
exit /b 0

:DO_STATUS_OBSCURA
echo.
echo [+] Comprobando conexion a 127.0.0.1:9222...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $c = New-Object Net.Sockets.TcpClient('127.0.0.1', 9222); $c.Close(); Write-Host '[ACTIVO] Obscura esta respondiendo en 127.0.0.1:9222' -ForegroundColor Green; exit 0 } catch { Write-Host '[INACTIVO] No hay servicio respondiendo en el puerto 9222' -ForegroundColor Yellow; exit 1 }"
exit /b 0

:DO_INSTALL_OBSCURA
echo.
echo ==================================================
echo         DESCARGA E INSTALACION DE OBSCURA
echo ==================================================
echo.

where powershell >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PowerShell no encontrado en el sistema.
    exit /b 1
)

if not exist "tools\obscura" mkdir "tools\obscura"
if not exist "%OBSCURA_DIR%" mkdir "%OBSCURA_DIR%"

echo [+] Descargando Obscura %OBSCURA_VERSION% desde GitHub...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%OBSCURA_URL%' -OutFile '%OBSCURA_ZIP%'"
if errorlevel 1 (
    echo [ERROR] Fallo la descarga de Obscura. Revisa tu conexion a Internet.
    exit /b 1
)

echo [+] Descomprimiendo binarios...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath '%OBSCURA_ZIP%' -DestinationPath '%OBSCURA_DIR%' -Force"
if errorlevel 1 (
    echo [ERROR] No se pudo descomprimir el archivo ZIP.
    exit /b 1
)

if not exist "%OBSCURA_EXE%" (
    echo [ERROR] No se encontro obscura.exe tras la descompresion.
    exit /b 1
)

echo.
echo [OK] Obscura instalado correctamente en: %OBSCURA_DIR%
echo [OK] Ejecutable listo: %OBSCURA_EXE%
exit /b 0


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
