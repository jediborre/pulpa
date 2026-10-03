@echo off
setlocal enabledelayedexpansion
title PULPA - Menu Obscura
cd /d "%~dp0"

set "OBSCURA_DIR=%~dp0tools\obscura\v0.1.5"
set "OBSCURA_EXE=%OBSCURA_DIR%\obscura.exe"
set "SSL_CERT_FILE=%~dp0.venv\Lib\site-packages\certifi\cacert.pem"
set "VERSION=v0.1.5"
set "ZIP_FILE=%OBSCURA_DIR%\obscura-x86_64-windows.zip"
set "URL=https://github.com/h4ckf0r0day/obscura/releases/download/%VERSION%/obscura-x86_64-windows.zip"

:: ── Verificación de parámetros directos (start / stop / install / status) ──
if /i "%~1"=="start" goto DO_START
if /i "%~1"=="iniciar" goto DO_START
if /i "%~1"=="stop" goto DO_STOP
if /i "%~1"=="detener" goto DO_STOP
if /i "%~1"=="apagar" goto DO_STOP
if /i "%~1"=="install" goto DO_INSTALL
if /i "%~1"=="instalar" goto DO_INSTALL
if /i "%~1"=="status" goto DO_STATUS
if /i "%~1"=="estado" goto DO_STATUS

:: ──────────────────────────────────────────────────────────
:MENU
cls
echo.
echo  ==================================================
echo               PULPA - MENU OBSCURA
echo  ==================================================
echo.
echo   1) Iniciar Obscura (CDP en 127.0.0.1:9222)
echo   2) Detener Obscura (Cerrar procesos activos)
echo   3) Instalar / Reinstalar Obscura (%VERSION%)
echo   4) Verificar Estado del Puerto (127.0.0.1:9222)
echo   0) Volver / Salir
echo.
set /p OPT="  Selecciona una opcion: "

if "%OPT%"=="1" (
    call :DO_START
    pause
    goto MENU
)
if "%OPT%"=="2" (
    call :DO_STOP
    pause
    goto MENU
)
if "%OPT%"=="3" (
    call :DO_INSTALL
    pause
    goto MENU
)
if "%OPT%"=="4" (
    call :DO_STATUS
    pause
    goto MENU
)
if "%OPT%"=="0" goto FIN

echo [ERROR] Opcion invalida.
timeout /t 2 /nobreak >nul
goto MENU

:: ──────────────────────────────────────────────────────────
:DO_START
echo.
echo [+] Verificando instalacion de Obscura...
if not exist "%OBSCURA_EXE%" (
    echo [ERROR] Obscura no esta instalado en: %OBSCURA_EXE%
    echo         Selecciona la opcion 3 o ejecuta: menu_obscura.bat install
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

:: ──────────────────────────────────────────────────────────
:DO_STOP
echo.
echo [+] Deteniendo proceso obscura.exe...
taskkill /IM obscura.exe /F >nul 2>&1
if errorlevel 1 (
    echo [OK] No habia ningun proceso de Obscura activo.
) else (
    echo [OK] Proceso obscura.exe detenido exitosamente.
)
exit /b 0

:: ──────────────────────────────────────────────────────────
:DO_STATUS
echo.
echo [+] Comprobando conexion a 127.0.0.1:9222...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $c = New-Object Net.Sockets.TcpClient('127.0.0.1', 9222); $c.Close(); Write-Host '[ACTIVO] Obscura esta respondiendo en 127.0.0.1:9222' -ForegroundColor Green; exit 0 } catch { Write-Host '[INACTIVO] No hay servicio respondiendo en el puerto 9222' -ForegroundColor Yellow; exit 1 }"
exit /b 0

:: ──────────────────────────────────────────────────────────
:DO_INSTALL
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

echo [+] Descargando Obscura %VERSION% desde GitHub...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%URL%' -OutFile '%ZIP_FILE%'"
if errorlevel 1 (
    echo [ERROR] Fallo la descarga de Obscura. Revisa tu conexion a Internet.
    exit /b 1
)

echo [+] Descomprimiendo binarios...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath '%ZIP_FILE%' -DestinationPath '%OBSCURA_DIR%' -Force"
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

:: ──────────────────────────────────────────────────────────
:FIN
endlocal
exit /b 0
