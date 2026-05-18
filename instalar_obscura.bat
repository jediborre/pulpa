@echo off
setlocal
title PULPA - Instalar Obscura
cd /d "%~dp0"

set "VERSION=v0.1.5"
set "BASE_DIR=tools\obscura\%VERSION%"
set "ZIP_FILE=%BASE_DIR%\obscura-x86_64-windows.zip"
set "URL=https://github.com/h4ckf0r0day/obscura/releases/download/%VERSION%/obscura-x86_64-windows.zip"

echo ==================================================
echo        PULPA - INSTALAR OBSCURA
echo ==================================================
echo.

where powershell >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PowerShell no encontrado.
    pause
    exit /b 1
)

if not exist "tools\obscura" mkdir "tools\obscura"
if not exist "%BASE_DIR%" mkdir "%BASE_DIR%"

echo [+] Descargando Obscura %VERSION%...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%URL%' -OutFile '%ZIP_FILE%'"
if errorlevel 1 (
    echo [ERROR] No se pudo descargar Obscura.
    pause
    exit /b 1
)

echo [+] Descomprimiendo...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath '%ZIP_FILE%' -DestinationPath '%BASE_DIR%' -Force"
if errorlevel 1 (
    echo [ERROR] No se pudo descomprimir Obscura.
    pause
    exit /b 1
)

if not exist "%BASE_DIR%\obscura.exe" (
    echo [ERROR] No se encontro obscura.exe tras la extraccion.
    pause
    exit /b 1
)

echo.
echo [OK] Obscura listo en %BASE_DIR%
echo [OK] Ejecutable: %BASE_DIR%\obscura.exe
echo [OK] Servidor CDP: "%BASE_DIR%\obscura.exe serve --port 9222"
echo.
choice /m "Quieres arrancar Obscura ahora"
if errorlevel 2 goto END
call start_obscura.bat
pause
:END
endlocal
