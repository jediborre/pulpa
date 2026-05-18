@echo off
setlocal
cd /d "%~dp0"

taskkill /IM obscura.exe /F >nul 2>&1
if errorlevel 1 (
    echo [OK] No habia proceso obscura corriendo.
) else (
    echo [OK] Obscura detenido.
)

endlocal
