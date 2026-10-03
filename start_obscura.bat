@echo off
setlocal
cd /d "%~dp0"

set "OBSCURA_DIR=%~dp0tools\obscura\v0.1.5"
set "OBSCURA_EXE=%OBSCURA_DIR%\obscura.exe"
set "SSL_CERT_FILE=%~dp0.venv\Lib\site-packages\certifi\cacert.pem"

if not exist "%OBSCURA_EXE%" (
    echo [ERROR] Obscura no esta instalado.
    echo         Ejecuta instalar_obscura.bat primero.
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $c = New-Object Net.Sockets.TcpClient('127.0.0.1', 9222); $c.Close(); exit 0 } catch { exit 1 }"
if not errorlevel 1 (
    echo [OK] Obscura ya estaba corriendo en 127.0.0.1:9222
    endlocal
    exit /b 0
)

powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command "$env:SSL_CERT_FILE='%SSL_CERT_FILE%'; Start-Process -WindowStyle Hidden -FilePath '%OBSCURA_EXE%' -ArgumentList @('serve','--port','9222','--stealth') -WorkingDirectory '%OBSCURA_DIR%'"
echo [OK] Obscura arrancado ahora con stealth y SSL_CERT_FILE habilitado en 127.0.0.1:9222
endlocal
