@echo off
setlocal
title PULPA - Instalacion de dependencias
cd /d "%~dp0"

echo ==================================================
echo        PULPA - INSTALAR DEPENDENCIAS
echo ==================================================
echo.

:: ── Verificar Python ──────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado. Instalalo desde https://python.org
    pause
    exit /b 1
)

:: ── Crear .venv si no existe ──────────────────────
if not exist ".venv\Scripts\activate.bat" (
    echo [+] Creando entorno virtual .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
    echo [OK] Entorno virtual creado.
) else (
    echo [OK] Entorno virtual ya existe.
)

:: ── Activar .venv ─────────────────────────────────
call .venv\Scripts\activate.bat

:: ── Actualizar pip ────────────────────────────────
echo.
echo [+] Actualizando pip...
python -m pip install --upgrade pip --quiet

:: ── Instalar dependencias Python ──────────────────
echo.
echo [+] Instalando dependencias Python (match/requirements.txt)...
pip install -r match\requirements.txt
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de dependencias Python.
    pause
    exit /b 1
)
echo [OK] Dependencias Python instaladas.

:: ── Instalar Playwright browsers ──────────────────
echo.
echo [+] Instalando navegadores Playwright...
python -m playwright install chromium
if errorlevel 1 (
    echo [AVISO] Playwright install fallo. Puede continuar si ya estaban instalados.
)
echo [OK] Playwright listo.

:: ── Instalar dependencias Node (dashboard) ────────
echo.
echo [+] Instalando dependencias Node.js (v13_dashboard)...
where npm >nul 2>&1
if errorlevel 1 (
    echo [AVISO] npm no encontrado. Omitiendo dashboard.
) else (
    cd v13_dashboard
    npm install
    if errorlevel 1 (
        echo [AVISO] npm install fallo en v13_dashboard.
    ) else (
        echo [OK] Dependencias Node instaladas.
    )
    cd ..
)

echo.
echo ==================================================
echo   Instalacion completada. Ejecuta menu.bat
echo ==================================================
pause
endlocal
