@echo off
REM Batch script to automate v6/v6.2 training and Q4 ROI report generation
REM Location: c:\Users\borre\OneDrive\OLD\Escritorio\pulpa\train_and_report.bat

setlocal enabledelayedexpansion

set WORKSPACE=C:\Users\App\Desktop\pulpa
set VENV=%WORKSPACE%\.venv\Scripts
set PYTHON=%VENV%\python.exe
set LOG=%WORKSPACE%\train_and_report.log
set MODE=%~1
set RUN_TRAIN=0
set RUN_REPORT=0

cd /d %WORKSPACE% || goto error

if "%MODE%"=="" goto ask_mode
if /I "%MODE%"=="1" goto mode_train
if /I "%MODE%"=="train" goto mode_train
if /I "%MODE%"=="2" goto mode_report
if /I "%MODE%"=="report" goto mode_report
if /I "%MODE%"=="3" goto mode_both
if /I "%MODE%"=="both" goto mode_both

echo [ERROR] Modo invalido: %MODE%
echo Usa: train_and_report.bat [train^|report^|both]
goto error

:ask_mode
echo.
echo ===============================
echo   TRAIN / REPORT PIPELINE
echo ===============================
echo   1^) Entrenar V6.2
echo   2^) Generar reporte Q4 ROI
echo   3^) Entrenar + Reporte
echo.
set /p MODE="Selecciona opcion (1/2/3): "

if "%MODE%"=="1" goto mode_train
if "%MODE%"=="2" goto mode_report
if "%MODE%"=="3" goto mode_both

echo [ERROR] Opcion invalida.
goto end

:mode_train
set RUN_TRAIN=1
goto mode_ready

:mode_report
set RUN_REPORT=1
goto mode_ready

:mode_both
set RUN_TRAIN=1
set RUN_REPORT=1
goto mode_ready

:mode_ready

echo ===== Training ^& Reporting Pipeline ===== >> %LOG%
echo Started: %date% %time% >> %LOG%
echo Mode: train=%RUN_TRAIN% report=%RUN_REPORT% >> %LOG%
echo. >> %LOG%

REM Activate venv (PowerShell approach for compatibility)
echo [INIT] Activating virtual environment...
call %VENV%\activate.bat

if "%RUN_TRAIN%"=="1" (
    REM Train V6.2 with external league exclusions
    echo [TRAIN] Training V6.2 models with external league exclusions...
    echo Training V6.2... >> %LOG%
    %PYTHON% match\training\train_q3_q4_models_v6_2.py >> %LOG% 2>&1
    if errorlevel 1 (
        echo ERROR: V6.2 training failed
        echo V6.2 training failed >> %LOG%
        goto error
    )
    echo V6.2 training completed successfully >> %LOG%
)

if "%RUN_REPORT%"=="1" (
    REM Generate Q4 ROI reports with reordered columns
    echo [REPORT] Generating Q4 ROI reports with betting metrics first...
    echo Generating Q4 ROI reports... >> %LOG%
    %PYTHON% match\training\generate_q4_roi_reordered.py >> %LOG% 2>&1
    if errorlevel 1 (
        echo ERROR: Q4 ROI generation failed
        echo Q4 ROI generation failed >> %LOG%
        goto error
    )
    echo Q4 ROI generation completed successfully >> %LOG%
)

echo.
echo ===== Pipeline completed successfully =====
echo Completed: %date% %time% >> %LOG%
echo. >> %LOG%

REM Open outputs directory
echo.
echo Opening results folder...
start "" "%WORKSPACE%\match\training\model_outputs_v6_2"

goto end

:error
echo.
echo ===== PIPELINE FAILED =====
echo Check %LOG% for details
echo. >> %LOG%
echo Failed: %date% %time% >> %LOG%
exit /b 1

:end
endlocal
exit /b 0
