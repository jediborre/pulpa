@echo off
REM Batch script to automate v6/v6.2 training and Q4 ROI report generation
REM Location: c:\Users\borre\OneDrive\OLD\Escritorio\pulpa\train_and_report.bat

setlocal enabledelayedexpansion

set WORKSPACE=c:\Users\borre\OneDrive\OLD\Escritorio\pulpa
set VENV=%WORKSPACE%\.venv\Scripts
set PYTHON=%VENV%\python.exe
set LOG=%WORKSPACE%\train_and_report.log

cd /d %WORKSPACE% || goto error

echo ===== Training & Reporting Pipeline ===== >> %LOG%
echo Started: %date% %time% >> %LOG%
echo. >> %LOG%

REM Activate venv (PowerShell approach for compatibility)
echo [1/4] Activating virtual environment...
call %VENV%\activate.bat

REM Train V6 baseline
echo [2/4] Training V6 baseline models...
echo Training V6... >> %LOG%
%PYTHON% match\training\train_q3_q4_models_v6.py >> %LOG% 2>&1
if errorlevel 1 (
    echo ERROR: V6 training failed
    echo V6 training failed >> %LOG%
    goto error
)
echo V6 training completed successfully >> %LOG%

REM Train V6.2 with external league exclusions
echo [3/4] Training V6.2 models (with external league exclusions)...
echo Training V6.2... >> %LOG%
%PYTHON% match\training\train_q3_q4_models_v6_2.py >> %LOG% 2>&1
if errorlevel 1 (
    echo ERROR: V6.2 training failed
    echo V6.2 training failed >> %LOG%
    goto error
)
echo V6.2 training completed successfully >> %LOG%

REM Generate Q4 ROI reports with reordered columns
echo [4/4] Generating Q4 ROI reports (betting metrics first)...
echo Generating Q4 ROI reports... >> %LOG%
%PYTHON% match\training\generate_q4_roi_reordered.py >> %LOG% 2>&1
if errorlevel 1 (
    echo ERROR: Q4 ROI generation failed
    echo Q4 ROI generation failed >> %LOG%
    goto error
)
echo Q4 ROI generation completed successfully >> %LOG%

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
