$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$ProgressPreference = "SilentlyContinue"

$root = "C:\Users\borre\OneDrive\OLD\Escritorio\pulpa\match"
Set-Location $root
$rep = Join-Path $root "training\v15\reports"

function Run-Config {
    param(
        [string]$Name,
        [int]$TrainDays,
        [int]$ValDays,
        [int]$CalDays,
        [int]$HoldoutDays,
        [int]$MinTrain,
        [int]$ActiveDays
    )
    Write-Host "=============================================================="
    Write-Host "[sweep] $Name  (train=$TrainDays  val=$ValDays  cal=$CalDays  holdout=$HoldoutDays  min=$MinTrain  active=$ActiveDays)"
    Write-Host "=============================================================="
    $trainOut = python -u -m training.v15.cli train `
        --train-days $TrainDays --val-days $ValDays `
        --cal-days $CalDays --holdout-days $HoldoutDays `
        --min-samples-train $MinTrain --active-days $ActiveDays 2>&1 | Out-String
    ($trainOut -split "`r?`n") | Where-Object {
        $_ -match "split temporal|filtro ligas|ligas candidatas|trained \d+|summary ->|pace thresholds|  \[train\]|  \[val\]|  \[cal\]|  \[holdout\]"
    } | Select-Object -Last 40 | ForEach-Object { Write-Host $_ }

    $roiOut = python -u -m training.v15.cli test-roi --odds 1.40 --min-bets 5 --top 30 2>&1 | Out-String
    ($roiOut -split "`r?`n") | Where-Object {
        $_ -match "apuestas|hit rate|ROI|P&L|portfolio|GLOBAL|PORTFOLIO|target hit|muestras holdout|break-even"
    } | Select-Object -First 25 | ForEach-Object { Write-Host $_ }

    if (Test-Path "$rep\test_roi_v15.json") {
        Copy-Item "$rep\test_roi_v15.json" "$rep\test_roi_v15_$Name.json" -Force
    }
    if (Test-Path "training\v15\model_outputs\training_summary_v15.json") {
        Copy-Item "training\v15\model_outputs\training_summary_v15.json" "training\v15\model_outputs\training_summary_v15_$Name.json" -Force
    }
}

Run-Config -Name "A_baseline"   -TrainDays 50 -ValDays 20 -CalDays 13 -HoldoutDays 7 -MinTrain 300 -ActiveDays 0
Run-Config -Name "B_med_train"  -TrainDays 65 -ValDays 15 -CalDays 3  -HoldoutDays 7 -MinTrain 200 -ActiveDays 14
Run-Config -Name "C_big_train"  -TrainDays 73 -ValDays 8  -CalDays 2  -HoldoutDays 7 -MinTrain 200 -ActiveDays 14

Write-Host ""
Write-Host "=============================================================="
Write-Host "[sweep] COMPARATIVA FINAL"
Write-Host "=============================================================="
python "$rep\_compare_sweep.py"
