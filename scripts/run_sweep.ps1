# Pareto sweep over k = {0, 2, 5} for the two production-candidate base models.
#
# For each base model:
#   1. Skip rescaling if best_model_obj_k{2,5}.keras already exist (idempotent).
#      Otherwise, call rescale_obj_logits.py to create them.
#   2. Run chain_eqe at k=0, k=2, k=5 (each evaluates float, quantizes, evaluates int8).
#   3. Parse the float / int8 mAP / precision / recall lines from each run's output.
#   4. Print a [RESULT] line as soon as each chain_eqe finishes.
#   5. After all six runs, print a final summary table and write it to
#      pareto_sweep_results.csv at the repo root.
#
# Run from repo root:
#   pwsh ./scripts/run_pareto_sweep.ps1
# or
#   powershell -ExecutionPolicy Bypass -File ./scripts/run_pareto_sweep.ps1
#
# Total expected wall time: ~10-15 minutes (six 1-2 min chain_eqe runs + two rescales).

$ErrorActionPreference = "Continue"

# Resolve repo root (assume this script lives in repo_root/scripts/)
$RepoRoot = Split-Path -Parent $PSScriptRoot

# Two production-candidate base models. Add more entries here to extend the sweep.
$Models = @(
    [PSCustomObject]@{
        Name        = "192v1"
        BaseKeras   = Join-Path $RepoRoot "object_detection/tf/src/experiments_outputs/2026_05_07_10_02_38/saved_models/best_model.keras"
        ConfigK0    = "my_chain_eqe_k0_192v1"
        ConfigK2    = "my_chain_eqe_obj_rescaled_k2_192v1"
        ConfigK5    = "my_chain_eqe_obj_rescaled_192v1"
    },
    [PSCustomObject]@{
        Name        = "256_default"
        BaseKeras   = Join-Path $RepoRoot "object_detection/tf/src/experiments_outputs/2026_05_08_20_48_06/saved_models/best_model.keras"
        ConfigK0    = "my_chain_eqe_k0_256_default"
        ConfigK2    = "my_chain_eqe_obj_rescaled_k2_256_default"
        ConfigK5    = "my_chain_eqe_obj_rescaled_k5_256_default"
    },
    [PSCustomObject]@{
        Name        = "256_default_val_map"
        BaseKeras   = Join-Path $RepoRoot "object_detection/tf/src/experiments_outputs/2026_05_09_02_47_31/saved_models/best_model.keras"
        ConfigK0    = "my_chain_eqe_k0_256_default_val_map"
        ConfigK2    = "my_chain_eqe_k2_256_default_val_map"
        ConfigK5    = "my_chain_eqe_k5_256_default_val_map"
    },
    [PSCustomObject]@{
      Name        = "192_default_val_map"
      BaseKeras   = Join-Path $RepoRoot "object_detection/tf/src/experiments_outputs/2026_05_10_09_45_46/saved_models/best_model.keras"
      ConfigK0    = "my_chain_eqe_k0_192_default_val_map"
      ConfigK2    = "my_chain_eqe_k2_192_default_val_map"
      ConfigK5    = "my_chain_eqe_k5_192_default_val_map"
    }
)

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Result {
    param([string]$Message)
    Write-Host "[RESULT] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Invoke-Rescale {
    param(
        [string]$BaseKeras,
        [int]$K
    )
    $OutPath = $BaseKeras -replace "best_model\.keras$", "best_model_obj_k${K}.keras"
    if (Test-Path $OutPath) {
        Write-Info "k=$K rescaled model already exists at $OutPath -- skipping rescale"
        return
    }
    Write-Info "Creating k=$K rescaled model at $OutPath"
    $RescaleScript = Join-Path $RepoRoot "scripts/rescale_obj_logits.py"
    & python $RescaleScript --in $BaseKeras --out $OutPath --k $K
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "rescale_obj_logits.py exited with code $LASTEXITCODE for k=$K"
    }
}

function Invoke-ChainEqe {
    param(
        [string]$ModelName,
        [int]$K,
        [string]$ConfigName
    )
    Write-Info "$ModelName : running chain_eqe with config '$ConfigName' (k=$K)"

    $ObjDetDir = Join-Path $RepoRoot "object_detection"
    Push-Location $ObjDetDir
    try {
        # Capture stdout+stderr for parsing while still streaming to console
        $tmpLog = [System.IO.Path]::GetTempFileName()
        & python stm32ai_main.py --config-name $ConfigName 2>&1 | Tee-Object -FilePath $tmpLog | Write-Host
        $exitCode = $LASTEXITCODE
        $output = Get-Content $tmpLog -Raw
        Remove-Item $tmpLog -ErrorAction SilentlyContinue

        if ($exitCode -ne 0) {
            Write-Warn "$ModelName k=$K chain_eqe exited with code $exitCode"
        }

        # Parse the two "| stop_sign | ... |" rows.
        # First occurrence = float eval, second = int8 eval.
        $stopLines = ($output -split "`r?`n") | Where-Object { $_ -match "\|\s*stop_sign\s*\|" }
        if ($stopLines.Count -lt 2) {
            Write-Warn "$ModelName k=$K : could not find both float and int8 result rows in output (found $($stopLines.Count))"
            return $null
        }

        function Parse-Row {
            param([string]$Row)
            $parts = ($Row -split "\|") | ForEach-Object { $_.Trim() }
            # parts[0] = "" (before first |), parts[1] = "stop_sign", parts[2] = prec, parts[3] = recall, parts[4] = AP
            return [PSCustomObject]@{
                Precision = $parts[2]
                Recall    = $parts[3]
                AP        = $parts[4]
            }
        }

        $floatRow = Parse-Row $stopLines[0]
        $int8Row  = Parse-Row $stopLines[1]

        Write-Result "$ModelName k=$K  -->  float (P=$($floatRow.Precision) R=$($floatRow.Recall) mAP=$($floatRow.AP))   int8 (P=$($int8Row.Precision) R=$($int8Row.Recall) mAP=$($int8Row.AP))"

        return [PSCustomObject]@{
            Model         = $ModelName
            K             = $K
            FloatPrec     = $floatRow.Precision
            FloatRec      = $floatRow.Recall
            FloatMap      = $floatRow.AP
            Int8Prec      = $int8Row.Precision
            Int8Rec       = $int8Row.Recall
            Int8Map       = $int8Row.AP
        }
    }
    finally {
        Pop-Location
    }
}

# ============================================================================
# Main
# ============================================================================

$startTime = Get-Date
Write-Host ""
Write-Info "Starting Pareto sweep at $startTime"
Write-Info "Repo root: $RepoRoot"
Write-Info "Sweeping $($Models.Count) base models x 3 k-values = $($Models.Count * 3) chain_eqe runs"
Write-Host ""

$AllResults = @()

foreach ($model in $Models) {
    Write-Host ""
    Write-Host "====================================================================" -ForegroundColor Magenta
    Write-Host "  $($model.Name)" -ForegroundColor Magenta
    Write-Host "====================================================================" -ForegroundColor Magenta

    if (-not (Test-Path $model.BaseKeras)) {
        Write-Warn "Base model not found: $($model.BaseKeras) -- skipping $($model.Name)"
        continue
    }

    # 1) Rescale (idempotent)
    Invoke-Rescale -BaseKeras $model.BaseKeras -K 2
    Invoke-Rescale -BaseKeras $model.BaseKeras -K 5

    # 2) chain_eqe for k=0, 2, 5
    foreach ($entry in @(@{K=0; Cfg=$model.ConfigK0}, @{K=2; Cfg=$model.ConfigK2}, @{K=5; Cfg=$model.ConfigK5})) {
        Write-Host ""
        $result = Invoke-ChainEqe -ModelName $model.Name -K $entry.K -ConfigName $entry.Cfg
        if ($null -ne $result) {
            $AllResults += $result
        }
    }
}

# 3) Final summary
$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host ""
Write-Host "====================================================================" -ForegroundColor Green
Write-Host "  FINAL SUMMARY" -ForegroundColor Green
Write-Host "====================================================================" -ForegroundColor Green
Write-Info "Total wall time: $($duration.ToString('hh\:mm\:ss'))"
Write-Host ""

if ($AllResults.Count -eq 0) {
    Write-Warn "No results collected -- check warnings above"
    exit 1
}

$AllResults | Format-Table -AutoSize Model, K, FloatPrec, FloatRec, FloatMap, Int8Prec, Int8Rec, Int8Map

# 4) Save CSV at repo root
$CsvPath = Join-Path $RepoRoot "pareto_sweep_results.csv"
$AllResults | Export-Csv -NoTypeInformation -Path $CsvPath
Write-Info "Wrote results to $CsvPath"

# 5) Highlight best int8 deployment pick
$best = $AllResults | Sort-Object { [double]$_.Int8Map } -Descending | Select-Object -First 1
Write-Host ""
Write-Result "Best int8 deployment pick: $($best.Model) k=$($best.K) -- int8 mAP=$($best.Int8Map)"
