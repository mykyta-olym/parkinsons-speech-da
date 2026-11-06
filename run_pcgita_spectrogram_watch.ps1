param(
    [string]$Pairs = "multi->words,multi->sentences2,multi->read_text,multi->monologue",
    [string]$Methods = "cnn_source,cnn_dann,cnn_mmd,cnn_coral",
    [string]$Seeds = "42,7",
    [int]$Epochs = 18,
    [int]$BatchSize = 32
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $PSScriptRoot "logs\pcgita_spectrogram_$stamp.log"

$env:PYTHONUNBUFFERED = "1"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (!(Test-Path $python)) {
    $python = "python"
}

Write-Host "Log: $log"
Write-Host "Progress: data\pcgita_results\pcgita_spectrogram_progress.txt"
Write-Host "Partial CSV: data\pcgita_results\pcgita_spectrogram_results_partial.csv"

& $python -u run_pcgita_spectrogram_experiments.py `
    --methods $Methods `
    --pairs $Pairs `
    --seeds $Seeds `
    --epochs $Epochs `
    --batch-size $BatchSize 2>&1 | Tee-Object -FilePath $log

exit $LASTEXITCODE
