# Start the PathSnap API for the evaluation (Windows PowerShell).
# Usage:  .\scripts\run_api.ps1 [trips_folder] [port]
# The trips folder may be the folder with the trip sub-folders, or its parent (a wrapper folder is fine).
param([string]$Trips = "data\eval_trips", [int]$Port = 8000)
Set-Location (Join-Path $PSScriptRoot "..")
$env:PATHSNAP_EVAL_DIR = $Trips
Write-Host "Serving trips from: $Trips   on port $Port"
python -m uvicorn src.api.app:app --host 0.0.0.0 --port $Port
