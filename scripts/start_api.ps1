# Start the PathSnap API (Laptop 1, Windows PowerShell).  Run from anywhere:
#   .\scripts\start_api.ps1 test     -> serves data\student_test   (rehearsal / trial)
#   .\scripts\start_api.ps1 eval     -> serves data\eval_trips     (the real evaluation)
#   .\scripts\start_api.ps1 "D:\some\other\folder"   -> any folder that holds trip folders
param([string]$Mode = "test", [int]$Port = 8000)
Set-Location (Join-Path $PSScriptRoot "..")
if ($Mode -eq "test") {
  # clean layout first, then the layout you get by unzipping "Student folder.zip" straight into data\
  if (Test-Path "data\student_test") { $Trips = "data\student_test" } else { $Trips = "data\Student folder\student_test" }
}
elseif ($Mode -eq "eval") { $Trips = "data\eval_trips" }
else { $Trips = $Mode }
if (-not (Test-Path $Trips)) { Write-Host "Folder not found: $Trips" -ForegroundColor Red; exit 1 }
$n = (Get-ChildItem $Trips -Directory).Count
Write-Host "Mode: $Mode | trips folder: $Trips | sub-folders: $n | port: $Port" -ForegroundColor Cyan
$env:PATHSNAP_EVAL_DIR = $Trips
$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } | Select-Object -First 1).IPAddress
Write-Host "Give Laptop 2 this API address:  http://${ip}:$Port" -ForegroundColor Green
python -m uvicorn src.api.app:app --host 0.0.0.0 --port $Port
