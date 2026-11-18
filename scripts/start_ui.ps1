# Serve the evaluator UI (Laptop 2, Windows PowerShell).  Usage: .\scripts\start_ui.ps1 http://<LAPTOP1_IP>:8000
param([string]$Api = "http://localhost:8000", [int]$Port = 8080)
Set-Location (Join-Path $PSScriptRoot "..")
$enc = [uri]::EscapeDataString($Api)
Write-Host "Open this address in the browser:  http://localhost:${Port}/?api=$enc" -ForegroundColor Green
python -m http.server $Port --directory src/ui
