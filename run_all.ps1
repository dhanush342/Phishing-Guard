$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendPath = Join-Path $projectRoot "backend"

Start-Process -FilePath "c:\Windows\py.exe" -ArgumentList "-m uvicorn app:app --host 127.0.0.1 --port 8000" -WorkingDirectory $backendPath
Start-Process -FilePath (Join-Path $projectRoot "index.html")

Write-Host "API running at http://127.0.0.1:8000 and UI opened in browser."
