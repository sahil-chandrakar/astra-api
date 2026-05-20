$ErrorActionPreference = "Stop"

$backendRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $backendRoot

$python = Join-Path $backendRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    py -3.11 -m venv .venv
}

& $python -m pip install -r requirements.txt
& $python -c "import playwright.async_api" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $python -m pip install playwright
}
& $python -m playwright install chromium

& $python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
