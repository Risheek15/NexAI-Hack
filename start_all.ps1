# Maintenance Copilot - One-Click Launcher for Windows PowerShell
# Starts FastAPI Backend on 127.0.0.1:8000 and Next.js Frontend on http://localhost:3000

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Starting Maintenance Copilot (FastAPI + Next.js UI)    " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$RootPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootPath

# 1. Start FastAPI Backend on port 8000
Write-Host "`n[1/2] Starting FastAPI Backend on http://127.0.0.1:8000..." -ForegroundColor Green
$BackendProcess = Start-Process -FilePath ".\venv\Scripts\uvicorn.exe" `
    -ArgumentList "src.api.main:app --host 127.0.0.1 --port 8000" `
    -PassThru -NoNewWindow

# 2. Start Next.js Frontend on port 3000
Write-Host "[2/2] Starting Next.js Frontend on http://localhost:3000..." -ForegroundColor Green
Set-Location "$RootPath\frontend"
$FrontendProcess = Start-Process -FilePath "npm.cmd" `
    -ArgumentList "run start" `
    -PassThru -NoNewWindow

Set-Location $RootPath

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "  Maintenance Copilot is Live!                            " -ForegroundColor Cyan
Write-Host "  - Frontend UI:  http://localhost:3000                   " -ForegroundColor White
Write-Host "  - Backend API:  http://127.0.0.1:8000                   " -ForegroundColor White
Write-Host "  - API Swagger:  http://127.0.0.1:8000/docs              " -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to terminate both servers.`n" -ForegroundColor Yellow

try {
    Wait-Process -Id $BackendProcess.Id, $FrontendProcess.Id
} finally {
    Write-Host "`nShutting down servers..." -ForegroundColor Yellow
    Stop-Process -Id $BackendProcess.Id -ErrorAction SilentlyContinue
    Stop-Process -Id $FrontendProcess.Id -ErrorAction SilentlyContinue
    Write-Host "Servers terminated cleanly." -ForegroundColor Green
}
