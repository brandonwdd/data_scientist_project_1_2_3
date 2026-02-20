# ============================================
# ML Platform - Stop Script (PowerShell)
# ============================================

param(
    [switch]$RemoveVolumes
)

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Stopping ML Platform Infrastructure" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Change to infra directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath
Set-Location ..

# Stop services
Write-Host "`nStopping services..." -ForegroundColor Green
if ($RemoveVolumes) {
    Write-Host "Warning: This will remove all volumes and data!" -ForegroundColor Red
    $confirm = Read-Host "Are you sure? (yes/no)"
    if ($confirm -eq "yes") {
        docker compose down -v
        Write-Host "Volumes removed." -ForegroundColor Yellow
    } else {
        Write-Host "Cancelled." -ForegroundColor Yellow
        exit 0
    }
} else {
    docker compose down
}

Write-Host "`nServices stopped." -ForegroundColor Green
