# ============================================
# ML Platform - Cleanup Script
# ============================================

param(
    [switch]$Volumes,
    [switch]$Images,
    [switch]$All
)

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "ML Platform Cleanup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Change to infra directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath
Set-Location ..

if ($All) {
    Write-Host "`nWarning: This will remove containers, volumes, and images!" -ForegroundColor Red
    $confirm = Read-Host "Are you sure? (yes/no)"
    if ($confirm -ne "yes") {
        Write-Host "Cancelled." -ForegroundColor Yellow
        exit 0
    }
    
    Write-Host "`nStopping and removing containers..." -ForegroundColor Green
    docker compose down -v
    
    Write-Host "Removing images..." -ForegroundColor Green
    docker compose down --rmi all
    
    Write-Host "`n✓ Cleanup completed!" -ForegroundColor Green
    exit 0
}

if ($Volumes) {
    Write-Host "`nWarning: This will remove all volumes and data!" -ForegroundColor Red
    $confirm = Read-Host "Are you sure? (yes/no)"
    if ($confirm -ne "yes") {
        Write-Host "Cancelled." -ForegroundColor Yellow
        exit 0
    }
    
    Write-Host "`nRemoving volumes..." -ForegroundColor Green
    docker compose down -v
    Write-Host "`n✓ Volumes removed!" -ForegroundColor Green
    exit 0
}

if ($Images) {
    Write-Host "`nRemoving images..." -ForegroundColor Green
    docker compose down --rmi all
    Write-Host "`n✓ Images removed!" -ForegroundColor Green
    exit 0
}

# Default: just stop containers
Write-Host "`nStopping containers..." -ForegroundColor Green
docker compose down
Write-Host "`n✓ Containers stopped!" -ForegroundColor Green
