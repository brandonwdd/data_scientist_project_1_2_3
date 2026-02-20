# ============================================
# ML Platform - Start Script (PowerShell)
# ============================================

param(
    [switch]$Build,
    [switch]$Detached = $true
)

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Starting ML Platform Infrastructure" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "Warning: .env file not found. Using defaults." -ForegroundColor Yellow
    Write-Host "Copy env.development.example to .env and customize if needed." -ForegroundColor Yellow
}

# Change to infra directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath
Set-Location ..

# Check Docker
Write-Host "`nChecking Docker..." -ForegroundColor Green
try {
    docker --version | Out-Null
    docker compose version | Out-Null
} catch {
    Write-Host "Error: Docker or Docker Compose not found!" -ForegroundColor Red
    exit 1
}

# Build images if requested
if ($Build) {
    Write-Host "`nBuilding images..." -ForegroundColor Green
    docker compose build
}

# Start services
Write-Host "`nStarting services..." -ForegroundColor Green
if ($Detached) {
    docker compose up -d
} else {
    docker compose up
}

# Wait for services to be healthy
Write-Host "`nWaiting for services to be healthy..." -ForegroundColor Green
Start-Sleep -Seconds 5

# Check service status
Write-Host "`nService Status:" -ForegroundColor Cyan
docker compose ps

# Display access information
Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "Services are starting. Access URLs:" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "PostgreSQL:  localhost:5432" -ForegroundColor Yellow
Write-Host "Redis:       localhost:6379" -ForegroundColor Yellow
Write-Host "MLflow UI:   http://localhost:5000" -ForegroundColor Yellow
Write-Host "Prometheus:  http://localhost:9090" -ForegroundColor Yellow
Write-Host "`nTo view logs: docker compose logs -f [service_name]" -ForegroundColor Gray
Write-Host "To stop:      docker compose down" -ForegroundColor Gray
