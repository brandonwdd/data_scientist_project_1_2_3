# ============================================
# ML Platform - Health Check Script
# ============================================

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "ML Platform Health Check" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Change to infra directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath
Set-Location ..

$allHealthy = $true

# Check Docker Compose services
Write-Host "`nChecking Docker Compose services..." -ForegroundColor Green
$services = docker compose ps --format json | ConvertFrom-Json
foreach ($service in $services) {
    $status = $service.State
    $name = $service.Service
    if ($status -eq "running") {
        Write-Host "  ✓ $name : $status" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $name : $status" -ForegroundColor Red
        $allHealthy = $false
    }
}

# Check PostgreSQL
Write-Host "`nChecking PostgreSQL..." -ForegroundColor Green
try {
    $env:PGPASSWORD = "dsplatform_dev"
    $result = docker compose exec -T postgres psql -U dsplatform -d dsplatform -c "SELECT 1;" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ PostgreSQL is accessible" -ForegroundColor Green
    } else {
        Write-Host "  ✗ PostgreSQL connection failed" -ForegroundColor Red
        $allHealthy = $false
    }
} catch {
    Write-Host "  ✗ PostgreSQL check failed: $_" -ForegroundColor Red
    $allHealthy = $false
}

# Check Redis
Write-Host "`nChecking Redis..." -ForegroundColor Green
try {
    $result = docker compose exec -T redis redis-cli -a redis_dev ping 2>&1
    if ($result -match "PONG") {
        Write-Host "  ✓ Redis is accessible" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Redis connection failed" -ForegroundColor Red
        $allHealthy = $false
    }
} catch {
    Write-Host "  ✗ Redis check failed: $_" -ForegroundColor Red
    $allHealthy = $false
}

# Check MLflow
Write-Host "`nChecking MLflow..." -ForegroundColor Green
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host "  ✓ MLflow UI is accessible" -ForegroundColor Green
    } else {
        Write-Host "  ✗ MLflow returned status: $($response.StatusCode)" -ForegroundColor Red
        $allHealthy = $false
    }
} catch {
    Write-Host "  ✗ MLflow check failed: $_" -ForegroundColor Red
    $allHealthy = $false
}

# Check Prometheus
Write-Host "`nChecking Prometheus..." -ForegroundColor Green
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host "  ✓ Prometheus is accessible" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Prometheus returned status: $($response.StatusCode)" -ForegroundColor Red
        $allHealthy = $false
    }
} catch {
    Write-Host "  ✗ Prometheus check failed: $_" -ForegroundColor Red
    $allHealthy = $false
}

# Summary
Write-Host "`n============================================" -ForegroundColor Cyan
if ($allHealthy) {
    Write-Host "All services are healthy! ✓" -ForegroundColor Green
    exit 0
} else {
    Write-Host "Some services are unhealthy! ✗" -ForegroundColor Red
    exit 1
}
