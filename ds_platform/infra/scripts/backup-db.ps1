# ============================================
# ML Platform - Database Backup Script
# ============================================

param(
    [string]$OutputPath = "backups",
    [string]$BackupName = "dsplatform_$(Get-Date -Format 'yyyyMMdd_HHmmss').sql"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Backing up ML Platform Database" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Change to infra directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath
Set-Location ..

# Create backup directory
$backupDir = Join-Path $PWD $OutputPath
if (-not (Test-Path $backupDir)) {
    New-Item -ItemType Directory -Path $backupDir | Out-Null
    Write-Host "Created backup directory: $backupDir" -ForegroundColor Green
}

$backupFile = Join-Path $backupDir $BackupName

Write-Host "`nBacking up database..." -ForegroundColor Green
Write-Host "Output: $backupFile" -ForegroundColor Gray

try {
    $env:PGPASSWORD = "dsplatform_dev"
    docker compose exec -T postgres pg_dump -U dsplatform -d dsplatform --clean --if-exists > $backupFile
    
    if ($LASTEXITCODE -eq 0) {
        $fileSize = (Get-Item $backupFile).Length / 1MB
        Write-Host "`n✓ Backup completed successfully!" -ForegroundColor Green
        Write-Host "  File: $backupFile" -ForegroundColor Gray
        Write-Host "  Size: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Gray
    } else {
        Write-Host "`n✗ Backup failed!" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "`n✗ Backup failed: $_" -ForegroundColor Red
    exit 1
}
