# ============================================
# ML Platform - Database Restore Script
# ============================================

param(
    [Parameter(Mandatory=$true)]
    [string]$BackupFile
)

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Restoring ML Platform Database" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Check if backup file exists
if (-not (Test-Path $BackupFile)) {
    Write-Host "Error: Backup file not found: $BackupFile" -ForegroundColor Red
    exit 1
}

Write-Host "`nWarning: This will overwrite the current database!" -ForegroundColor Red
$confirm = Read-Host "Are you sure? (yes/no)"
if ($confirm -ne "yes") {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

# Change to infra directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath
Set-Location ..

Write-Host "`nRestoring database from: $BackupFile" -ForegroundColor Green

try {
    $env:PGPASSWORD = "dsplatform_dev"
    Get-Content $BackupFile | docker compose exec -T postgres psql -U dsplatform -d dsplatform
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ Restore completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "`n✗ Restore failed!" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "`n✗ Restore failed: $_" -ForegroundColor Red
    exit 1
}
