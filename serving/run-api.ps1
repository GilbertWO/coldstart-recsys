# Starts the API with POSTGRES_PASSWORD loaded from serving/.env (the value is never printed).
# Usage from the repo root:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\serving\run-api.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\serving\run-api.ps1 --recs.list-cache-enabled=true
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$AppArgs = @())

$envFile = Join-Path $PSScriptRoot '.env'
if (-not (Test-Path $envFile)) { Write-Host "missing $envFile (copy .env.example and set POSTGRES_PASSWORD)"; exit 1 }

Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*([^#=\s]+)\s*=\s*(.*)$') { Set-Item -Path "Env:$($matches[1])" -Value $matches[2].Trim() }
}
if (-not $env:POSTGRES_PASSWORD) { Write-Host 'POSTGRES_PASSWORD is empty or missing in .env; not starting'; exit 1 }
Write-Host 'password is set; starting the API'

Set-Location (Join-Path $PSScriptRoot 'api')
if ($AppArgs.Count -gt 0) {
    .\mvnw spring-boot:run "-Dspring-boot.run.arguments=$($AppArgs -join ' ')"
} else {
    .\mvnw spring-boot:run
}
