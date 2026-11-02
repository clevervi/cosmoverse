[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

Write-Step "GPU device audit"
uv run --extra gpu cosmoverse devices
if ($LASTEXITCODE -ne 0) {
    throw "GPU device audit failed with exit code $LASTEXITCODE"
}

Write-Step "GPU Particle-Mesh benchmark"
uv run --extra gpu cosmoverse gpu-pm --mesh 128 --steps 1
if ($LASTEXITCODE -ne 0) {
    throw "GPU PM benchmark failed with exit code $LASTEXITCODE"
}

Write-Step "GPU regression tests"
uv run --extra gpu --extra dev python -m pytest -q tests/test_acceleration.py
if ($LASTEXITCODE -ne 0) {
    throw "GPU tests failed with exit code $LASTEXITCODE"
}

Write-Host "`nGPU CI passed." -ForegroundColor Green
