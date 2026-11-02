[CmdletBinding()]
param(
    [switch]$Full
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Assert-LastExitCode {
    param([string]$Name)
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Write-Step "Required files"
$required = @(
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "CHANGELOG.md",
    "AGENTS.md",
    "CLAUDE.md",
    "docs/MASTER_DOCUMENT.md"
)
foreach ($file in $required) {
    if (-not (Test-Path $file)) {
        throw "Missing required file: $file"
    }
}

Write-Step "Docs/config validation"
uv run python scripts/check_repo.py
Assert-LastExitCode "check_repo.py"

Write-Step "Ruff lint"
uv run ruff check .
Assert-LastExitCode "ruff check"

if ($Full) {
    Write-Step "Pytest suite (full, includes slow calibration/transfer tests)"
    uv run pytest -q
} else {
    Write-Step "Pytest suite (fast, excludes slow calibration/transfer tests)"
    uv run pytest -q -m "not slow"
}
Assert-LastExitCode "pytest"

Write-Step "Whitespace"
$base = "HEAD"
git rev-parse --verify --quiet origin/develop | Out-Null
if ($LASTEXITCODE -eq 0) {
    $mergeBase = git merge-base HEAD origin/develop
    if ($LASTEXITCODE -eq 0) {
        $base = $mergeBase
    }
}
git diff --check $base
Assert-LastExitCode "git diff --check"

Write-Host "`nLocal CI passed." -ForegroundColor Green
