param(
    [switch]$Yes,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Run scripts\setup\01_create_environment.bat first.'
}

Set-Location -LiteralPath $repoRoot
if (-not $Yes -and -not $DryRun) {
    $confirmation = Read-Host 'Type TRAIN_NEW_TEN to continue'
    if ($confirmation -cne 'TRAIN_NEW_TEN') {
        Write-Host 'Cancelled.'
        exit 3
    }
}

Write-Host 'Verifying native Windows backends and pinned pretrained weights...'
& $python -m tools.setup.backends
if ($LASTEXITCODE -ne 0) {
    throw 'Backend verification failed. Run scripts\setup\02_setup_backends.bat.'
}

if ($DryRun) {
    Write-Host 'Dry-running only the 10 pending NIR extension jobs...'
    & $python -m tools.workflow.train_new_nir
    exit $LASTEXITCODE
}

Write-Host 'Starting or safely resuming only the 10 pending NIR extension jobs...'
& $python -m tools.workflow.train_new_nir --execute-training
exit $LASTEXITCODE
