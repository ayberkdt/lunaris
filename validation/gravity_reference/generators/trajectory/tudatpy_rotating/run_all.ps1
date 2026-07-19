param(
    [string]$Scenario = 'scenario.json',
    [string]$Results = 'outputs\validation\tudat\results_1day',
    [string]$RepoRoot = '',
    [Parameter(Mandatory = $true)][string]$Mamba,
    [Parameter(Mandatory = $true)][string]$TudatPrefix,
    [Parameter(Mandatory = $true)][string]$LunarisPython,
    [string]$MambaRoot = ''
)

$ErrorActionPreference = 'Stop'

$ValidationRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $RepoRoot) {
    $RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $ValidationRoot '..\..\..\..\..'))
}
$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
if (-not [System.IO.Path]::IsPathRooted($Mamba)) {
    $Mamba = [System.IO.Path]::GetFullPath($Mamba)
}
if (-not [System.IO.Path]::IsPathRooted($TudatPrefix)) {
    $TudatPrefix = [System.IO.Path]::GetFullPath($TudatPrefix)
}
if (-not [System.IO.Path]::IsPathRooted($LunarisPython)) {
    $LunarisPython = Join-Path $RepoRoot $LunarisPython
}
if (-not $MambaRoot) {
    $MambaRoot = Join-Path (Split-Path -Parent $TudatPrefix) '.mamba-root'
}

if (-not [System.IO.Path]::IsPathRooted($Scenario)) {
    $Scenario = if (Test-Path -LiteralPath $Scenario) {
        (Resolve-Path -LiteralPath $Scenario).Path
    } else {
        Join-Path $ValidationRoot $Scenario
    }
}
if (-not [System.IO.Path]::IsPathRooted($Results)) {
    $Results = Join-Path $RepoRoot $Results
}
$env:LUNARIS_TUDAT_SCENARIO = [System.IO.Path]::GetFullPath($Scenario)
$env:LUNARIS_TUDAT_RESULTS = [System.IO.Path]::GetFullPath($Results)
$env:LUNARIS_REPO_ROOT = [System.IO.Path]::GetFullPath($RepoRoot)

if (-not (Test-Path $Mamba)) { throw "micromamba not found: $Mamba" }
if (-not (Test-Path $LunarisPython)) { throw "Lunaris Python not found: $LunarisPython" }

Push-Location $ValidationRoot
try {
    & $Mamba run --root-prefix $MambaRoot --prefix $TudatPrefix python run_tudat.py
    if ($LASTEXITCODE -ne 0) { throw "Tudat run failed with exit code $LASTEXITCODE" }

    & $LunarisPython run_lunaris.py
    if ($LASTEXITCODE -ne 0) { throw "Lunaris run failed with exit code $LASTEXITCODE" }

    & $LunarisPython compare_results.py
    if ($LASTEXITCODE -ne 0) { throw "Comparison failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
    Remove-Item Env:LUNARIS_TUDAT_SCENARIO -ErrorAction SilentlyContinue
    Remove-Item Env:LUNARIS_TUDAT_RESULTS -ErrorAction SilentlyContinue
    Remove-Item Env:LUNARIS_REPO_ROOT -ErrorAction SilentlyContinue
}
