[CmdletBinding()]
param(
    [switch]$InstallTraining,
    [switch]$NoPull,
    [switch]$NoDashboard,
    [switch]$TestOnly,
    [ValidateSet("3.11", "3.12")][string]$PythonVersion = "3.11",
    [ValidateRange(1, 65535)][int]$DashboardPort = 8766,
    [ValidateSet("none", "sft", "grpo")][string]$Train = "none",
    [int]$MaxSteps = 1
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Executable" }
}
try {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "Install Git for Windows first." }
    if (-not ($NoPull -or $TestOnly)) {
        Invoke-Checked -Executable "git" -Arguments @("pull", "--ff-only")
    }
    $Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $Python)) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py "-$PythonVersion" -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) {
                Invoke-Checked -Executable "py" -Arguments @("-$PythonVersion", "-m", "venv", ".venv")
            } else {
                $Fallback = if ($PythonVersion -eq "3.11") { "-3.12" } else { "-3.11" }
                Invoke-Checked -Executable "py" -Arguments @($Fallback, "-m", "venv", ".venv")
            }
        } elseif (Get-Command python -ErrorAction SilentlyContinue) {
            Invoke-Checked -Executable "python" -Arguments @("-m", "venv", ".venv")
        } else { throw "Install Python 3.11+ (64-bit) and enable the Python launcher." }
    }
    Invoke-Checked -Executable $Python -Arguments @("-c", "import sys; assert sys.version_info[:2] in ((3,11),(3,12)), 'Python 3.11 or 3.12 required'")
    Invoke-Checked -Executable $Python -Arguments @("-m", "pip", "install", "-e", ".[test]")
    Invoke-Checked -Executable $Python -Arguments @("-m", "pytest", "-v")
    if ($InstallTraining) {
        if (-not (Test-Path "training\requirements-train.txt")) { throw "Missing training scaffold; fetch the complete handoff branch." }
        Invoke-Checked -Executable $Python -Arguments @("-m", "pip", "install", "torch>=2.7,<3", "--index-url", "https://download.pytorch.org/whl/cu128")
        Invoke-Checked -Executable $Python -Arguments @("-m", "pip", "install", "-r", "training/requirements-train.txt")
        Invoke-Checked -Executable $Python -Arguments @("-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0), torch.version.cuda); assert tuple(map(int,torch.version.cuda.split('.')[:2])) >= (12,8), 'CUDA 12.8+ required for this handoff'")
    }
    if ($Train -ne "none") {
        if ($MaxSteps -lt 1) { throw "MaxSteps must be positive." }
        Invoke-Checked -Executable $Python -Arguments @("-m", "training.train_$Train", "--max-steps", "$MaxSteps")
    }
    $env:PATH = "$(Split-Path -Parent $Python);$env:PATH"
    Write-Host "Aegis ready. Virtual environment: $Python"
    if (-not $TestOnly) {
        $InitOutput = & $Python -X utf8 -m luna.cli init --repo $RepoRoot --json
        $InitExit = $LASTEXITCODE
        if ($InitExit -notin @(0, 2) -or -not $InitOutput) { throw "Luna init failed ($InitExit)." }
        $Init = ($InitOutput -join "`n") | ConvertFrom-Json
        if (-not $Init.keys.fingerprint -or -not (Test-Path -LiteralPath $Init.template)) {
            throw "Luna init did not verify the signing key and configuration template."
        }
        $InitOutput | Write-Host
        if (-not $NoDashboard) {
            Invoke-Checked -Executable $Python -Arguments @("-m", "luna.cli", "dashboard", "--repo", $RepoRoot, "--port", "$DashboardPort")
        }
    }
} finally { Pop-Location }
