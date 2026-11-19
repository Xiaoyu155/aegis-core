[CmdletBinding()]
param(
    [switch]$InstallTraining,
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
    $Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $Python)) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            Invoke-Checked -Executable "py" -Arguments @("-3.11", "-m", "venv", ".venv")
        } elseif (Get-Command python -ErrorAction SilentlyContinue) {
            Invoke-Checked -Executable "python" -Arguments @("-m", "venv", ".venv")
        } else { throw "Install Python 3.11+ (64-bit) and enable the Python launcher." }
    }
    Invoke-Checked -Executable $Python -Arguments @("-c", "import sys; assert sys.version_info >= (3,11), 'Python 3.11+ required'")
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
} finally { Pop-Location }
