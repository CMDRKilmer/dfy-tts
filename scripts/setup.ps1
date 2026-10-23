#requires -Version 5.1
<#
.SYNOPSIS
    Build the Piper training environment on Windows (no MSVC required).

.DESCRIPTION
    Creates .venv with Python 3.12, installs piper-tts[train], swaps the CPU
    torch build for the CUDA build matching this machine's driver, then applies
    the numba fallback for the monotonic_align Cython extension that the PyPI
    wheel does not contain.

.EXAMPLE
    pwsh -File scripts/setup.ps1
#>
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$PythonVersion = "3.12"
$TorchVersion = "2.14.1"
$CudaIndexUrl = "https://download.pytorch.org/whl/cu130"

Write-Host "==> creating .venv (Python $PythonVersion)" -ForegroundColor Cyan
uv venv .venv --python $PythonVersion

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

Write-Host "==> installing piper-tts[train] from PyPI" -ForegroundColor Cyan
uv pip install --python $VenvPython "piper-tts[train]"

Write-Host "==> installing CUDA build of torch $TorchVersion" -ForegroundColor Cyan
$LocalWheel = Get-ChildItem -Path $RepoRoot -Filter "torch-$TorchVersion+cu*-cp312-cp312-win_amd64.whl" -File -ErrorAction SilentlyContinue | Select-Object -First 1
if ($LocalWheel) {
    # A pre-downloaded wheel (e.g. fetched through a proxy) installs in seconds
    # and skips the dependency solve -- torch's deps are already in the venv.
    Write-Host "    using local wheel: $($LocalWheel.FullName)" -ForegroundColor Yellow
    uv pip install --python $VenvPython --no-deps --upgrade --reinstall $LocalWheel.FullName
} else {
    uv pip install --python $VenvPython --index-url $CudaIndexUrl --upgrade --reinstall "torch==$TorchVersion"
}

Write-Host "==> installing onnxscript (ONNX export dependency)" -ForegroundColor Cyan
uv pip install --python $VenvPython onnxscript

Write-Host "==> patching monotonic_align (numba fallback)" -ForegroundColor Cyan
& $VenvPython scripts/patch_monotonic_align.py

Write-Host "==> verifying" -ForegroundColor Cyan
& $VenvPython -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
& $VenvPython scripts/test_monotonic_align.py

Write-Host "Done. Activate with: .venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "Full end-to-end check: .venv\Scripts\python.exe scripts\smoke_test.py" -ForegroundColor Green
