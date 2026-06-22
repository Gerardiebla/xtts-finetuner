# XTTS Studio Setup Script
# Run this once to install all dependencies
# Usage: powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
cd $Root

Write-Host "=== XTTS Studio Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "Checking Python 3.11+..." -ForegroundColor Yellow
python --version 2>&1 | Select-Object -First 1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Please install Python 3.11+ from https://www.python.org/" -ForegroundColor Red
    exit 1
}

# Create venv
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv_tts")) {
    python -m venv venv_tts
    Write-Host "venv_tts created." -ForegroundColor Green
}

# Activate venv
$pip = ".\venv_tts\Scripts\pip.exe"

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
& $pip install --upgrade pip

# Install torch (CUDA 12.4)
Write-Host ""
Write-Host "Installing PyTorch with CUDA 12.4 (this may take a few minutes)..." -ForegroundColor Yellow
& $pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install PyTorch. Check your internet connection." -ForegroundColor Red
    exit 1
}

# Install other dependencies
Write-Host ""
Write-Host "Installing XTTS Studio dependencies..." -ForegroundColor Yellow
& $pip install coqui-tts transformers==4.57.1 faster-whisper librosa soundfile numpy spacy

# Verify CUDA
Write-Host ""
Write-Host "Verifying CUDA setup..." -ForegroundColor Yellow
& ".\venv_tts\Scripts\python.exe" -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: CUDA not available. Training will be very slow.')
"

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "You can now run: XTTS_Studio.bat" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to close"
