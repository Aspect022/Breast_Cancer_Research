param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

Write-Host "=== RTX 5050 Genomics Environment Setup ==="

if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv..."
    & $Python -m venv .venv
}

Write-Host "Activating .venv for this script..."
$venvPython = ".\.venv\Scripts\python.exe"

& $venvPython -m pip install --upgrade pip setuptools wheel

Write-Host "Installing PyTorch CUDA 12.1 build..."
& $venvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "Installing project requirements..."
& $venvPython -m pip install -r requirements.txt

Write-Host "Installing genomics dependencies..."
& $venvPython -m pip install GEOparse gseapy xgboost imbalanced-learn lifelines pycombat

Write-Host "Verifying GPU..."
& $venvPython -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

Write-Host "Setup complete. Use: .\.venv\Scripts\Activate.ps1"

