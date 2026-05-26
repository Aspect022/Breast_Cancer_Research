param(
    [string]$Config = "config_genomics_rtx5050.yaml",
    [string]$Phase = "phase1",
    [switch]$Synthetic
)

$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run scripts\genomics\rtx5050_setup.ps1 first."
}

$env:CUDA_VISIBLE_DEVICES = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }
$env:OMP_NUM_THREADS = if ($env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS } else { "6" }
$env:MKL_NUM_THREADS = if ($env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS } else { "6" }

Write-Host "=== RTX 5050 Genomics Run ==="
Write-Host "Config: $Config"
Write-Host "Phase:  $Phase"

& $python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

$syntheticArg = @()
if ($Synthetic) {
    $syntheticArg = @("--synthetic")
}

switch ($Phase) {
    "smoke" {
        & $python run_genomic.py --config $Config --models g_baseline_mlp @syntheticArg
    }
    "phase1" {
        & $python run_genomic.py --config $Config --models g_baseline_mlp g_baseline_trees @syntheticArg
    }
    "mlp" {
        & $python run_genomic.py --config $Config --models g_baseline_mlp @syntheticArg
    }
    "trees" {
        & $python run_genomic.py --config $Config --models g_baseline_trees @syntheticArg
    }
    default {
        throw "Unknown phase '$Phase'. Use: smoke, phase1, mlp, trees."
    }
}

Write-Host "RTX 5050 genomics run finished."

