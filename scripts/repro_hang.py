import torch
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from src.models.quantum.vectorized_circuit import get_qenn

def benchmark_qenn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}")
    
    model = get_qenn(
        num_classes=2,
        n_qubits=8,
        n_layers=2,
        rotation_config='ry_only'
    ).to(device)
    
    # Input: batch of images (B, 3, 224, 224)
    batch_size = 4 # Small batch to start
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print(f"Starting forward pass with batch size {batch_size}...")
    start_time = time.time()
    
    with torch.no_grad():
        output = model(x)
        
    end_time = time.time()
    print(f"Forward pass completed in {end_time - start_time:.4f} seconds")
    print(f"Time per sample: {(end_time - start_time) / batch_size:.4f} seconds")

if __name__ == "__main__":
    benchmark_qenn()
