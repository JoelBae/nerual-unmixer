import torch
import torch.nn as nn
import time
import os
import sys
from tqdm import tqdm
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.proxy.ott import OTTProxy
from src.models.losses import DynamicsLoss
from src.data.dataset import NeuralProxyDataset

def benchmark_ott(device="cpu", num_batches=10):
    print(f"\n--- OTT Benchmark on {device} ---")
    
    # 1. Setup Data
    dataset = NeuralProxyDataset(effect_name="ott", dataset_dir="dataset/ott", split="val", preload=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    
    # 2. Setup Model and Loss
    model = OTTProxy().to(device)
    criterion = DynamicsLoss().to(device)
    
    # 3. Benchmark Speed
    latencies = []
    losses = []
    
    # Warmup
    print("   > Warming up...")
    for i, (input_audio, params, target_audio) in enumerate(loader):
        if i >= 2: break
        _ = model(input_audio.to(device), params.to(device))
        
    print(f"   > Running {num_batches} batches...")
    start_total = time.time()
    for i, (input_audio, params, target_audio) in enumerate(tqdm(loader, total=num_batches)):
        if i >= num_batches: break
        
        input_audio = input_audio.to(device)
        params = params.to(device)
        target_audio = target_audio.to(device)
        
        # Time the forward pass
        start_batch = time.time()
        with torch.no_grad():
            output_audio = model(input_audio, params)
        torch.cuda.synchronize() if device == "cuda" else None
        # Note: MPS synchronization is implicit or handled by torch
        
        latencies.append(time.time() - start_batch)
        
        # Calculate Accuracy
        loss = criterion(output_audio, target_audio)
        losses.append(loss.item())

    end_total = time.time()
    
    avg_latency = np.mean(latencies)
    avg_loss = np.mean(losses)
    throughput = 1.0 / avg_latency if avg_latency > 0 else 0
    
    print(f"   > Average Latency: {avg_latency*1000:.2f} ms")
    print(f"   > Throughput: {throughput:.2f} batches/sec")
    print(f"   > Average Dynamics Loss: {avg_loss:.4f}")
    
    return {
        "device": device,
        "latency_ms": avg_latency * 1000,
        "throughput": throughput,
        "loss": avg_loss
    }

if __name__ == "__main__":
    results = []
    results.append(benchmark_ott("cpu", num_batches=10))
    if torch.backends.mps.is_available():
        results.append(benchmark_ott("mps", num_batches=10))
    
    print("\n\n--- Summary Table ---")
    print(f"{'Device':<10} | {'Latency (ms)':<15} | {'Throughput (b/s)':<18} | {'Loss':<10}")
    print("-" * 65)
    for r in results:
        print(f"{r['device']:<10} | {r['latency_ms']:<15.2f} | {r['throughput']:<18.2f} | {r['loss']:<10.4f}")
