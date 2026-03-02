import torch
import torch.nn.functional as F
import time

def benchmark_conv1d(device="mps", kernel_size=512, signal_len=88200, batch_size=16):
    print(f"\n--- Benchmarking Conv1d on {device} ---")
    print(f"Kernel: {kernel_size} | Signal: {signal_len} | Batch: {batch_size}")
    
    x = torch.randn(batch_size, 2, signal_len).to(device)
    w = torch.randn(2, 1, kernel_size).to(device)
    
    # Warmup
    for _ in range(5):
        y = F.conv1d(x, w, groups=2)
        if device == "mps": torch.mps.synchronize()
        
    # Timing
    start = time.time()
    num_runs = 50
    for _ in range(num_runs):
        y = F.conv1d(x, w, groups=2)
        if device == "mps": torch.mps.synchronize()
    
    avg_ms = (time.time() - start) / num_runs * 1000
    print(f"   > Avg Time: {avg_ms:.2f} ms")

if __name__ == "__main__":
    benchmark_conv1d("mps", kernel_size=512)
    benchmark_conv1d("mps", kernel_size=128)
    benchmark_conv1d("cpu", kernel_size=512)
