import torch
import time

def benchmark_cummax(device="mps", size=88200, batch_size=16):
    print(f"\n--- Benchmarking Cummax on {device} ---")
    print(f"Size: {size} | Batch: {batch_size}")
    
    x = torch.randn(batch_size, size).to(device)
    
    # Warmup
    for _ in range(5):
        y, _ = torch.cummax(x, dim=1)
        if device == "mps": torch.mps.synchronize()
        
    # Timing
    start = time.time()
    num_runs = 50
    for _ in range(num_runs):
        y, _ = torch.cummax(x, dim=1)
        if device == "mps": torch.mps.synchronize()
    
    avg_ms = (time.time() - start) / num_runs * 1000
    print(f"   > Avg Time (Full): {avg_ms:.2f} ms")

    # Downsampled case
    size_ds = size // 64
    x_ds = torch.randn(batch_size, size_ds).to(device)
    start = time.time()
    for _ in range(num_runs):
        y, _ = torch.cummax(x_ds, dim=1)
        if device == "mps": torch.mps.synchronize()
    avg_ms_ds = (time.time() - start) / num_runs * 1000
    print(f"   > Avg Time (Downsampled): {avg_ms_ds:.2f} ms")

if __name__ == "__main__":
    benchmark_cummax("mps", size=88200)
    benchmark_cummax("cpu", size=88200)
