import torch
import torch.nn.functional as F
import time

def benchmark_ops(device="mps", batch_size=16, time_dim=88200):
    print(f"\n--- Benchmarking OTT Ops on {device} ---")
    
    # 1. FFT (rfft + irfft)
    x = torch.randn(batch_size, 2, time_dim).to(device)
    n_fft = 131072
    h = torch.randn(1, 1, n_fft//2 + 1, dtype=torch.complex64).to(device)
    
    # Warmup
    for _ in range(5):
        X = torch.fft.rfft(x, n=n_fft)
        Y = X * h
        y = torch.fft.irfft(Y, n=n_fft)
        if device == "mps": torch.mps.synchronize()
        
    start = time.time()
    for _ in range(50):
        X = torch.fft.rfft(x, n=n_fft)
        Y = X * h
        y = torch.fft.irfft(Y, n=n_fft)
        if device == "mps": torch.mps.synchronize()
    print(f"   > FFT (131k): {(time.time()-start)/50*1000:.2f} ms")

    # 2. Interpolate (1378 -> 88200)
    x_low = torch.randn(batch_size * 3, 1, 1378).to(device)
    for _ in range(5):
        y = F.interpolate(x_low, size=time_dim, mode='linear', align_corners=True)
        if device == "mps": torch.mps.synchronize()
        
    start = time.time()
    for _ in range(50):
        y = F.interpolate(x_low, size=time_dim, mode='linear', align_corners=True)
        if device == "mps": torch.mps.synchronize()
    print(f"   > Interpolate: {(time.time()-start)/50*1000:.2f} ms")

    # 3. Residual Net (CNN7x3)
    conv1 = torch.nn.Conv1d(2, 8, 7, padding=3).to(device)
    conv2 = torch.nn.Conv1d(8, 2, 7, padding=3).to(device)
    for _ in range(5):
        y = conv2(F.relu(conv1(x)))
        if device == "mps": torch.mps.synchronize()
        
    start = time.time()
    for _ in range(50):
        y = conv2(F.relu(conv1(x)))
        if device == "mps": torch.mps.synchronize()
    print(f"   > Residual Net: {(time.time()-start)/50*1000:.2f} ms")

if __name__ == "__main__":
    benchmark_ops("mps")
    benchmark_ops("cpu")
