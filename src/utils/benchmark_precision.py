import torch
import torch.nn.functional as F
import time

def benchmark_precision(device="mps", batch_size=64, time_dim=88200):
    print(f"\n--- Benchmarking Precision on {device} ---")
    
    x32 = torch.randn(batch_size, 2, time_dim).to(device)
    x16 = x32.half()
    
    # 1. FFT 32
    start = time.time()
    for _ in range(20):
        X = torch.fft.rfft(x32, n=131072)
        y = torch.fft.irfft(X, n=131072)
        if device == "mps": torch.mps.synchronize()
    print(f"   > FFT FP32: {(time.time()-start)/20*1000:.2f} ms")

    # 2. FFT 16 (Note: MPS might not support half FFT, check)
    try:
        start = time.time()
        for _ in range(20):
            # MPS RFFT doesn't support half well, might need to cast
            X = torch.fft.rfft(x16.float(), n=131072).half()
            y = torch.fft.irfft(X.float(), n=131072).half()
            if device == "mps": torch.mps.synchronize()
        print(f"   > FFT FP16 (with cast): {(time.time()-start)/20*1000:.2f} ms")
    except Exception as e:
        print(f"   > FFT FP16 failed: {e}")

    # 3. Conv1d 32
    conv32 = torch.nn.Conv1d(2, 8, 7, padding=3).to(device)
    start = time.time()
    for _ in range(20):
        y = conv32(x32)
        if device == "mps": torch.mps.synchronize()
    print(f"   > Conv1d FP32: {(time.time()-start)/20*1000:.2f} ms")

    # 4. Conv1d 16
    conv16 = torch.nn.Conv1d(2, 8, 7, padding=3).to(device).half()
    start = time.time()
    for _ in range(20):
        y = conv16(x16)
        if device == "mps": torch.mps.synchronize()
    print(f"   > Conv1d FP16: {(time.time()-start)/20*1000:.2f} ms")

if __name__ == "__main__":
    benchmark_precision("mps")
