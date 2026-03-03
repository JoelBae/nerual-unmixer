import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.proxy.chain import ProxyChainer
from src.models.losses import SpectralLoss
from src.utils.normalization import denormalize_params

def run_inversion_test(steps=200, lr=0.1):
    """
    Sanity Check: Can we recover 63 parameters from audio via pure backprop?
    This proves differentiability and correct parameter mapping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- 🧪 Differentiable Inversion Sanity Check ---")
    print(f"Device: {device} | Iterations: {steps}")

    # 1. Setup Models
    chainer = ProxyChainer().to(device)
    chainer.eval() # Proxies should be in eval mode for inversion
    
    # 2. Create "Ground Truth" Parameters (Random but fixed)
    # 63 parameters in total (Operator 16, Saturator 5, EQ8 32, OTT 7, Reverb 3)
    torch.manual_seed(42)
    norm_truth = torch.rand(1, 63, device=device)
    
    # Generate Ground Truth Audio
    with torch.no_grad():
        raw_truth = denormalize_params(norm_truth)
        audio_target = chainer.forward_flat(raw_truth)
        print("✅ Generated Ground Truth Audio Reference.")

    # 3. Setup Learnable Parameters (Starting from random points)
    # We initialize with a guess (e.g., 0.5 for all normalized knobs)
    norm_pred = nn.Parameter(torch.full((1, 63), 0.5, device=device))
    
    # 4. Optimization Loop
    optimizer = optim.Adam([norm_pred], lr=lr)
    criterion = SpectralLoss().to(device)
    
    print("\nStarting Optimization loop...")
    start_time = time.time()
    
    for i in range(steps):
        optimizer.zero_grad()
        
        # Denormalize current guess to raw Ableton values
        raw_pred = denormalize_params(norm_pred)
        
        # Pass through the differentiable chain
        audio_pred = chainer.forward_flat(raw_pred)
        
        # Compute Loss
        loss = criterion(audio_pred, audio_target)
        
        if torch.isnan(loss):
            print("❌ Optimization failed (NaN loss). Check DSP ranges.")
            break
            
        # Backprop (This is the critical test: can gradients flow back to norm_pred?)
        loss.backward()
        
        # Clamp gradients for stability
        torch.nn.utils.clip_grad_norm_([norm_pred], 1.0)
        
        optimizer.step()
        
        # Clamp params to [0, 1] range (normalized)
        with torch.no_grad():
            norm_pred.clamp_(0.0, 1.0)
            
        if i % 20 == 0:
            param_error = torch.mean(torch.abs(norm_pred - norm_truth)).item()
            print(f"Step {i:3d} | Spectral Loss: {loss.item():.4f} | Param Error (avg): {param_error:.4f}")

    end_time = time.time()
    final_error = torch.mean(torch.abs(norm_pred - norm_truth)).item()
    
    print(f"\n--- Results ---")
    print(f"Final Param Error: {final_error:.4f}")
    if final_error < 0.05:
        print("✅ SUCCESS: The system is fully differentiable!")
    else:
        print("⚠️  PARTIAL: Audio matched but parameters differ (Potential non-unique solution or learning rate issue).")
    print(f"Time Taken: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    run_inversion_test()
