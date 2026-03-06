import torch
import sys
import os

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.losses import VectorizedMultiScaleSpectralLoss

def run_edge_case_tests():
    print("==================================================")
    print("      V8.8 TUNGSTEN GUARDRAIL EDGE CASE TEST      ")
    print("==================================================")
    
    # Initialize our hardened STFT loss function
    # Using the exact configuration from train_inverter_audio.py
    loss_fn = VectorizedMultiScaleSpectralLoss()
    
    # Create test tensors
    batch_size = 2
    audio_len = 131072 # About 3 seconds at 44.1kHz
    
    # CASE 1: Normal Audio (Baseline)
    normal_pred = torch.randn(batch_size, 1, audio_len, requires_grad=True) * 0.5
    normal_target = torch.randn(batch_size, 1, audio_len) * 0.5
    
    # CASE 2: Pure Silence vs Pure Silence
    silent_pred = torch.zeros(batch_size, 1, audio_len, requires_grad=True)
    silent_target = torch.zeros(batch_size, 1, audio_len)
    
    # CASE 3: Normal Audio vs Pure Silence Target
    # (e.g., model failed to learn volume=0)
    normal_pred_2 = torch.randn(batch_size, 1, audio_len, requires_grad=True) * 0.5
    
    # CASE 4: Pure Silence Pred vs Normal Target
    # (e.g., model guessed EQ cutoff instead of letting wave through)
    silent_pred_2 = torch.zeros(batch_size, 1, audio_len, requires_grad=True)
    
    # CASE 5: Micro-Noise Pred vs Pure Silence Target
    # (Testing the boundary of the 1e-4 epsilon)
    micro_pred = torch.randn(batch_size, 1, audio_len, requires_grad=True) * 1e-9
    
    tests = [
        ("Baseline (Normal vs Normal)", normal_pred, normal_target),
        ("Silence vs Silence", silent_pred, silent_target),
        ("Normal vs Silence", normal_pred_2, silent_target),
        ("Silence vs Normal", silent_pred_2, normal_target),
        ("Micro-Noise vs Silence", micro_pred, silent_target)
    ]
    
    all_passed = True
    
    for name, pred, target in tests:
        print(f"\n--- Testing Edge Case: {name} ---")
        pred.retain_grad()
        
        # 1. Forward Pass (Check for NaN/Inf Loss)
        try:
            loss = loss_fn(pred, target)
            print(f"Forward Pass: Loss = {loss.item():.4f}")
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("❌ FAILED: Loss is NaN or Infinity!")
                all_passed = False
                continue
                
            if loss.item() > 1000.0:  # We clamped at 100 per scale, total shouldn't exceed loosely 600
                print(f"❌ FAILED: Loss value ({loss.item()}) is suspiciously massive (Solar Flare!).")
                all_passed = False
                continue
                
        except Exception as e:
            print(f"❌ FAILED: Forward pass crashed! {e}")
            all_passed = False
            continue
            
        # 2. Backward Pass (Check for NaN/Inf Gradients)
        try:
            loss.backward()
            grad_max = torch.max(torch.abs(pred.grad))
            print(f"Backward Pass: Max Gradient = {grad_max.item():.6f}")
            
            if torch.isnan(grad_max) or torch.isinf(grad_max):
                print("❌ FAILED: Gradients exploded to NaN or Infinity!")
                all_passed = False
                continue
                
            if grad_max.item() > 100.0:
                print(f"❌ FAILED: Gradients are astronomically large ({grad_max.item()}). Optimizers will be poisoned.")
                all_passed = False
                continue
                
            print("✅ PASSED: Numerically Stable.")
            
        except Exception as e:
            print(f"❌ FAILED: Backward pass crashed! {e}")
            all_passed = False
            continue
            
    print("\n==================================================")
    if all_passed:
        print("🎉 ALL V8.8 TUNGSTEN GUARDRAILS HOLD STRONG!")
        print("Silence and Micro-Noise no longer cause Solar Flares.")
    else:
        print("💥 CRITICAL FAILURE IN GUARDRAILS DETECTED!")
    print("==================================================")

if __name__ == "__main__":
    run_edge_case_tests()
