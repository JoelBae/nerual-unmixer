import torch
import numpy as np
import soundfile as sf
import os
from src.models.proxies.kilohearts_ddsp import KHDistortionDSP

def generate_demo():
    print("Generating KH Distortion Proxy Demo...")
    sample_rate = 44100
    duration = 1.0
    freq = 440.0 # A4
    
    # 1. Create Sine Wave
    t = torch.linspace(0, duration, int(sample_rate * duration))
    x = torch.sin(2 * np.pi * freq * t).unsqueeze(0) # (batch, samples)
    
    # 2. Initialize DSP
    dsp = KHDistortionDSP(sample_rate)
    
    # 3. Parameters for test
    # High Drive, Low Bias, Full Mix, Soft Clip (type 0)
    drive = torch.tensor([[0.8]]) 
    bias = torch.tensor([[0.5]]) # 0.5 is no offset
    mix = torch.tensor([[1.0]])
    type_idx = 0
    
    # 4. Process
    with torch.no_grad():
        y = dsp(x, drive, bias, mix, type_idx)
    
    # 5. Save Results
    output_dir = "test_audio_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    dry_path = os.path.join(output_dir, "kh_demo_dry.wav")
    wet_path = os.path.join(output_dir, "kh_demo_wet_softclip.wav")
    
    sf.write(dry_path, x.squeeze().numpy(), sample_rate)
    sf.write(wet_path, y.squeeze().numpy(), sample_rate)
    
    print(f"  [DONE] Saved dry: {dry_path}")
    print(f"  [DONE] Saved wet: {wet_path}")
    
    # 6. Simple Harmonic Analysis
    peak_dry = torch.max(torch.abs(x)).item()
    peak_wet = torch.max(torch.abs(y)).item()
    print(f"  Analyses:")
    print(f"    Peak Dry: {peak_dry:.4f}")
    print(f"    Peak Wet: {peak_wet:.4f} (Tanh clipped at ~1.0)")
    
    # Check if harmonics were added (RMS of high frequencies)
    # Very simple spectral check: high pass at 1kHz
    def simple_rms(signal):
        return torch.sqrt(torch.mean(signal**2)).item()
    
    print(f"    Dry RMS: {simple_rms(x):.4f}")
    print(f"    Wet RMS: {simple_rms(y):.4f}")

if __name__ == "__main__":
    generate_demo()
