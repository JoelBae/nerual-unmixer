import os
import sys
import torch
import torchaudio

# Ensure src module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.models.proxy.ddsp_modules import HarmonicOscillator

if __name__ == "__main__":
    print("Testing the Additive Harmonic Oscillator...")
    
    sample_rate = 44100
    oscillator = HarmonicOscillator(sample_rate=sample_rate, num_harmonics=64)
    
    # 440 Hz (A4)
    f0 = torch.tensor([[440.0]], dtype=torch.float32)
    
    # Let's create a perfect mathematically generated Sawtooth wave!
    # A true sawtooth's harmonics decrease in amplitude by 1/n.
    # So harmonic 1 is 1.0, harmonic 2 is 0.5, harmonic 3 is 0.33...
    harmonic_amplitudes = torch.zeros((1, 64), dtype=torch.float32)
    for n in range(1, 65):
        # We alternate the sign for a perfect sawtooth to avoid a phase jump
        # Actually, standard saw is just (-1)^(n+1) * 1/n for sine series?
        # A simple trailing saw is just +1/n if we don't care about absolute phase.
        harmonic_amplitudes[0, n-1] = 1.0 / n
        
    # Scale down overall volume so we don't clip the audio interface
    harmonic_amplitudes = harmonic_amplitudes * 0.5
        
    print("Generating a 64-harmonic Sawtooth Wave...")
    audio_tensor = oscillator(f0, harmonic_amplitudes)
    
    output_path = "test_saw.wav"
    torchaudio.save(output_path, audio_tensor, sample_rate)
    
    print(f"Success! Audio saved to: {os.path.abspath(output_path)}")
    print(f"Shape: {audio_tensor.shape}")
