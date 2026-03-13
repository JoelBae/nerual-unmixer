import torch
import numpy as np
import soundfile as sf
import os
from src.models.proxies.vital_ddsp import VitalProxy

def main():
    print("Initializing VitalProxy...")
    sr = 44100
    duration = 2.0
    n_samples = int(sr * duration)
    proxy = VitalProxy(n_harmonics=64, sample_rate=sr)
    
    # Define parameters for a "plucky" sound with a filter sweep
    # (batch, 1) or (batch, samples, 1)
    
    # 1. Pitch: Steady C4 (261.63 Hz)
    f0 = torch.ones((1, n_samples, 1)) * 261.63
    
    # 2. ADSR: Pluck
    attack = torch.tensor([[0.005]])   # fast attack
    decay = torch.tensor([[0.5]])      # medium decay
    sustain = torch.tensor([[0.1]])    # low sustain
    release = torch.tensor([[0.2]])    # short release
    
    # 3. Filter: Sweep from high to low
    # We'll create a trajectory for cutoff
    cutoff_traj = torch.linspace(0.8, 0.2, n_samples).view(1, n_samples)
    res = torch.tensor([[0.4]])
    
    # 4. Gate: Note on for 1.5s then release
    gate = torch.zeros((1, n_samples))
    gate[0, :int(sr * 1.5)] = 1.0
    
    # No wavetable mapping yet, so we use internal ones in proxy
    wavetable_pos = torch.tensor([[0.5]])

    print("Rendering audio through proxy...")
    with torch.no_grad():
        # Note: VitalProxy.forward expects certain shapes and does some internal broadcasting
        # We need to make sure we pass what it expects.
        # Based on my implementation:
        # cutoff can be (1, 1) or (1, samples)
        # res can be (1, 1) or (1, samples)
        
        audio = proxy(
            f0=f0,
            wavetable_pos=wavetable_pos,
            cutoff=cutoff_traj,
            res=res,
            attack=attack,
            decay=decay,
            sustain=sustain,
            release=release,
            gate=gate
        )
    
    # Convert to numpy and normalize
    audio_np = audio.squeeze().numpy()
    audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-7)
    
    output_dir = "test_audio_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vital_proxy_test.wav")
    
    print(f"Saving to {output_path}...")
    sf.write(output_path, audio_np, sr)
    print("Done! You can now listen to the proxy's output.")

if __name__ == "__main__":
    main()
