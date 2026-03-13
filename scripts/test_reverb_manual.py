import torch
import numpy as np
import soundfile as sf
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.proxies.kilohearts_ddsp import KHReverbDSP
from src.data.pedalboard_engine import PedalboardEngine

def generate_test_audio(duration_sec=1.5, sample_rate=44100):
    """Generates a short noise burst (snare-like)."""
    n_samples = int(duration_sec * sample_rate)
    # Short noise burst at the beginning
    noise = np.random.randn(n_samples).astype(np.float32) * 0.2
    envelope = np.exp(-np.linspace(0, 10, n_samples)) 
    # Just the first 0.1s is burst
    burst_end = int(0.1 * sample_rate)
    noise[burst_end:] = 0
    return noise * envelope

def test_reverb():
    sr = 44100
    duration = 2.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Generate Input
    audio_np = generate_test_audio(duration, sr)
    audio_torch = torch.from_numpy(audio_np).unsqueeze(0).to(device) # (1, samples)
    
    # 2. Setup Proxy
    reverb_dsp = KHReverbDSP(sample_rate=sr).to(device)
    
    # Test Parameters (Something with long tail)
    params = {
        "decay": torch.tensor([[0.7]], device=device),
        "dampen": torch.tensor([[0.3]], device=device),
        "size": torch.tensor([[0.5]], device=device),
        "width": torch.tensor([[0.8]], device=device),
        "early": torch.tensor([[0.5]], device=device),
        "mix": torch.tensor([[0.5]], device=device)
    }
    
    print("Rendering through Proxy...")
    with torch.no_grad():
        out_proxy = reverb_dsp(audio_torch, **params)
    
    out_proxy_np = out_proxy.squeeze(0).cpu().numpy() # (2, samples)
    
    # 3. Setup VST (Optional)
    engine = PedalboardEngine(sample_rate=sr)
    out_vst_np = None
    
    # Match VST parameters
    # KH Reverb: Decay, Dampen, Size, Width, Early, Mix
    vst_params = {
        "Decay": 0.7,
        "Dampen": 0.3,
        "Size": 0.5,
        "Width": 0.8,
        "Early": 0.5,
        "Mix": 0.5
    }
    
    try:
        # We need to add KH Reverb to PedalboardEngine.VST_PATHS if it's not there
        # But for now let's hope it's standard.
        # Actually KH Reverb is usually "kHs Reverb.vst3"
        print("Rendering through VST...")
        # Add path if missing or use generic name
        engine.VST_PATHS["kh_reverb"] = "/Library/Audio/Plug-Ins/VST3/Kilohearts/kHs Reverb.vst3"
        
        # Prepare stereo input for VST (KH Reverb usually expects stereo)
        input_stereo = np.stack([audio_np, audio_np], axis=0)
        out_vst_np = engine.render("kh_reverb", vst_params, duration_sec=duration, input_audio=input_stereo)
    except Exception as e:
        print(f"VST Rendering failed (skipping): {e}")

    # 4. Save results
    output_dir = "test_audio_outputs/reverb"
    os.makedirs(output_dir, exist_ok=True)
    sf.write(os.path.join(output_dir, "reverb_test_proxy.wav"), out_proxy_np.T, sr)
    print(f"Saved {output_dir}/reverb_test_proxy.wav")
    
    if out_vst_np is not None:
        sf.write(os.path.join(output_dir, "reverb_test_vst.wav"), out_vst_np.T, sr)
        print(f"Saved {output_dir}/reverb_test_vst.wav")

if __name__ == "__main__":
    test_reverb()
