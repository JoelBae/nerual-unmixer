
import torch
import numpy as np
import sys
import os
import soundfile as sf
sys.path.append('src')
from models.proxies.ott_ddsp import OTTProxy
from data.pedalboard_engine import PedalboardEngine

def render_comparison():
    print("--- Rendering OTT Comparison Audio ---")
    engine = PedalboardEngine()
    proxy = OTTProxy()
    
    duration_sec = 2.0
    sample_rate = 44100
    n_samples = int(duration_sec * sample_rate)
    
    # 1. Create a rich input signal: Frequency sweep + Drum-like pulses
    t = np.linspace(0, duration_sec, n_samples)
    sweep = np.sin(2 * np.pi * 50 * np.power(200, t/duration_sec)) # 50Hz to 10kHz
    
    # Add pulses
    pulse_freq = 4 # 4 pulses per second
    pulses = np.abs(np.sin(np.pi * pulse_freq * t))**10
    
    input_audio_np = (sweep * pulses * 0.5).astype(np.float32)
    input_audio_stereo = np.stack([input_audio_np, input_audio_np]) # Stereo for engine
    input_audio_torch = torch.from_numpy(input_audio_np).unsqueeze(0) # Mono for proxy
    
    # 2. Settings (Aggressive OTT)
    params_vst = {
        "depth": 1.0,
        "thresh_l": 0.2, # Low threshold = more compression
        "thresh_m": 0.2,
        "thresh_h": 0.2,
        "gain_l_db": 0.5,
        "gain_m_db": 0.5,
        "gain_h_db": 0.5
    }
    
    # 3. Render VST
    print("Rendering VST...")
    vst_audio = engine.render("ott", params_vst, duration_sec=duration_sec, input_audio=input_audio_stereo)
    vst_audio_mono = vst_audio[0]
    
    # 4. Render Proxy
    print("Rendering Proxy...")
    # Map normalized [0, 1] params to proxy
    depth = torch.tensor([[1.0]])
    tl = torch.tensor([[0.2]])
    tm = torch.tensor([[0.2]])
    th = torch.tensor([[0.2]])
    gl = torch.tensor([[0.5]])
    gm = torch.tensor([[0.5]])
    gh = torch.tensor([[0.5]])
    
    proxy_audio = proxy(input_audio_torch, depth, tl, tm, th, gl, gm, gh)
    proxy_audio_np = proxy_audio.detach().cpu().numpy()[0]
    
    # 5. Save outputs
    output_dir = "test_audio_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    sf.write(f"{output_dir}/ott_comparison_input.wav", input_audio_np, sample_rate)
    sf.write(f"{output_dir}/ott_comparison_vst.wav", vst_audio_mono, sample_rate)
    sf.write(f"{output_dir}/ott_comparison_proxy.wav", proxy_audio_np, sample_rate)
    
    print(f"\nSaved audio to {output_dir}/")
    print(f" - ott_comparison_input.wav")
    print(f" - ott_comparison_vst.wav")
    print(f" - ott_comparison_proxy.wav")

if __name__ == "__main__":
    render_comparison()
