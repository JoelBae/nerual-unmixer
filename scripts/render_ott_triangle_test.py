
import torch
import numpy as np
import sys
import os
import soundfile as sf
from scipy import signal
sys.path.append('src')
from models.proxies.ott_ddsp import OTTProxy
from data.pedalboard_engine import PedalboardEngine

def render_triangle_test():
    print("--- Rendering OTT Triangle Wave Test ---")
    engine = PedalboardEngine()
    proxy = OTTProxy()
    
    duration_sec = 2.0
    sample_rate = 44100
    n_samples = int(duration_sec * sample_rate)
    
    # 1. Create a Triangle Wave (440Hz with some pitch bend)
    t = np.linspace(0, duration_sec, n_samples)
    freq = 440 * np.power(2, -t/duration_sec) # 1 octave drop
    triangle = signal.sawtooth(2 * np.pi * np.cumsum(freq)/sample_rate, 0.5)
    
    # Add a simple envelope to make it more "plucky"
    env = np.exp(-3 * t)
    input_audio_np = (triangle * env * 0.8).astype(np.float32)
    
    input_audio_stereo = np.stack([input_audio_np, input_audio_np])
    input_audio_torch = torch.from_numpy(input_audio_np).unsqueeze(0)
    
    # 2. Settings (Standard OTT)
    params_vst = {
        "depth": 1.0,
        "thresh_l": 0.5,
        "thresh_m": 0.5,
        "thresh_h": 0.5,
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
    depth = torch.tensor([[1.0]])
    tl = torch.tensor([[0.5]])
    tm = torch.tensor([[0.5]])
    th = torch.tensor([[0.5]])
    gl = torch.tensor([[0.5]])
    gm = torch.tensor([[0.5]])
    gh = torch.tensor([[0.5]])
    
    proxy_audio = proxy(input_audio_torch, depth, tl, tm, th, gl, gm, gh)
    proxy_audio_np = proxy_audio.detach().cpu().numpy()[0]
    
    # 5. Save outputs
    output_dir = "test_audio_outputs/ott"
    os.makedirs(output_dir, exist_ok=True)
    
    sf.write(f"{output_dir}/triangle_input.wav", input_audio_np, sample_rate)
    sf.write(f"{output_dir}/triangle_vst.wav", vst_audio_mono, sample_rate)
    sf.write(f"{output_dir}/triangle_proxy.wav", proxy_audio_np, sample_rate)
    
    print(f"\nSaved audio to {output_dir}/")
    print(f" - triangle_input.wav")
    print(f" - triangle_vst.wav")
    print(f" - triangle_proxy.wav")

if __name__ == "__main__":
    render_triangle_test()
