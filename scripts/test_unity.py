import sys
import os
import time
import json
import torch
import torchaudio
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.ableton_client import AbletonClient

def test():
    client = AbletonClient()
    # 1. Turn OFF EQ8 on the wet track (Lane 0, Track 1, Device 1)
    # /live/device/set/is_active [track, device, val]
    print("Disabling EQ8 on Track 1...")
    client.set_device_enabled(1, 1, False)
    time.sleep(1.0)
    
    # 2. Record 2 seconds
    import sounddevice as sd
    from src.data.generator_parallel import detect_blackhole
    bh_idx, bh_channels, _ = detect_blackhole(min_channels=4)
    
    print("Recording Unity Test...")
    client.play()
    sr = 44100
    duration = 2.0
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=4, blocking=True, device=bh_idx)
    client.stop()
    
    # 3. Split
    dry = recording[:, 0:2]
    wet = recording[:, 2:4]
    
    # 4. Compare
    d_rms = np.sqrt(np.mean(dry**2))
    w_rms = np.sqrt(np.mean(wet**2))
    print(f"Dry RMS: {d_rms:.4f}, Wet RMS: {w_rms:.4f}")
    
    # Check spectrum
    from src.models.losses import VectorizedMultiScaleSpectralLoss
    loss_fn = VectorizedMultiScaleSpectralLoss()
    d_torch = torch.tensor(dry).T.unsqueeze(0)
    w_torch = torch.tensor(wet).T.unsqueeze(0)
    
    # Naive alignment (latency check)
    corr = np.correlate(dry[:,0], wet[:,0], mode='full')
    offset = np.argmax(corr) - (len(dry) - 1)
    print(f"Latency: {offset} samples")
    
    # Slice
    if offset > 0:
        w_s = w_torch[:, :, offset:]
        d_s = d_torch[:, :, :-offset]
    elif offset < 0:
        abs_o = abs(offset)
        w_s = w_torch[:, :, abs_o:]
        d_s = d_torch[:, :, :-abs_o]
    else:
        w_s = w_torch
        d_s = d_torch
        
    min_len = min(w_s.shape[-1], d_s.shape[-1])
    loss = loss_fn(w_s[:, :, :min_len], d_s[:, :, :min_len])
    print(f"Unity Spectral Loss: {loss.item():.4f}")
    
    # Re-enable for future use
    client.set_device_enabled(1, 1, True)

if __name__ == "__main__":
    test()
