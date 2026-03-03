import sys, os, time, torch, sounddevice as sd, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.ableton_client import AbletonClient
from src.data.generator_parallel import detect_blackhole

client = AbletonClient()
bh_idx, _, _ = detect_blackhole(min_channels=4)
sr = 44100
duration = 1.0

# Ensure both tracks are active but we'll use Mute for solo simulation if needed 
# Actually Track Solo is better if available
# Address: /live/track/set/solo [int track, bool solo]

def record_track(track_idx, name):
    print(f"--- Testing {name} (Track {track_idx}) ---")
    # Solo the track
    client.client.send_message("/live/track/set/solo", [track_idx, 1])
    time.sleep(0.5)
    
    client.play()
    rec = sd.rec(int(duration * sr), samplerate=sr, channels=4, blocking=True, device=bh_idx)
    client.stop()
    
    rms = [np.sqrt(np.mean(rec[:, i]**2)) for i in range(4)]
    print(f"{name} RMS: Ch1={rms[0]:.4f}, Ch2={rms[1]:.4f}, Ch3={rms[2]:.4f}, Ch4={rms[3]:.4f}")
    
    # Unsolo
    client.client.send_message("/live/track/set/solo", [track_idx, 0])
    time.sleep(0.1)

record_track(0, "Track 0 (Dry)")
record_track(1, "Track 1 (Wet)")
