import sys, os, time, torch, sounddevice as sd, numpy as np
from pythonosc import udp_client

def test():
    print("--- Start Routing Check V2 ---", flush=True)
    client = udp_client.SimpleUDPClient("127.0.0.1", 11000)
    
    # 1. Solo Track 0
    print("Soloing Track 0...", flush=True)
    client.send_message("/live/track/set/solo", [0, 1])
    time.sleep(0.5)
    
    print("Starting Playback...", flush=True)
    client.send_message("/live/song/start_playing", [])
    
    sr = 44100
    duration = 1.0
    # Try recording without detect_blackhole, just use a default or specify
    print("Recording 4 channels (1s) on device 4...", flush=True)
    rec = sd.rec(int(duration * sr), samplerate=sr, channels=4, blocking=True, device=4)
    
    print("Stopping Playback...", flush=True)
    client.send_message("/live/song/stop_playing", [])
    
    rms = [np.sqrt(np.mean(rec[:, i]**2)) for i in range(4)]
    print(f"Track 0 Solo RMS: Ch1={rms[0]:.4f}, Ch2={rms[1]:.4f}, Ch3={rms[2]:.4f}, Ch4={rms[3]:.4f}", flush=True)

    # 2. Solo Track 1
    print("Soloing Track 1...", flush=True)
    client.send_message("/live/track/set/solo", [0, 0])
    client.send_message("/live/track/set/solo", [1, 1])
    time.sleep(0.5)
    
    client.send_message("/live/song/start_playing", [])
    rec2 = sd.rec(int(duration * sr), samplerate=sr, channels=4, blocking=True, device=4)
    client.send_message("/live/song/stop_playing", [])
    
    rms2 = [np.sqrt(np.mean(rec2[:, i]**2)) for i in range(4)]
    print(f"Track 1 Solo RMS: Ch1={rms2[0]:.4f}, Ch2={rms2[1]:.4f}, Ch3={rms2[2]:.4f}, Ch4={rms2[3]:.4f}", flush=True)
    
    # Unsolo all
    client.send_message("/live/track/set/solo", [1, 0])

if __name__ == "__main__":
    test()
