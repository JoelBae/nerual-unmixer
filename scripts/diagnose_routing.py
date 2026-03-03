import sys
import os
import sounddevice as sd
import numpy as np
import time

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.data.ableton_client import AbletonClient

def diagnose():
    client = AbletonClient()
    print("--- 1. OSC CONNECTION TEST ---")
    print("Handshaking with Ableton...")
    if client.ping(timeout=1.0):
        print("✅ OSC Handshake SUCCESSFUL.")
        tracks = client.get_track_count()
        if tracks is not None:
            print(f"✅ Ableton reported {tracks} tracks.")
    else:
        print("⚠️  OSC Handshake FAILED (No response on port 11001).")
        print("   This might mean AbletonOSC is not configured to send to 11001,")
        print("   or a firewall is blocking the return path.")

    print("\n--- 2. SEND-ONLY TEST ---")
    print("Moving Master Volume to 50% (check Ableton Master fader)...")
    client.set_master_volume(0.5)
    time.sleep(1.0)
    print("Moving Master Volume to 85%...")
    client.set_master_volume(0.85)
    print(">>> DID THE MASTER FADER MOVE? (Yes/No)")

    print("--- 2. AUDIO ROUTING TEST (BlackHole 64ch) ---")
    devices = sd.query_devices()
    bh_idx = -1
    for i, d in enumerate(devices):
        if "BlackHole 64" in d["name"]:
            bh_idx = i
            break
    
    if bh_idx == -1:
        print("❌ BlackHole 64ch NOT FOUND. Check System Settings.")
        return

    print(f"🎧 Found {devices[bh_idx]['name']} (index {bh_idx})")
    bh_info = sd.query_devices(bh_idx, 'input')
    default_sr = bh_info['default_samplerate']
    print(f"   Default Sample Rate: {default_sr} Hz")
    if default_sr != 44100:
        print(f"⚠️  WARNING: Sample rate mismatch! Ableton/Script expect 44100, BlackHole is {default_sr}.")
        print("   Fix: Open 'Audio MIDI Setup' app and set BlackHole 64ch to 44.1kHz.")

    print("\nRecording 3 seconds of ALL 64 CHANNELS...")
    print(">>> PLEASE PLAY WHITE NOISE IN ABLETON NOW! <<<")
    
    try:
        rec = sd.rec(int(44100 * 3), samplerate=44100, channels=64, device=bh_idx, blocking=True)
    except Exception as e:
        print(f"❌ ERROR: Could not start recording: {e}")
        print("   This is usually a macOS Microphone Permission issue.")
        print("   Fix: System Settings > Privacy & Security > Microphone > [Your Terminal/App] -> ON.")
        return
    
    active_channels = []
    for ch in range(64):
        rms = np.sqrt(np.mean(rec[:, ch]**2))
        if rms > 0.001:
            active_channels.append(ch + 1) # 1-indexed for display
            print(f"  [Ch {ch+1:02d}] ACTIVE (RMS: {rms:.4f})")
    
    if not active_channels:
        print("❌ NO AUDIO DETECTED on any of the 64 channels.")
        print("   Check: 1. Ableton Output Device 2. Track Routing 3. macOS Microphone Permissions.")
    else:
        print(f"✅ Found {len(active_channels)} active channels: {active_channels}")
        if len(active_channels) < 32:
            print("⚠️  Warning: For 16 lanes, you need 32 tracks (64 channels) active.")

if __name__ == "__main__":
    diagnose()
