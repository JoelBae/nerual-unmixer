import sys
import argparse
import time
import os
import json
import numpy as np
import soundfile as sf

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.ableton_client import AbletonClient
from src.data.generator import record_audio

def generate_wave_sweep(output_dir: str, duration: float = 1.0, sample_rate: int = 44100, device_name: str = None):
    # We assume the user has set the Operator and chain to their liking.
    # We only swing the 'Osc-A Wave' parameter (Macro 2).
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    import sounddevice as sd
    client = AbletonClient()
    
    # Identify recording device
    devices = sd.query_devices()
    target_device = None
    
    if device_name:
        for i, dev in enumerate(devices):
            if device_name.lower() in dev["name"].lower() and dev["max_input_channels"] >= 2:
                target_device = i
                print(f"✅ Forced device: [{i}] {dev['name']}")
                break
    else:
        for i, dev in enumerate(devices):
            if "BlackHole" in dev["name"] and dev["max_input_channels"] >= 2:
                target_device = i
                print(f"🎧 Auto-detected: [{i}] {dev['name']}")
                break
    
    if target_device is None:
        print("⚠️  No BlackHole device found. Using default system input.")

    metadata = []
    
    print(f"Starting wave sweep (0-127)... saving to {output_dir}")
    
    # Sweep Osc-A Wave (Macro 2)
    for wave_val in range(0, 128):
        print(f"--- Wave Value: {wave_val}/127 ---")
        
        # 1. Set Parameter
        client.set_track_parameter(0, 0, 2, float(wave_val))
        time.sleep(0.1)

        # 2. Record
        client.play()
        # Custom record block to use our selected device
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, blocking=True, device=target_device)
        client.stop()
        
        # 3. Save
        filename = f"wave_{wave_val:03d}.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, audio_data, sample_rate)
        
        metadata.append({
            "wave_value": wave_val,
            "filename": filename
        })
        
    # Save Metadata
    with open(os.path.join(output_dir, "wave_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("Wave sweep complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Wave Sweep Dataset for Operator")
    parser.add_argument("--output_dir", type=str, default="./dataset/operator_wave_sweep", help="Output directory")
    parser.add_argument("--duration", type=float, default=1.0, help="Duration per sample")
    parser.add_argument("--device", type=str, default=None, help="Force a specific device (e.g. '2ch' or '64ch')")

    args = parser.parse_args()
    generate_wave_sweep(args.output_dir, args.duration, device_name=args.device)
