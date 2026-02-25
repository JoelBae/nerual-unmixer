import sys
import os
import time

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.data.ableton_client import AbletonClient
from src.data.generator import ParameterRandomizer

def reset_audio():
    print("Connecting to Reset Audio...")
    client = AbletonClient()
    
    # 1. Reset Track 0 Volume (Loud)
    print("Resetting Track 0 Volume to 0.85 (Loud)...")
    client.client.send_message("/live/track/set/volume", [0, 0.85])
    time.sleep(0.5)
    
    # 2. Force Enable Device 0 (Instrument)
    print("Ensuring Device 0 (Instrument Rack) is ON...")
    client.set_track_parameter(0, 0, 0, 1.0) # Param 0 -> 1.0 (ON)
    time.sleep(0.5)

    # 3. Disable Downstream Effects (1-5)
    print("Disabling Devices 1-5 (Saturator, EQ, etc.)...")
    for i in range(1, 6):
        client.set_track_parameter(0, i, 0, 0.0) # Param 0 -> Off
    time.sleep(0.5)
    
    # 4. Apply INIT PATCH to Device 0 (Instrument)
    # Fixed, deterministic values for a clean starting point.
    print("Applying INIT PATCH to Device 0...")
    
    init_patch = [
        {"index": 1, "name": "Transpose", "value": 64.0}, # 0st (Midpoint, not 0.5)
        {"index": 2, "name": "Osc-A Wave", "value": 0.0}, 
        {"index": 3, "name": "Filter Freq", "value": 127.0}, # Open (127, not 1.0)
        {"index": 4, "name": "Filter Res", "value": 0.0}, 
        {"index": 5, "name": "Fe Amount", "value": 64.0}, # 0% Modulation (Midpoint)
        {"index": 6, "name": "Fe Attack", "value": 0.0}, 
        {"index": 7, "name": "Fe Decay", "value": 64.0}, 
        {"index": 8, "name": "Fe Sustain", "value": 127.0}, 
        {"index": 9, "name": "Fe Release", "value": 20.0}, 
        {"index": 10, "name": "Pe Amount", "value": 64.0}, # 0% Modulation
        {"index": 11, "name": "Pe Decay", "value": 10.0}, 
        {"index": 12, "name": "Pe Peak", "value": 64.0}, 
        {"index": 13, "name": "Ae Attack", "value": 0.0}, 
        {"index": 14, "name": "Ae Decay", "value": 64.0}, 
        {"index": 15, "name": "Ae Sustain", "value": 127.0}, # Full Sustain
        {"index": 16, "name": "Ae Release", "value": 20.0}, 
    ]

    for p in init_patch:
        client.set_track_parameter(0, 0, p["index"], p["value"])
        # print(f"   Set {p['name']} -> {p['value']}")
        # time.sleep(0.01) # Fast burst
                
    # 5. Try to START from Start
    print("Starting Transport (Play)...")
    client.stop()
    time.sleep(0.5)
    client.stop() 
    time.sleep(0.5)
    
    client.play()
    print("Playing for 3 seconds...")
    time.sleep(3.0)
    client.stop()
    print("Done. Synths Reset.")

if __name__ == "__main__":
    reset_audio()
