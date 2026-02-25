import sys
import os
import time

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.ableton_client import AbletonClient

def debug_eq_raw():
    print("Connecting to Ableton...")
    client = AbletonClient()
    
    # 2. EQ EIGHT (Device Index 2)
    # Band 1 Gain: Index 7
    
    print("\n--- Testing EQ8 (Device 2) RAW Ranges ---")
    p_idx = 7 # Band 1 Gain
    
    print(f"Sweeping Band 1 Gain (Param {p_idx})...")
    
    # Try sending raw dB values
    print(f"Setting Gain to -15.0...")
    client.set_track_parameter(0, 2, p_idx, -15.0)
    time.sleep(2.0)
    
    print(f"Setting Gain to 0.0...")
    client.set_track_parameter(0, 2, p_idx, 0.0)
    time.sleep(2.0)
    
    print(f"Setting Gain to +15.0...")
    client.set_track_parameter(0, 2, p_idx, 15.0)
    time.sleep(2.0)
    
    print("\nCheck Ableton: Did it go from -15dB to +15dB?")

if __name__ == "__main__":
    debug_eq_raw()
