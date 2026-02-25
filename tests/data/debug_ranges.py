import sys
import os
import time

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.ableton_client import AbletonClient

def debug_ranges():
    print("Connecting to Ableton...")
    client = AbletonClient()
    
    print("\n--- Testing Device 0 (Instrument Rack) Range ---")
    print("Hypothesis: Macros use 0-127 Range.")
    
    # Macro 1 (Transpose)
    p_idx = 1
    
    print(f"Setting Macro {p_idx} to 0.0...")
    client.set_track_parameter(0, 0, p_idx, 0.0)
    time.sleep(1.0)
    
    print(f"Setting Macro {p_idx} to 64.0 (Midpoint)...")
    client.set_track_parameter(0, 0, p_idx, 64.0)
    time.sleep(1.0)
    
    print(f"Setting Macro {p_idx} to 127.0 (Max)...")
    client.set_track_parameter(0, 0, p_idx, 127.0)
    time.sleep(1.0)

    print("\n--- Testing Device 1 (Saturator) Range ---")
    print("Hypothesis: Native Devices use 0.0-1.0 Range.")
    
    # Drive (Param 1)
    p_idx = 1
    
    print(f"Setting Drive {p_idx} to 0.0...")
    client.set_track_parameter(0, 1, p_idx, 0.0)
    time.sleep(1.0)
    
    print(f"Setting Drive {p_idx} to 0.5 (Midpoint)...")
    client.set_track_parameter(0, 1, p_idx, 0.5)
    time.sleep(1.0)
    
    print(f"Setting Drive {p_idx} to 1.0 (Max)...")
    client.set_track_parameter(0, 1, p_idx, 1.0)
    time.sleep(1.0)
    
    print("\nCheck Ableton: Did BOTH move full range?")

if __name__ == "__main__":
    debug_ranges()
