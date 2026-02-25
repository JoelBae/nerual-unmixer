import sys
import os
import time

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.ableton_client import AbletonClient
from src.data.verify_setup import AbletonQueryClient

def debug_enum():
    print("Connecting to Ableton...")
    client = AbletonClient()
    query = AbletonQueryClient()
    
    # Device 1: Saturator
    # Param 3: Type (Enum)
    # 0 = Analog Clip
    # 1 = Soft Sine
    # 2 = Medium Curve
    # 3 = Hard Curve
    # 4 = Sinoid Fold
    # 5 = Digital Clip
    # 6 = Waveshaper
    
    d = 1
    p = 3
    
    print(f"\n--- Testing Saturator Type (Param {p}) with RAW INTEGERS ---")
    
    for val in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        print(f"Setting Type to {val}...")
        client.set_track_parameter(0, d, p, val)
        time.sleep(1.5) # Time to look at screen

if __name__ == "__main__":
    debug_enum()
