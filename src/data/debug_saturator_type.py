import sys
import os
import time
import argparse
from pythonosc import udp_client

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.ableton_client import AbletonClient
from src.data.verify_setup import AbletonQueryClient

def debug_type():
    print("Connecting to Ableton...")
    client = AbletonClient()
    query = AbletonQueryClient()
    
    # Device 1: Saturator
    # Param 3: Type
    d = 1
    p = 3
    
    steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    print(f"\n--- Testing Saturator Type (Param {p}) ---")
    
    for val in steps:
        print(f"Setting Type to {val}...")
        client.set_track_parameter(0, d, p, val)
        time.sleep(0.5)
        
        # Query value to see what Ableton thinks it is (might be raw index?)
        read_val = query.query_param_value(0, d, p)
        print(f"   -> Read back: {read_val}")
        time.sleep(1.0) # Time to look at screen

if __name__ == "__main__":
    debug_type()
