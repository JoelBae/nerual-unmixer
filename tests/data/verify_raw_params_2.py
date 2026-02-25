import sys
import os
import time

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.ableton_client import AbletonClient
from src.data.verify_setup import AbletonQueryClient

def verify_raw_params_2():
    print("Connecting to Ableton...")
    client = AbletonClient()
    query = AbletonQueryClient()
    
    # PROBE LIST: Devices that might have non-normalized scales
    to_check = [
        {"desc": "Saturator 'Drive' (idx 1)", "track": 0, "device": 1, "param": 1},
        {"desc": "EQ8 'Freq Band 1' (idx 6)", "track": 0, "device": 2, "param": 6},
        {"desc": "OTT 'Output Gain' (idx 5)", "track": 0, "device": 3, "param": 5},
        {"desc": "Phaser 'Freq' (idx 3)", "track": 0, "device": 4, "param": 3},
        {"desc": "Reverb 'Decay Time' (idx 20)", "track": 0, "device": 5, "param": 20}
    ]

    print(f"\n{'PARAM':<30} | {'SENT':<10} | {'GOT':<10} | {'IsRaw?'}")
    print("-" * 70)

    for item in to_check:
        t, d, p = item["track"], item["device"], item["param"]
        desc = item["desc"]
        
        # Test 1: Send 2.0 (If Normalized, should cap at 1.0)
        client.set_track_parameter(t, d, p, 2.0)
        time.sleep(0.2)
        val_2 = query.query_param_value(t, d, p)
        
        # Test 2: Send 20.0 (If Hz or dB, should be 20.0)
        client.set_track_parameter(t, d, p, 20.0)
        time.sleep(0.2)
        val_20 = query.query_param_value(t, d, p)
        
        # Interpret
        if val_2 is None: val_2 = -1.0
        if val_20 is None: val_20 = -1.0
        
        is_raw = "UNKNOWN"
        if val_2 > 1.05: 
            is_raw = "YES (RAW)"
        elif val_2 <= 1.0: 
            is_raw = "NO (Normalized)"
            
        print(f"{desc:<30} | {'2.0':<10} | {str(val_2):<10} | {is_raw}")
        print(f"{'':<30} | {'20.0':<10} | {str(val_20):<10} | {'Maybe? (dB/Hz)'}")

if __name__ == "__main__":
    verify_raw_params_2()
