import sys
import os
import time
import random

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.ableton_client import AbletonClient
from src.data.verify_setup import AbletonQueryClient
from src.data.generator import ParameterRandomizer

def debug_saturator_randomization():
    print("Connecting to Ableton...")
    client = AbletonClient()
    query = AbletonQueryClient()
    randomizer = ParameterRandomizer()
    
    # Target: Saturator (Device 1)
    target_d_idx = 1
    
    print("\n--- Visual Verification: Saturator Type Randomization ---")
    print("Watch the 'Type' dropdown in Ableton's Saturator device.")
    print("It should change every 2 seconds.\n")
    
    for i in range(10):
        print(f"--- Iteration {i+1}/10 ---")
        
        # 1. Randomize from Schema
        flat_params, settings_log = randomizer.randomize(target_device_index=target_d_idx)
        
        # Extract the 'Type' setting for reporting
        type_setting = next((s for s in settings_log if s['device'] == target_d_idx and s['param'] == 3), None)
        
        # 2. Send to Ableton
        for setting in settings_log:
            # We only care about Saturator for this test
            if setting['device'] == target_d_idx:
                client.set_track_parameter(
                    setting["track"], 
                    setting["device"], 
                    setting["param"], 
                    setting["value"]
                )
        
        # 3. Report
        if type_setting:
            sent_val = type_setting['value']
            print(f"   [SENT] Type Value: {sent_val:.4f}")
            
            # 4. Verify Readback
            time.sleep(0.2) # Allow update
            read_val = query.query_param_value(0, target_d_idx, 3)
            print(f"   [READ] Type Value: {read_val}")
        else:
            print("   [ERROR] 'Type' parameter not found in randomized settings!")
            
        time.sleep(2.0)

if __name__ == "__main__":
    debug_saturator_randomization()
