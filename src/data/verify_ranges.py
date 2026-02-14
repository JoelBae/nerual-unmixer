import sys
import os
import time
import argparse
from pythonosc import udp_client, dispatcher, osc_server
import threading

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# We need the schema to know what to test
from src.data.generator import ParameterRandomizer

class RangeVerifier:
    def __init__(self, ip="127.0.0.1", send_port=11000, recv_port=11001):
        self.client = udp_client.SimpleUDPClient(ip, send_port)
        self.dispatcher = dispatcher.Dispatcher()
        self.server = osc_server.ThreadingOSCUDPServer((ip, recv_port), self.dispatcher)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        self.last_value = None
        self.event = threading.Event()
        
        # Map value replies
        self.dispatcher.map("/live/device/get/parameter/value", self.handle_value)
        self.dispatcher.map("/live/device/get/parameters/value", self.handle_value) # Fallback

    def handle_value(self, address, *args):
        # args might be [track, device, param, value] or just [value] depending on impl
        # We assume the last arg is the value
        if len(args) > 0:
            self.last_value = args[-1]
            self.event.set()

    def get_value(self, track, device, param):
        self.last_value = None
        self.event.clear()
        self.client.send_message("/live/device/get/parameter/value", [track, device, param])
        if self.event.wait(timeout=1.0):
            return self.last_value
        return None

    def set_value(self, track, device, param, value):
        self.client.send_message("/live/device/set/parameter/value", [track, device, param, value])
        time.sleep(0.1) # Wait for processing

    def verify_all(self):
        randomizer = ParameterRandomizer()
        
        print(f"{'DEVICE':<15} | {'PARAM':<20} | {'IDX':<4} | {'0.0 -> ?':<10} | {'1.0 -> ?':<10} | {'RAW?':<5}")
        print("-" * 80)
        
        for device in randomizer.schema:
            t_idx = device["track_index"]
            d_idx = device["device_index"]
            
            # Skip Device 0 (Instrument Macros) as we verified they are 0-127 MIDI
            if d_idx == 0:
                continue

            # Skip Device 2 (EQ8) Gain params as we verified they are -15 to +15
            # But let's check Freq/Q
            
            for param in device["params"]:
                p_idx = param["index"]
                p_name = param["name"]
                
                # Skip known verified ranges
                if "Gain" in p_name and d_idx == 2: continue 
                
                # Test 1: Set 0.0
                self.set_value(t_idx, d_idx, p_idx, 0.0)
                val_0 = self.get_value(t_idx, d_idx, p_idx)
                
                # Test 2: Set 1.0
                self.set_value(t_idx, d_idx, p_idx, 1.0)
                val_1 = self.get_value(t_idx, d_idx, p_idx)
                
                # Analysis
                # If 1.0 sets it to ~1.0, it's normalized.
                # If 1.0 sets it to 1.0 but max is 20000, it might be raw but 1Hz is valid.
                # Let's try 0.5.
                self.set_value(t_idx, d_idx, p_idx, 0.5)
                val_05 = self.get_value(t_idx, d_idx, p_idx)
                
                is_raw_suspect = False
                if val_1 is not None and val_05 is not None:
                     # If 0.5 results in exactly 0.5, it's likely normalized.
                     # If 0.5 results in something else (like logarithmic mapping), it's weird.
                     pass

                print(f"Dev {d_idx} {p_name[:12]:<12} | {p_idx:<4} | {str(val_0):<10} | {str(val_1):<10} | {val_05}")

if __name__ == "__main__":
    verifier = RangeVerifier()
    verifier.verify_all()
