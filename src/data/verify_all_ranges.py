import sys
import os
import time
import threading
from pythonosc import udp_client, dispatcher, osc_server

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.generator import ParameterRandomizer

class GlobalVerifier:
    def __init__(self, ip="127.0.0.1", send_port=11000, recv_port=11001):
        self.client = udp_client.SimpleUDPClient(ip, send_port)
        self.dispatcher = dispatcher.Dispatcher()
        self.server = osc_server.ThreadingOSCUDPServer((ip, recv_port), self.dispatcher)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        self.last_value = None
        self.event = threading.Event()
        
        # Map likely return paths
        self.dispatcher.map("/live/device/get/parameter/value", self.handle_value)
        self.dispatcher.map("/live/device/get/parameters/value", self.handle_value)

    def handle_value(self, address, *args):
        # Flatten args to find the value
        # Often args is a tuple like (track, device, param, value) OR just (value)
        # We'll take the LAST element as the value
        if len(args) > 0:
            self.last_value = args[-1]
            self.event.set()

    def query_param(self, t, d, p):
        self.last_value = None
        self.event.clear()
        # Try device specific query first
        self.client.send_message("/live/device/get/parameter/value", [t, d, p])
        if self.event.wait(timeout=0.2):
            return self.last_value
        return None

    def set_param(self, t, d, p, val):
        self.client.send_message("/live/device/set/parameter/value", [t, d, p, val])
        time.sleep(0.05)

    def run(self):
        verifier = ParameterRandomizer()
        print(f"\n{'DEVICE':<15} | {'PARAM':<20} | {'IDX':<4} | {'TEST (2.0)':<10} | {'RESULT':<10} | {'TYPE'}")
        print("-" * 90)

        for device in verifier.schema:
            t_idx = device["track_index"]
            d_idx = device["device_index"]
            
            # Skip Instrument Macros (0) - Verified 0-127
            if d_idx == 0: continue
            
            for param in device["params"]:
                p_idx = param["index"]
                p_name = param["name"]
                
                # Skip EQ Gains (2) - Verified -15 to +15
                if d_idx == 2 and "Gain" in p_name: continue

                # Send 2.0. 
                # If Normalized -> Clamps to 1.0 (or 0.0-1.0 range)
                # If Raw -> Becomes 2.0 (or 2.0 Hz, 2.0 ms, etc.)
                self.set_param(t_idx, d_idx, p_idx, 2.0)
                val = self.query_param(t_idx, d_idx, p_idx)
                
                # Logic
                p_type = "UNKNOWN"
                if val is not None:
                    if val > 1.05:
                        p_type = "🔴 RAW"
                    elif 0.9 <= val <= 1.05:
                        p_type = "🟢 NORM"
                    else:
                        p_type = f"⚠️ {val}"
                else:
                    p_type = "❌ NO REPLY"

                print(f"Dev {d_idx:<11} | {p_name[:18]:<20} | {p_idx:<4} | {'2.0':<10} | {str(val)[:8]:<10} | {p_type}")
                
        print("-" * 90)

if __name__ == "__main__":
    v = GlobalVerifier()
    v.run()
