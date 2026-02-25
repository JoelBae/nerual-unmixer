
from pythonosc import udp_client
from pythonosc import osc_server
from pythonosc import dispatcher
import threading
import time
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the schema from generator
from src.data.generator import ParameterRandomizer

class AbletonQueryClient:
    def __init__(self, ip="127.0.0.1", send_port=11000, recv_port=11001):
        self.client = udp_client.SimpleUDPClient(ip, send_port)
        self.dispatcher = dispatcher.Dispatcher()
        self.server = osc_server.ThreadingOSCUDPServer((ip, recv_port), self.dispatcher)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        self.device_names = {}
        self.param_names = {}
        self.last_value = None
        
        # Setup handlers
        self.dispatcher.map("/live/device/get/name", self.handle_device_name)
        self.dispatcher.map("/live/device/get/parameters/name", self.handle_param_names)
        
        # Trying to catch a parameter value reply
        # Standard: /live/device/get/param/value -> returns /live/device/get/param/value [track, device, param, index, value]
        self.dispatcher.map("/live/device/get/parameters/value", self.handle_param_value)
        self.dispatcher.map("/live/track/device/get/parameters/value", self.handle_param_value)
        self.dispatcher.map("/live/device/param/value", self.handle_param_value)

    def handle_device_name(self, address, *args):
        # Expected args: track_index, device_index, name
        if len(args) >= 3:
            key = f"{args[0]}_{args[1]}"
            self.device_names[key] = args[2]
            print(f"[REPLY] Device {args[1]}: {args[2]}")
        
    def handle_param_value(self, address, *args):
        print(f"[VALUE REPLY] From {address}: {args}")
        self.last_value = args

    def handle_param_names(self, address, *args):
        # Arg structure varies. Usually: track, device, param1, param2, ...
        # Or sometimes just a list of names if we queried specific index range?
        # Actually standard AbletonOSC returns: track, device, params...
        if len(args) >= 2:
            track_idx = args[0]
            dev_idx = args[1]
            params = args[2:]
            key = f"{track_idx}_{dev_idx}"
            self.param_names[key] = params
            print(f"[REPLY] Device {dev_idx} Params Received ({len(params)} params)")
    
    def query_param_value(self, track_index, device_index, param_index):
        # Trying a few variations for GET
        paths = [
            "/live/device/get/parameters/value",
            "/live/track/device/get/parameters/value",
            "/live/device/get/parameter/value",
            "/live/track/device/get/parameter/value"
        ]
        for path in paths:
             print(f"   > Querying {path}...")
             self.client.send_message(path, [track_index, device_index, param_index])
             time.sleep(0.1)

    def query_device_name(self, track_index, device_index):
        self.client.send_message("/live/device/get/name", [track_index, device_index])
        
    def query_param_names(self, track_index, device_index):
        self.client.send_message("/live/device/get/parameters/name", [track_index, device_index])

    def stop(self):
        self.server.shutdown()

def verify_setup():
    print("--- Verifying Ableton Setup ---")
    print("Ensure AbletonOSC is running and listening on 11000/11001.")
    
    client = AbletonQueryClient()
    
    # Init randomizer to get schema
    randomizer = ParameterRandomizer()
    schema = randomizer.schema
    
    print(f"Checking {len(schema)} devices defined in generator.py...")
    
    for device_def in schema:
        t_idx = device_def["track_index"]
        d_idx = device_def["device_index"]
        
        print(f"\n--- Checking Device {d_idx} ---")
        
        # 1. Query Name
        client.query_device_name(t_idx, d_idx)
        # 2. Query Params
        client.query_param_names(t_idx, d_idx)
        
        time.sleep(0.5) # Wait for replies
        
        # Check Params
        key = f"{t_idx}_{d_idx}"
        if key in client.param_names:
            actual_params = client.param_names[key]
            
            # Check each expected param
            for p in device_def["params"]:
                p_idx = p["index"]
                expected_name = p["name"]
                
                if p_idx < len(actual_params):
                    actual_name = actual_params[p_idx]
                    # Flexible fuzzy match or just report
                    match_status = "✅" if expected_name.lower() in actual_name.lower() or actual_name.lower() in expected_name.lower() else "❓ CHECK"
                    print(f"   [{match_status}] Index {p_idx}: Expected '{expected_name}' | Got '{actual_name}'")
                else:
                    print(f"   [❌ ERROR] Index {p_idx}: Out of bounds (Max {len(actual_params)-1})")
        else:
            print("   [⚠️ WARN] No parameter reply received. Is AbletonOSC active?")
            
    client.stop()

if __name__ == "__main__":
    verify_setup()
