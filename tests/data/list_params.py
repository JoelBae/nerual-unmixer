import sys
import os
import time
import threading
from pythonosc import udp_client, dispatcher, osc_server

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

class ParamLister:
    def __init__(self, ip="127.0.0.1", send_port=11000, recv_port=11001):
        self.client = udp_client.SimpleUDPClient(ip, send_port)
        self.dispatcher = dispatcher.Dispatcher()
        self.server = osc_server.ThreadingOSCUDPServer((ip, recv_port), self.dispatcher)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        self.last_name = None
        self.event = threading.Event()
        
        # Map name replies
        self.dispatcher.map("/live/device/get/parameter/name", self.handle_name)
        self.dispatcher.map("/live/device/get/parameters/name", self.handle_name)

    def handle_name(self, address, *args):
        # args might be [track, device, param, name] or just [name]
        # We assume the last arg is the name
        if len(args) > 0:
            self.last_name = args[-1]
            self.event.set()

    def get_param_name(self, track, device, param):
        self.last_name = None
        self.event.clear()
        self.client.send_message("/live/device/get/parameter/name", [track, device, param])
        if self.event.wait(timeout=0.2):
            return self.last_name
        return None

    def list_params(self, track=0, device=1):
        print(f"\n--- Listing Parameters for Track {track}, Device {device} ---")
        print(f"{'IDX':<5} | {'NAME'}")
        print("-" * 30)
        
        for i in range(40): # Check first 40 params
            name = self.get_param_name(track, device, i)
            if name:
                print(f"{i:<5} | {name}")
            else:
                print(f"{i:<5} | [No Reply]")

if __name__ == "__main__":
    lister = ParamLister()
    # List params for Device 1 (Saturator)
    lister.list_params(0, 1)
