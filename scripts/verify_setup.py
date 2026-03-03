import sys
import os
import time
import threading

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.ableton_client import AbletonClient

def verify():
    client = AbletonClient()
    client.start_server()
    
    # 1. Ping
    if client.ping():
        print("✅ Ready to query Ableton.")
    else:
        print("❌ Ableton not responding on 11000/11001.")
        # return # Proceed anyway to see if fire-and-forget works

    # 2. Add listener for device names
    results = {}
    def on_device_name(address, *args):
        results['devices'] = args
        print(f"Devices on Track 1: {args}")

    def on_param_name(address, *args):
        # /live/device/get/parameters/name [track, device, index, name]
        print(f"Param: {args}")

    client.dispatcher.map("/live/track/get/devices/name", on_device_name)
    client.dispatcher.map("/live/device/get/parameters/name", on_param_name)

    print("\n--- Querying Track 1 Devices ---")
    client.client.send_message("/live/track/get/devices/name", [1])
    time.sleep(1.0)

    print("\n--- Querying Device 1 Parameters ---")
    # Query first 10 parameters
    for i in range(10):
        client.client.send_message("/live/device/get/parameters/name", [1, 1, i])
        time.sleep(0.05)
    
    time.sleep(1.0)
    print("\nVerification session finished.")

if __name__ == "__main__":
    verify()
