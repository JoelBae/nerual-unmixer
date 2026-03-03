from src.data.ableton_osc_client import AbletonClient
import time

client = AbletonClient()
if client.ping():
    print("✅ Ableton Connection OK")
    # Get names for Track 1 (the first wet track)
    # Using raw OSC
    client.client.send_message("/live/track/get/devices/name", [1])
    time.sleep(1.0)
    # The response should be handled by the feedback server if it's running
    # Since I don't have a listener here, I'll just use my client's query logic if it supports it
