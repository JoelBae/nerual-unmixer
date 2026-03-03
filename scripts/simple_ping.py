import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.ableton_client import AbletonClient

client = AbletonClient()
print("Starting OSC Server...")
client.start_server()
print("Pinging Ableton...")
if client.ping(timeout=2.0):
    print("✅ PING SUCCESS")
else:
    print("❌ PING FAILED")
