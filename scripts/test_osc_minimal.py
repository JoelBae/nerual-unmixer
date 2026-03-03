from pythonosc import udp_client
import time

# Ableton default send port is 11000
IP = "127.0.0.1"
PORT = 11000

client = udp_client.SimpleUDPClient(IP, PORT)

print(f"Sending OSC messages to {IP}:{PORT}...")
print("Check Track 1 Volume and Master Fader in Ableton.")

# 1. Try moving Track 1 (Index 0) Volume
# Address: /live/track/set/volume [track_index, volume (0.0 - 1.0)]
print("Setting Track 1 Volume to 0.5...")
client.send_message("/live/track/set/volume", [0, 0.5])
time.sleep(1)

print("Setting Track 1 Volume to 0.8...")
client.send_message("/live/track/set/volume", [0, 0.8])
time.sleep(1)

# 2. Try starting Transport
print("Starting Transport (Play)...")
client.send_message("/live/song/start_playing", [])
time.sleep(2)

# 3. Try stopping Transport
print("Stopping Transport (Stop)...")
client.send_message("/live/song/stop_playing", [])

print("Test complete.")
