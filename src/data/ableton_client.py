from pythonosc import udp_client
from pythonosc import osc_message_builder
import time

class AbletonClient:
    """
    Client for communicating with Ableton Live via AbletonOSC.
    Assumes AbletonOSC is installed and running on port 11000 (send) / 11001 (receive).
    """
    def __init__(self, ip="127.0.0.1", port=11000):
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.ip = ip
        self.port = port

    def send_message(self, address, value=None):
        """
        Send a generic OSC message.
        """
        if value is None:
            self.client.send_message(address, [])
        else:
            self.client.send_message(address, value)

    def set_track_parameter(self, track_index, device_index, parameter_index, value):
        """
        /live/track/device/param/value (int track_index, int device_index, int parameter_index, int|float value)
        """
        # Verified working path for this user setup:
        # /live/device/set/parameter/value [track, device, param, value]
        address = "/live/device/set/parameter/value"
        self.client.send_message(address, [track_index, device_index, parameter_index, value])

    def set_device_enabled(self, track_index, device_index, enabled: bool):
        """
        /live/track/device/set/is_active (int track_index, int device_index, bool enabled)
        Note: AbletonOSC property is typically 'is_active' or 'is_enabled'. 
        We use generic 'set' command if available, or try specific endpoint.
        
        Using device-centric path:
        /live/device/set/is_active
        """
        # Converting bool to int (0 or 1)
        val = 1 if enabled else 0
        address = "/live/device/set/is_active" 
        self.client.send_message(address, [track_index, device_index, val])

    def set_master_volume(self, volume):
        """
        /live/master/volume (float volume)
        """
        self.client.send_message("/live/master/volume", volume)

    def play(self):
        self.client.send_message("/live/song/start_playing", [])

    def stop(self):
        self.client.send_message("/live/song/stop_playing", [])
    
    def get_track_name(self, track_index):
        # Note: This is a fire-and-forget client. 
        # For receiving data, we would need an OSC server running in a separate thread.
        # This function is just a placeholder for the command.
        pass
