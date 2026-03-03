from pythonosc import udp_client
from pythonosc import osc_message_builder
from pythonosc import dispatcher, osc_server
import time
import threading

class AbletonClient:
    """
    Client for communicating with Ableton Live via AbletonOSC.
    Assumes AbletonOSC is installed and running on port 11000 (send) / 11001 (receive).
    """
    def __init__(self, ip="127.0.0.1", send_port=11000, recv_port=11001):
        self.client = udp_client.SimpleUDPClient(ip, send_port)
        self.ip = ip
        self.send_port = send_port
        self.recv_port = recv_port
        
        # Bidirectional communication setup
        self.dispatcher = dispatcher.Dispatcher()
        self.server = None
        self.server_thread = None
        self._ping_event = threading.Event()
        self._track_count = None
        self._track_count_event = threading.Event()
        
        # Setup listeners
        self.dispatcher.map("/live/song/get/num_tracks", self._handle_track_count)
        self.dispatcher.map("/live/test/ping", self._handle_ping) # Custom ping if supported by script
        # Alternate ping: get master volume as a heartbeat
        self.dispatcher.map("/live/master/get/volume", self._handle_ping)

    def start_server(self):
        """Start the internal OSC server to listen for responses from Ableton."""
        if self.server is not None:
            return
            
        try:
            self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.recv_port), self.dispatcher)
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            print(f"📡 OSC Feedback Server started on {self.ip}:{self.recv_port}")
        except Exception as e:
            print(f"⚠️ Warning: Could not start OSC server: {e}. Check if port {self.recv_port} is in use.")

    def _handle_ping(self, address, *args):
        self._ping_event.set()

    def _handle_track_count(self, address, *args):
        if len(args) > 0:
            self._track_count = args[0]
            self._track_count_event.set()

    def ping(self, timeout=0.5):
        """
        Check if Ableton is responsive.
        Returns True if a handshake is successful.
        """
        if self.server is None:
            self.start_server()
            
        self._ping_event.clear()
        # We request master volume as a standard heartbeat
        self.client.send_message("/live/master/get/volume", [])
        return self._ping_event.wait(timeout=timeout)

    def get_track_count(self, timeout=0.5):
        """Request the number of tracks from Ableton."""
        if self.server is None:
            self.start_server()
            
        self._track_count_event.clear()
        self.client.send_message("/live/song/get/num_tracks", [])
        if self._track_count_event.wait(timeout=timeout):
            return self._track_count
        return None

    def send_message(self, address, value=None, throttle=0.0):
        """
        Send a generic OSC message with optional throttling.
        """
        if value is None:
            self.client.send_message(address, [])
        else:
            self.client.send_message(address, value)
            
        if throttle > 0:
            time.sleep(throttle)

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
