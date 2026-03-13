import numpy as np
import soundfile as sf
import os
from typing import Optional, List, Dict, Union, Tuple, Any
from pedalboard import Pedalboard, load_plugin, VST3Plugin
import mido
import torch

class PedalboardEngine:
    """
    High-performance rendering engine using Spotify's pedalboard.
    Generates ground-truth audio and parameter Jacobians for Neural Unmixer.
    """
    
    VST_PATHS = {
        "vital": "/Library/Audio/Plug-Ins/VST3/Vital.vst3",
        "ott": "/Library/Audio/Plug-Ins/VST3/OTT.vst3",
        "kh_eq3band": "/Library/Audio/Plug-Ins/VST3/Kilohearts/kHs 3-Band EQ.vst3",
        "kh_distortion": "/Library/Audio/Plug-Ins/VST3/Kilohearts/kHs Distortion.vst3",
        "kh_reverb": "/Library/Audio/Plug-Ins/VST3/Kilohearts/kHs Reverb.vst3"
    }

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.loaded_plugins: Dict[str, VST3Plugin] = {}
        self.param_cache: Dict[str, Dict[str, Any]] = {}

    def load_plugin(self, plugin_id: str) -> VST3Plugin:
        """Loads a VST3 plugin by its ID."""
        if plugin_id not in self.VST_PATHS:
            raise ValueError(f"Unknown plugin ID: {plugin_id}")
        
        path = self.VST_PATHS[plugin_id]
        if not os.path.exists(path):
            raise FileNotFoundError(f"VST3 plugin not found at: {path}")
            
        if plugin_id not in self.loaded_plugins:
            print(f"Loading {plugin_id}... (this may be slow for large plugins)")
            plugin = load_plugin(path)
            self.loaded_plugins[plugin_id] = plugin
            self.param_cache[plugin_id] = {}
            
        return self.loaded_plugins[plugin_id]

    def _get_cached_param(self, plugin_id: str, name: str) -> Optional[Any]:
        plugin = self.loaded_plugins[plugin_id]
        cache = self.param_cache[plugin_id]
        
        if name in cache:
            return cache[name]
        
        # Try direct lookup
        if name in plugin.parameters:
            cache[name] = plugin.parameters[name]
            return cache[name]
            
        # Try lowercase lookup (one-time scan for cache)
        if "_all_lower" not in cache:
            print(f"Building case-insensitive parameter map for {plugin_id}...")
            cache["_all_lower"] = {k.lower(): v for k, v in plugin.parameters.items()}
            
        lower_name = name.lower()
        if lower_name in cache["_all_lower"]:
            param = cache["_all_lower"][lower_name]
            cache[name] = param
            return param
            
        return None

    def render(
        self, 
        plugin_id: str, 
        parameters: Dict[str, float], 
        duration_sec: float = 1.0,
        input_audio: Optional[np.ndarray] = None,
        note_freq: Optional[float] = None
    ) -> np.ndarray:
        plugin = self.load_plugin(plugin_id)
        plugin.reset() # Ensure clean state for verification
        
        # Set parameters
        for name, value in parameters.items():
            target_param = self._get_cached_param(plugin_id, name)
            
            if target_param is not None:
                # Use raw_value for normalized [0, 1] setting
                target_param.raw_value = value
            else:
                print(f"Warning: Parameter '{name}' not found in {plugin_id}")

        num_samples = int(duration_sec * self.sample_rate)
        
        if plugin_id == "vital":
            # For Vital, we trigger a MIDI note at the start
            note = 60 if note_freq is None else int(12 * np.log2(note_freq / 440.0) + 69)
            messages = [
                mido.Message("note_on", note=note, velocity=100, time=0),
                mido.Message("note_off", note=note, time=num_samples - 1)
            ]
            audio = plugin.process(messages, duration=duration_sec, sample_rate=self.sample_rate)
        else:
            if input_audio is None:
                input_audio = np.zeros((2, num_samples), dtype=np.float32)
            
            audio = plugin.process(input_audio, sample_rate=self.sample_rate)
            
        return audio

    def compute_jacobian(
        self,
        plugin_id: str,
        base_params: Dict[str, float],
        target_param_names: List[str],
        duration_sec: float = 0.5,
        delta: float = 1e-3,
        input_audio: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Computes the Jacobian of the audio output with respect to parameters using finite differences.
        
        Returns:
            A dictionary mapping parameter names to their partial derivatives (np.ndarray).
        """
        base_audio = self.render(plugin_id, base_params, duration_sec, input_audio)
        jacobians = {}
        
        for param_name in target_param_names:
            perturbed_params = base_params.copy()
            
            # Step forward
            perturbed_params[param_name] = min(1.0, base_params[param_name] + delta)
            audio_plus = self.render(plugin_id, perturbed_params, duration_sec, input_audio)
            
            # Step backward
            perturbed_params[param_name] = max(0.0, base_params[param_name] - delta)
            audio_minus = self.render(plugin_id, perturbed_params, duration_sec, input_audio)
            
            # Central difference
            actual_delta = (min(1.0, base_params[param_name] + delta) - 
                           max(0.0, base_params[param_name] - delta))
            
            if actual_delta > 0:
                grad = (audio_plus - audio_minus) / actual_delta
            else:
                grad = np.zeros_like(base_audio)
                
            jacobians[param_name] = grad
            
        return jacobians

if __name__ == "__main__":
    # Quick sanity check
    engine = PedalboardEngine()
    try:
        ott = engine.load_plugin("ott")
        print("Successfully loaded OTT")
        
        # Simple render test
        silence = np.zeros((2, 44100), dtype=np.float32)
        out = engine.render("ott", {"depth": 0.5}, input_audio=silence)
        print(f"Rendered audio shape: {out.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
