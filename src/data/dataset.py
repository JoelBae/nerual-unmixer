import os
import json
import torch
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset
import random

class ProxyDataset(Dataset):
    """
    Dataset for training Neural Proxies.
    
    For Effects (Saturator, EQ, etc.):
        Input: Dry Audio (from input_file)
        Params: Effect Parameters
        Target: Wet Audio (from output_file)
        
    For Source (Operator):
        Input: Fixed Source Signal (White Noise / Chirp) generated on-the-fly or fixed
        Params: Synth Parameters
        Target: Output Audio
    """
    def __init__(self, dataset_dir, effect_name, sample_rate=44100, duration=2.0, param_names=None):
        """
        Args:
            dataset_dir (str): Path to the specific effect dataset (e.g. 'dataset/saturator')
            effect_name (str): Name of the effect ('operator', 'saturator', etc.)
            sample_rate (int): Audio sample rate.
            duration (float): Audio duration in seconds.
            param_names (list): List of parameter names to extract from metadata, in order.
                                If None, attempts to infer or uses all available.
        """
        self.dataset_dir = dataset_dir
        self.effect_name = effect_name.lower()
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples_per_item = int(sample_rate * duration)
        
        self.metadata_path = os.path.join(dataset_dir, "metadata.json")
        
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")
            
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)
            
        self.param_names = param_names
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # 1. Load Audio
        output_path = os.path.join(self.dataset_dir, item["output_file"])
        
        # Use soundfile
        audio_data, sr = sf.read(output_path)
        
        # Ensure correct shape [Channels, Time]
        # sf.read return shape: (samples, channels) for multi-channel, or (samples,) for mono
        if audio_data.ndim == 1:
            # Mono: (samples,) -> (1, samples)
            target_audio = torch.from_numpy(audio_data).unsqueeze(0).float()
        else:
            # Multi: (samples, channels) -> (channels, samples)
            target_audio = torch.from_numpy(audio_data.T).float()
        
        # Resample/Crop if necessary
        if target_audio.size(1) > self.num_samples_per_item:
            target_audio = target_audio[:, :self.num_samples_per_item]
        elif target_audio.size(1) < self.num_samples_per_item:
            # Pad
            padding = self.num_samples_per_item - target_audio.size(1)
            target_audio = torch.nn.functional.pad(target_audio, (0, padding))
            
        if self.effect_name == "operator":
            # Source Proxy: Input is White Noise
            input_audio = torch.rand_like(target_audio) * 2 - 1 # White Noise [-1, 1]
        else:
            # Effect Proxy: Input is Dry Audio
            input_file = item.get("input_file")
            if input_file:
                input_path = os.path.join(self.dataset_dir, input_file)
                input_data_np, sr = sf.read(input_path)  # Use soundfile
                
                if input_data_np.ndim == 1:
                    input_audio = torch.from_numpy(input_data_np).unsqueeze(0).float()
                else:
                    input_audio = torch.from_numpy(input_data_np.T).float()
                
                if input_audio.size(1) > self.num_samples_per_item:
                    input_audio = input_audio[:, :self.num_samples_per_item]
                elif input_audio.size(1) < self.num_samples_per_item:
                    padding = self.num_samples_per_item - input_audio.size(1)
                    input_audio = torch.nn.functional.pad(input_audio, (0, padding))
            else:
                input_audio = torch.zeros_like(target_audio)

        # 2. Extract Parameters
        settings_map = {s["name"]: s["value"] for s in item["settings"]}
        
        params = []
        if self.param_names:
            for name in self.param_names:
                val = settings_map.get(name, 0.0)
                params.append(val)
        else:
            for name in sorted(settings_map.keys()):
                params.append(settings_map[name])
                
        params_tensor = torch.tensor(params, dtype=torch.float32)
        
        return input_audio, params_tensor, target_audio

def get_param_names(effect_name):
    # Hardcoded mapping matching generator.py
    if effect_name == "saturator":
        return ["Drive", "Output", "WS Curve", "WS Depth"]
    elif effect_name == "reverb":
        return ["Decay Time", "Size", "Dry/Wet"]
    elif effect_name == "operator":
        return [
            "Transpose", "Osc-A Wave", "Filter Freq", "Filter Res",
            "Fe Amount", "Fe Attack", "Fe Decay", "Fe Sustain", "Fe Release",
            "Pe Amount", "Pe Decay", "Pe Peak",
            "Ae Attack", "Ae Decay", "Ae Sustain", "Ae Release"
        ]
    elif effect_name == "eq8":
        # Note: 'Bond' typo should be checked in generator.py
        # Current metadata has "Bond 1 Freq" -> Keeping it for compatibility
        return [
             "Bond 1 Freq", "Band 1 Gain",
             "Band 2 Freq", "Band 2 Gain", "Band 2 Q",
             "Band 3 Freq", "Band 3 Gain", "Band 3 Q",
             "Band 4 Freq", "Band 4 Gain"
        ]
    elif effect_name == "ott":
        return ["Amount", "Time", "Output Gain"]
    elif effect_name == "phaser":
        return ["Frequency", "Feedback", "Amount"]
    return []
