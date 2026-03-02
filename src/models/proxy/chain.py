import torch
import torch.nn as nn
import os
import itertools
from src.models.proxy.ddsp_modules import OperatorProxy
from src.models.proxy.eq8 import EQEightProxy
from src.models.proxy.saturator import SaturatorProxy
from src.models.proxy.reverb import ReverbProxy
from src.models.proxy.ott import OTTProxy

class ProxyChainer(nn.Module):
    """
    Dynamically routes audio through a sequence of Differentiable DSP Proxies.
    This enables end-to-end learning where gradients flow through the entire signal chain.
    """
    def __init__(self, sr=44100):
        super().__init__()
        self.sr = sr
        
        # Instantiate all available proxy devices
        self.devices = nn.ModuleDict({
            'operator': OperatorProxy(sample_rate=sr),
            'eq8': EQEightProxy(sr=sr),
            'saturator': SaturatorProxy(),
            'reverb': ReverbProxy(sr=sr),
            'ott': OTTProxy(sr=sr)
        })
        
    def load_checkpoints(self, checkpoint_dir="checkpoints"):
        """Automatically loads available proxy weights."""
        # 1. Operator Wave Table
        wave_path = os.path.join(checkpoint_dir, "wave_table.pt")
        self.devices['operator'].load_wave_table(wave_path)
        
        # 2. OTT Weights
        ott_path = os.path.join(checkpoint_dir, "ott_proxy.pt")
        if os.path.exists(ott_path):
            self.devices['ott'].load_state_dict(torch.load(ott_path, map_location='cpu'))
            print(f"✅ Loaded OTT Proxy from {ott_path}")
            
        # 3. Reverb Weights
        reverb_path = os.path.join(checkpoint_dir, "reverb_proxy.pt")
        if os.path.exists(reverb_path):
            self.devices['reverb'].load_state_dict(torch.load(reverb_path, map_location='cpu'))
            print(f"✅ Loaded Reverb Proxy from {reverb_path}")

    def forward(self, params_dict, sequence=None):
        """
        Args:
            params_dict: Dictionary containing the parameter tensors for each device.
                         e.g., {'operator': tensor, 'eq8': tensor, ...}
            sequence: List of strings (device names) to process in order. 
                      If None, uses a default static chain.
        Returns:
            audio: (batch, channels, time) processed audio
        """
        # Default chain if none provided
        if sequence is None:
            sequence = ['operator', 'eq8', 'saturator', 'ott', 'reverb']
            
        # The chain must start with a generator like Operator for synth parameters.
        # Alternatively, we could accept an input audio arg for processing existing files.
        if sequence[0] != 'operator':
            raise ValueError("The first device in the chain must be an audio generator like 'operator'.")
            
        # 1. Generate initial audio waveform
        audio = self.devices['operator'](params_dict['operator'])
        
        # 2. Pass through the effect chain dynamically
        for device_name in sequence[1:]:
            if device_name in self.devices and device_name in params_dict:
                proxy = self.devices[device_name]
                device_params = params_dict[device_name]
                
                # Apply the effect
                audio = proxy(audio, device_params)
                
        return audio

    def forward_flat(self, flat_params, order_idx=None, sequence=None):
        """
        Helper for when parameters are a single flat tensor (Inverter output).
        Maps indices based on normalization.py / AGENT_GUIDE.md.
        
        Args:
            flat_params: (Batch, 62)
            order_idx: (Batch,) LongTensor - optional, overrides sequence
            sequence: List of strings - optional, default static if order_idx is None
        """
        # Map indices (Batch, 62) -> dict
        params_dict = {
            'operator':  flat_params[:, 0:16],
            'saturator': flat_params[:, 16:20],
            'eq8':       flat_params[:, 20:52],
            'ott':       flat_params[:, 52:59],
            'reverb':    flat_params[:, 59:62]
        }
        
        # If order_idx is provided, decode it into a sequence
        if order_idx is not None:
            effects = ['saturator', 'eq8', 'ott', 'reverb']
            perms = list(itertools.permutations(effects))
            
            # We assume the whole batch uses the same order for the differentiable pass
            # during a single step (standard for this system). 
            # We take the most common or first index.
            # In a fully vectorized world, we'd need to handle heterogenous batches,
            # but Ableton chains are static per-track.
            idx = order_idx[0].item()
            sequence = ['operator'] + list(perms[idx])
            
        return self.forward(params_dict, sequence=sequence)
