import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
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
            try:
                self.devices['ott'].load_state_dict(torch.load(ott_path, map_location='cpu'))
                print(f"✅ Loaded OTT Proxy from {ott_path}")
            except Exception as e:
                print(f"⚠️ Failed to load OTT Proxy weights (schema mismatch?): {e}")
            
        # 3. Reverb Weights
        reverb_path = os.path.join(checkpoint_dir, "reverb_proxy.pt")
        if os.path.exists(reverb_path):
            try:
                self.devices['reverb'].load_state_dict(torch.load(reverb_path, map_location='cpu'))
                print(f"✅ Loaded Reverb Proxy from {reverb_path}")
            except Exception as e:
                print(f"⚠️ Failed to load Reverb Proxy weights: {e}")

    def forward(self, params_dict, sequence=None, wave_logits=None, order_logits=None):
        """
        Args:
            params_dict: Dictionary containing the parameter tensors for each device.
                         e.g., {'operator': tensor, 'eq8': tensor, ...}
            sequence: List of strings (device names) to process in order. 
                      If None, uses a default static chain.
            order_logits: (Batch, max_len, num_tokens) for V8 Autoregressive routing
        Returns:
            audio: (batch, channels, time) processed audio
        """
        # 1. Generate initial audio waveform
        audio = self.devices['operator'](params_dict['operator'], wave_logits=wave_logits)
        
        # 2A. V8 Autoregressive Differentiable Multiplexing
        if order_logits is not None:
            max_len = order_logits.shape[1]
            
            def _mux_step(audio_in, step_logits, sat_params, eq8_params, ott_params, rev_params):
                """One step of the multiplexer. Wrapped in checkpoint to save VRAM."""
                weights = torch.nn.functional.gumbel_softmax(step_logits, tau=1.0, hard=True, dim=-1)
                
                # Note: Gain Staging removed (V8.8 Tungsten)
                # Normalizing between effects ruins amplitude-dependent effects like Saturator.
                # Silence is now prevented naturally by the "Sound Floor" in the Operator proxy.
                
                # Clamp input to prevent NaN propagation from previous steps
                audio_in = torch.clamp(audio_in, -4.0, 4.0)
                audio_in = torch.nan_to_num(audio_in, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Armor: Neutralize NaNs in predicted parameters before they hit DSP math
                sat_params_safe = torch.nan_to_num(sat_params, nan=0.0)
                eq8_params_safe = torch.nan_to_num(eq8_params, nan=0.0)
                ott_params_safe = torch.nan_to_num(ott_params, nan=0.0)
                rev_params_safe = torch.nan_to_num(rev_params, nan=0.0)
                
                audio_sat = self.devices['saturator'](audio_in, sat_params_safe)
                audio_eq8 = self.devices['eq8'](audio_in, eq8_params_safe)
                audio_ott = self.devices['ott'](audio_in, ott_params_safe)
                audio_rev = self.devices['reverb'](audio_in, rev_params_safe)
                audio_eos = audio_in
                
                # Clamp each effect's output to prevent runaway amplitudes
                audio_sat = torch.nan_to_num(torch.clamp(audio_sat, -4.0, 4.0), nan=0.0)
                audio_eq8 = torch.nan_to_num(torch.clamp(audio_eq8, -4.0, 4.0), nan=0.0)
                audio_ott = torch.nan_to_num(torch.clamp(audio_ott, -4.0, 4.0), nan=0.0)
                audio_rev = torch.nan_to_num(torch.clamp(audio_rev, -4.0, 4.0), nan=0.0)
                
                all_audios = torch.stack([audio_sat, audio_eq8, audio_ott, audio_rev, audio_eos], dim=1)
                bcast_weights = weights.unsqueeze(-1).unsqueeze(-1)
                audio_out = (all_audios * bcast_weights).sum(dim=1)
                
                # Final safety clean for the next step/final output
                return torch.nan_to_num(audio_out, nan=0.0)
            
            for step in range(max_len):
                step_logits = order_logits[:, step, :]
                # Gradient checkpointing: discard intermediates, recompute during backward
                audio = cp.checkpoint(
                    _mux_step, audio, step_logits,
                    params_dict['saturator'], params_dict['eq8'],
                    params_dict['ott'], params_dict['reverb'],
                    use_reentrant=False
                )
                
            return audio
            
        # 2B. Legacy Static Sequence Processing
        if sequence is None:
            sequence = ['operator', 'eq8', 'saturator', 'ott', 'reverb']
            
        if sequence[0] != 'operator':
            raise ValueError("The first device in the chain must be an audio generator like 'operator'.")
            
        for device_name in sequence[1:]:
            if device_name in self.devices and device_name in params_dict:
                proxy = self.devices[device_name]
                device_params = params_dict[device_name]
                audio = proxy(audio, device_params)
                
        return audio

    def forward_flat(self, flat_params, order_idx=None, sequence=None, wave_logits=None, order_logits=None):
        """
        Helper for when parameters are a single flat tensor (Inverter output).
        Maps indices based on normalization.py / AGENT_GUIDE.md.
        
        Args:
            flat_params: (Batch, 62)
            order_idx: (Batch,) LongTensor - optional, overrides sequence
            sequence: List of strings - optional, default static if order_idx is None
        """
        # Map indices (Batch, 63) -> dict
        # Layout: Operator(16), Saturator(5), EQ8(32), OTT(7), Reverb(3)
        params_dict = {
            'operator':  flat_params[:, 0:16],
            'saturator': flat_params[:, 16:21],
            'eq8':       flat_params[:, 21:53],
            'ott':       flat_params[:, 53:60],
            'reverb':    flat_params[:, 60:63]
        }
        
        # Legacy: If order_idx is provided (from old datasets), try to decode it, 
        # but skip if we have order_logits
        if order_idx is not None and order_logits is None:
            effects = ['saturator', 'eq8', 'ott', 'reverb']
            perms = list(itertools.permutations(effects))
            idx = order_idx[0].item() if order_idx.dim() > 0 else order_idx
            sequence = ['operator'] + list(perms[idx])
            
        return self.forward(params_dict, sequence=sequence, wave_logits=wave_logits, order_logits=order_logits)
