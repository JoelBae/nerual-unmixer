import torch
import torch.nn as nn
from .modules import ProxySaturator, ProxyEQ8, ProxyOTT, ProxyPhaser, ProxyReverb

class ProxyChain(nn.Module):
    """
    Dynamically assembles and runs a chain of effect proxies.
    """
    def __init__(self):
        super().__init__()
        # Registry of available effects
        self.effects = nn.ModuleDict({
            'saturator': ProxySaturator(),
            'eq8': ProxyEQ8(),
            'ott': ProxyOTT(),
            'phaser': ProxyPhaser(),
            'reverb': ProxyReverb()
        })

    def forward(self, input_audio, chain_config):
        """
        input_audio: (batch, channels, length)
        chain_config: List of dicts [{"type": "filter", "params": tensor}, ...]
                      OR a batch-friendly structure. 
                      
        For batched training with DYNAMIC chains, typically we use a 
        Fixed Maximum Chain (e.g., Slot 1, Slot 2, Slot 3) and mask/bypass unused slots.
        
        Here we assume a fixed order for now (Series Chain) or a provided list for inference.
        """
        x = input_audio
        
        # Example: Hardcoded chain flow for v1, or iterate if config provided
        if isinstance(chain_config, list):
             for effect_def in chain_config:
                 effect_name = effect_def['type']
                 params = effect_def['params']
                 
                 if effect_name in self.effects:
                     x = self.effects[effect_name](x, params)
        
        return x
