import torch.nn as nn
from .base import BaseProxyModule

class ProxySaturator(BaseProxyModule):
    """
    Simulates Ableton Saturator.
    Params: Drive, Curve, Color, etc. (~6 params).
    Non-linearity is key here.
    """
    def __init__(self):
        super().__init__(param_dim=6, num_channels=[16, 32, 64, 64])

class ProxyEQ8(BaseProxyModule):
    """
    Simulates Ableton EQ Eight.
    4 Bands x 3 Params (Freq, Gain, Q) + Global = ~13 params.
    Filters need good frequency resolution.
    """
    def __init__(self):
        super().__init__(param_dim=13, num_channels=[32, 64, 128, 128])

class ProxyOTT(BaseProxyModule):
    """
    Simulates Multiband Dynamics (OTT Preset).
    Params: Depth, Time, In Gain, Out Gain, High/Mid/Low thresholds (~10 params).
    Strong dynamic processing.
    """
    def __init__(self):
        super().__init__(param_dim=10, num_channels=[32, 64, 128, 128])

class ProxyPhaser(BaseProxyModule):
    """
    Simulates Ableton Phaser-Flanger.
    Params: Rate, Amount, Feedback, Phase, Center (~5 params).
    Modulation effects need to capture time-varying characteristics.
    """
    def __init__(self):
        super().__init__(param_dim=5, num_channels=[24, 48, 48])

class ProxyReverb(BaseProxyModule):
    """
    Simulates Ableton Reverb.
    Params: Decay, Size, Stereo, Dry/Wet, Input Filter (~8 params).
    Long temporal scope required.
    """
    def __init__(self):
        # Increased depth and channels for long reverb tails
        super().__init__(param_dim=8, num_channels=[32, 64, 128, 128, 128])
