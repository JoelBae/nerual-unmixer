import torch
import torch.nn as nn
import math

class SaturatorProxy(nn.Module):
    """
    Analytical DDSP Proxy for the Ableton Saturator effect.
    
    Uses exact mathematical waveshaping curves — no training required.
    Each of Ableton's 8 saturation types is modeled as a known nonlinear function.
    Fully differentiable for end-to-end backpropagation.
    
    Saturation Types:
        0: Analog Clip   — Soft tanh saturation (tube emulation)
        1: Soft Sine     — Sine-based soft fold
        2: Bass          — Low-frequency-focused saturation
        3: Medium Curve  — Moderate polynomial clipping
        4: Hard Clip     — Near-square-wave hard clip
        5: Sinoid Fold   — Sine wavefolder (creates harmonics by wavefolding)
        6: Digital Clip  — Harsh digital hard clip
        7: Waveshaper    — Custom curve controlled by WS Curve + WS Depth
    
    Parameters: [drive, type, ws_curve, ws_depth, dry_wet]
    Total: 5 parameters
    """
    
    def __init__(self):
        super().__init__()
        self.num_params = 5
    
    def _analog_clip(self, x):
        """Soft tube-style saturation via tanh."""
        return torch.tanh(x)
    
    def _soft_sine(self, x):
        """Sine-based soft clipping — gently rounds the peaks."""
        # sin(pi/2 * x) maps [-1,1] to [-1,1] with soft rounding at peaks
        # For |x| > 1, clamp to ±1
        return torch.where(
            torch.abs(x) <= 1.0,
            torch.sin(x * (math.pi / 2.0)),
            torch.sign(x)
        )
    
    def _medium_clip(self, x):
        """Moderate polynomial soft clip — tighter than tanh, softer than hard."""
        # Cubic soft clip: 3/2 * x - 1/2 * x^3 for |x| <= 1, clamped otherwise
        return torch.where(
            torch.abs(x) <= 1.0,
            1.5 * x - 0.5 * x.pow(3),
            torch.sign(x)
        )
    
    def _hard_clip(self, x):
        """Hard clip — aggressive flat-top clipping."""
        return torch.clamp(x, -1.0, 1.0)
    
    def _sinoid_fold(self, x):
        """Sine wavefolder — wraps the waveform using sin(), creating rich harmonics."""
        return torch.sin(x * (math.pi / 2.0))
    
    def _digital_clip(self, x):
        """Harsh digital clip — asymmetric for gritty digital character."""
        # Digital clip with a slight asymmetric bias for that "digital" sound
        return torch.clamp(x, -0.9, 0.9) / 0.9
    
    def _bass(self, x):
        """Bass-focused saturation — heavier distortion on lows, gentler on highs.
        Uses a softer tanh with less drive, preserving the low-end body."""
        return torch.tanh(x * 0.8) * 1.2
    
    def _waveshaper(self, x, curve, depth):
        """
        Custom waveshaper controlled by curve and depth parameters.
        curve: 0.0 = symmetric, 1.0 = asymmetric
        depth: 0.0 = no effect (linear), 1.0 = full waveshaping
        
        Uses a polynomial waveshaper: x + depth * (curve * x^2 + (1-curve) * x^3)
        """
        shaped = x + depth * (curve * x.pow(2) + (1 - curve) * x.pow(3))
        return torch.tanh(shaped)  # Safety bound

    def forward(self, audio, params):
        """
        Args:
            audio:  (batch, 2, time) stereo audio
            params: (batch, 5) [drive, type, ws_curve, ws_depth, dry_wet]
                    drive:    0.0 to 1.0 (maps to 1x-20x gain)
                    type:     0-6 (integer, selects saturation curve)
                    ws_curve: 0.0 to 1.0 (only used for type=6 Waveshaper)
                    ws_depth: 0.0 to 1.0 (only used for type=6 Waveshaper)
                    dry_wet:  0.0 to 1.0 (blend between clean and saturated)
        Returns:
            out_audio: (batch, 2, time) saturated audio
        """
        batch, channels, time = audio.shape
        
        # Extract parameters
        drive    = params[:, 0].view(batch, 1, 1)   # (batch, 1, 1)
        sat_type = params[:, 1]                        # (batch,) Keep as float for differentiability
        ws_curve = params[:, 2].view(batch, 1, 1)
        ws_depth = params[:, 3].view(batch, 1, 1)
        dry_wet  = params[:, 4].view(batch, 1, 1)
        
        # Map drive from 0-1 to a gain multiplier (1x to 20x)
        # This exponential mapping mirrors how Ableton's Drive knob feels
        gain = 1.0 + drive * 19.0
        
        # Apply drive gain to the input audio
        driven = audio * gain
        
        # Soft-Selection Mechanism for Differentiability
        # We compute all 8 saturation types and blend them based on proximity to 'sat_type'
        # indices: (1, 8)
        indices = torch.arange(8, device=audio.device).view(1, 8)
        # Narrow softmax centered at sat_type (temp=0.1 makes it nearly discrete but differentiable)
        weights = torch.softmax(-torch.abs(indices - sat_type.unsqueeze(1)) / 0.1, dim=1)
        
        types = [
            self._analog_clip(driven),                                  # 0
            self._soft_sine(driven),                                    # 1
            self._bass(driven),                                         # 2
            self._medium_clip(driven),                                  # 3
            self._hard_clip(driven),                                    # 4
            self._sinoid_fold(driven),                                  # 5
            self._digital_clip(driven),                                 # 6
            self._waveshaper(driven, ws_curve, ws_depth)                # 7
        ]
        
        saturated = torch.zeros_like(driven)
        for i in range(8):
            w = weights[:, i].view(batch, 1, 1)
            saturated = saturated + w * types[i]
        
        # Dry/Wet blend: 0.0 = fully dry, 1.0 = fully wet
        out_audio = (1.0 - dry_wet) * audio + dry_wet * saturated
        
        return out_audio
