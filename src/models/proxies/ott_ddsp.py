import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

import scipy.signal

class Biquad(nn.Module):
    """General Biquad Filter Module (Optimized)"""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, samples)
            b: (batch, 3) [b0, b1, b2]
            a: (batch, 2) [a1, a2] (a0 assumed 1.0)
        """
        # Fast path for Jacobian estimation (no grad)
        if not x.requires_grad:
            x_np = x.detach().cpu().numpy()
            b_np = b.detach().cpu().numpy()
            a_np = a.detach().cpu().numpy()
            out = np.zeros_like(x_np)
            for i in range(x.shape[0]):
                # Scipy lfilter expects [1, a1, a2]
                coeffs_a = np.array([1.0, a_np[i, 0], a_np[i, 1]])
                out[i] = scipy.signal.lfilter(b_np[i], coeffs_a, x_np[i])
            return torch.from_numpy(out).to(x.device).float()
            
        # Standard path (Differentiable but slow)
        batch_size, n_samples = x.shape
        device = x.device
        z1 = torch.zeros(batch_size, device=device)
        z2 = torch.zeros(batch_size, device=device)
        out = torch.zeros_like(x)
        for i in range(n_samples):
            v0 = x[:, i]
            v1 = b[:, 0] * v0 + z1
            out[:, i] = v1
            z1 = b[:, 1] * v0 - a[:, 0] * v1 + z2
            z2 = b[:, 2] * v0 - a[:, 1] * v1
        return out

@torch.jit.script
def smooth_log_level(rect_db: torch.Tensor, alpha_at: float, alpha_rel: float):
    batch_size, n_samples = rect_db.shape
    device = rect_db.device
    
    # Fast warm-up: initialize with the first sample
    curr_level_db = rect_db[:, 0]
    level_db = torch.zeros_like(rect_db)
    
    for i in range(n_samples):
        v = rect_db[:, i]
        mask = (v > curr_level_db).float()
        alpha = mask * alpha_at + (1.0 - mask) * alpha_rel
        curr_level_db = alpha * curr_level_db + (1.0 - alpha) * v
        level_db[:, i] = curr_level_db
    return level_db

class LinkwitzRiley4Filters(nn.Module):
    """4th order Linkwitz-Riley Filter Pair (LP and HP)"""
    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.sample_rate = sample_rate
        self.biquad1 = Biquad()
        self.biquad2 = Biquad()

    def _get_coeffs(self, cutoff_hz: float, mode: str):
        # Bilinear transform for Butterworth
        omega = 2 * np.pi * cutoff_hz / self.sample_rate
        sn = np.sin(omega)
        cs = np.cos(omega)
        alpha = sn / np.sqrt(2) # Q = 1/sqrt(2) for Butterworth
        
        a0 = 1 + alpha
        if mode == 'lp':
            b0 = (1 - cs) / 2; b1 = 1 - cs; b2 = (1 - cs) / 2
            a1 = -2 * cs; a2 = 1 - alpha
        else: # hp
            b0 = (1 + cs) / 2; b1 = -(1 + cs); b2 = (1 + cs) / 2
            a1 = -2 * cs; a2 = 1 - alpha
            
        return [b0/a0, b1/a0, b2/a0], [a1/a0, a2/a0]

    def forward(self, x: torch.Tensor, cutoff_hz: float) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        device = x.device
        b_lp, a_lp = self._get_coeffs(cutoff_hz, 'lp')
        b_lp = torch.tensor(b_lp, device=device).unsqueeze(0).repeat(batch_size, 1)
        a_lp = torch.tensor(a_lp, device=device).unsqueeze(0).repeat(batch_size, 1)
        b_hp, a_hp = self._get_coeffs(cutoff_hz, 'hp')
        b_hp = torch.tensor(b_hp, device=device).unsqueeze(0).repeat(batch_size, 1)
        a_hp = torch.tensor(a_hp, device=device).unsqueeze(0).repeat(batch_size, 1)
        lp = self.biquad2(self.biquad1(x, b_lp, a_lp), b_lp, a_lp)
        hp = self.biquad2(self.biquad1(x, b_hp, a_hp), b_hp, a_hp)
        return lp, hp

class DifferentiableCrossover3Band(nn.Module):
    """3-band Linkwitz-Riley Crossover using proper filters"""
    def __init__(self, low_hz: float = 88.3, high_hz: float = 2500, sample_rate: int = 44100):
        super().__init__()
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.lr_low_1 = LinkwitzRiley4Filters(sample_rate)
        self.lr_high = LinkwitzRiley4Filters(sample_rate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low, remainder = self.lr_low_1(x, self.low_hz)
        mid, high = self.lr_high(remainder, self.high_hz)
        return low, mid, high

class DifferentiableCompressor(nn.Module):
    """Refined Differentiable Upward/Downward Compressor"""
    def __init__(self, sample_rate: int = 44100, 
                 attack_ms: float = 10.0, 
                 release_ms: float = 100.0,
                 ratio_dn: float = 66.7,
                 ratio_up: float = 0.1,
                 knee: float = 0.1,
                 mode: str = 'peak',
                 upward_range_db: float = 36.0,
                 thresh_min: float = -40.0,
                 thresh_max: float = 0.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.ratio_dn = ratio_dn
        self.ratio_up = ratio_up
        self.knee = knee
        self.mode = mode
        self.upward_range_db = upward_range_db
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

    def forward(self, x: torch.Tensor, threshold: torch.Tensor, depth: torch.Tensor = torch.tensor(1.0)) -> torch.Tensor:
        batch_size, n_samples = x.shape
        device = x.device
        alpha_at = np.exp(-1.0 / (self.attack_ms * self.sample_rate / 1000.0))
        alpha_rel = np.exp(-1.0 / (self.release_ms * self.sample_rate / 1000.0))
        rect = torch.abs(x)
        if self.mode == 'rms': rect = x**2
        rect_db = 20 * torch.log10(rect + 1e-8)
        level_db = smooth_log_level(rect_db, float(alpha_at), float(alpha_rel))
        thresh_db = self.thresh_min + threshold * (self.thresh_max - self.thresh_min)
        diff_dn = level_db - thresh_db
        gain_dn_db = torch.where(diff_dn > -self.knee, 
                                 -(1.0 - 1.0/self.ratio_dn) * 0.5 * (diff_dn + self.knee + torch.sqrt((diff_dn - self.knee)**2 + 1e-4)), 
                                 torch.zeros_like(diff_dn))
        diff_up = thresh_db - level_db
        raw_gain_up_db = torch.where(diff_up > -self.knee,
                                     (1 - self.ratio_up) * 0.5 * (diff_up + self.knee + torch.sqrt((diff_up - self.knee)**2 + 1e-4)),
                                     torch.zeros_like(diff_up))
        gain_up_db = torch.clamp(raw_gain_up_db, max=self.upward_range_db)
        gain_db = (gain_dn_db + gain_up_db) * depth
        gain = 10 ** (gain_db / 20)
        return x * gain

class SoftSaturator(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor, drive_db: float = 0.0) -> torch.Tensor:
        drive = 10 ** (drive_db / 20)
        return torch.tanh(x * drive) / drive

class OTTProxy(nn.Module):
    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        # Use tuned frequencies: 60Hz and 4500Hz
        self.crossover = DifferentiableCrossover3Band(low_hz=60.0, high_hz=4500.0, sample_rate=sample_rate)
        self.comp_h = DifferentiableCompressor(sample_rate=sample_rate, attack_ms=118.8, release_ms=132.0, ratio_dn=4.0, ratio_up=0.1, knee=0.01, mode='rms', upward_range_db=18.0)
        self.comp_m = DifferentiableCompressor(sample_rate=sample_rate, attack_ms=197.1, release_ms=282.0, ratio_dn=4.0, ratio_up=0.1, knee=0.01, mode='rms', upward_range_db=18.0)
        self.comp_l = DifferentiableCompressor(sample_rate=sample_rate, attack_ms=420.6, release_ms=282.0, ratio_dn=100.0, ratio_up=0.1, knee=0.01, mode='peak', upward_range_db=18.0)
    
    def forward(self, x, depth, thresh_l, thresh_m, thresh_h, gain_l, gain_m, gain_h):
        low_dry, mid_dry, high_dry = self.crossover(x)
        intensity = depth
        low_comp = self.comp_l(low_dry, thresh_l, intensity)
        mid_comp = self.comp_m(mid_dry, thresh_m, intensity)
        high_comp = self.comp_h(high_dry, thresh_h, intensity)
        def map_gain(g): return 10 ** ((-12 + g * 24) / 20)
        low_wet = low_comp * map_gain(gain_l) * (10 ** (-6.8 / 20))
        mid_wet = mid_comp * map_gain(gain_m) * (10 ** (-1.1 / 20))
        high_wet = high_comp * map_gain(gain_h) * (10 ** (-2.3 / 20))
        wet = low_wet + mid_wet + high_wet
        
        # Parallel Mix (The "Amount" knob)
        return (1.0 - depth) * x + depth * wet
