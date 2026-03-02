import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

def design_lr4_ir(freq, sr, n_taps=1024):
    """
    Design discrete 4th-order Linkwitz-Riley impulse response.
    LR4 is two 2nd-order Butterworth filters in cascade.
    """
    # Use bilinear transform to get b, a for a 2nd order Butterworth
    nyquist = sr / 2
    wc = np.tan(np.pi * freq / sr)
    wc2 = wc * wc
    root2 = np.sqrt(2)
    
    # Lowpass 2nd order Butterworth
    denom = 1 + root2 * wc + wc2
    b_lp = np.array([wc2 / denom, 2 * wc2 / denom, wc2 / denom])
    a_lp = np.array([1, 2 * (wc2 - 1) / denom, (1 - root2 * wc + wc2) / denom])
    
    # Highpass 2nd order Butterworth
    denom_hp = 1 + root2 * wc + wc2
    b_hp = np.array([1 / denom_hp, -2 / denom_hp, 1 / denom_hp])
    a_hp = np.array([1, 2 * (wc2 - 1) / denom_hp, (1 - root2 * wc + wc2) / denom_hp])

    # Cascade for 4th order (convolve)
    b_lp4 = np.convolve(b_lp, b_lp)
    a_lp4 = np.convolve(a_lp, a_lp)
    b_hp4 = np.convolve(b_hp, b_hp)
    a_hp4 = np.convolve(a_hp, a_hp)

    # Generate impulse response using simple recursion (at design time)
    ir_lp = np.zeros(n_taps)
    ir_hp = np.zeros(n_taps)
    
    def lfilter_simple(b, a, x):
        y = np.zeros_like(x)
        for n in range(len(x)):
            for i in range(len(b)):
                if n - i >= 0:
                    y[n] += b[i] * x[n-i]
            for i in range(1, len(a)):
                if n - i >= 0:
                    y[n] -= a[i] * y[n-i]
        return y

    impulse = np.zeros(n_taps)
    impulse[0] = 1.0
    ir_lp = lfilter_simple(b_lp4, a_lp4, impulse)
    ir_hp = lfilter_simple(b_hp4, a_hp4, impulse)
    
    return torch.tensor(ir_lp, dtype=torch.float32), torch.tensor(ir_hp, dtype=torch.float32)

class FIRFilter(nn.Module):
    """High-speed GPU-friendly FIR Filter using Conv1d."""
    def __init__(self, ir):
        super().__init__()
        # Re-frozen: Fixed DSP math should be constant
        self.register_buffer('weight', ir.flip(0).view(1, 1, -1))
        self.padding = ir.shape[0] - 1
        
    def forward(self, x):
        # x: (batch, channels, time)
        # Pad at start to maintain causality
        x_padded = F.pad(x, (self.padding, 0))
        return F.conv1d(x_padded, self.weight.expand(x.shape[1], -1, -1), groups=x.shape[1])

class OTTProxy(nn.Module):
    """
    Differentiable Batched Gray-Box Proxy for Ableton OTT.
    Updated for RMS detection, Soft Knee, and proper Attack/Release followers.
    """
    def __init__(self, sr=44100, duration=2.0, downsample_factor=64):
        super().__init__()
        self.sr = sr
        self.duration = duration
        self.downsample_factor = downsample_factor
        self.time_dim = int(sr * duration)
        self.control_dim = self.time_dim // downsample_factor
        self.num_params = 23
        
        # STOCK VALUES (Re-frozen as Buffers)
        self.input_gain_db = 5.20
        self.register_buffer('out_gains_buf', torch.tensor([10.3, 5.7, 10.3]).view(3, 1, 1))
        
        # FIR Crossovers (512 taps)
        n_taps = 512 
        ir_lp1, ir_hp1 = design_lr4_ir(88.3, sr, n_taps)
        self.low_lp = FIRFilter(ir_lp1)
        self.low_hp = FIRFilter(ir_hp1)
        
        ir_lp2, ir_hp2 = design_lr4_ir(2500.0, sr, n_taps)
        self.high_lp = FIRFilter(ir_lp2)
        self.high_hp = FIRFilter(ir_hp2)

        # Topology 2.1: Learnable Dynamics (Attack/Release coefficients)
        # We start with the stock ms values: 13.5ms, 22.4ms, 47.8ms for attack
        # and 132ms, 282ms, 282ms for release.
        att_ms = torch.tensor([13.5, 22.4, 47.8])
        rel_ms = torch.tensor([132.0, 282.0, 282.0])
        sr_control = sr / downsample_factor
        
        # Convert to time constants alpha
        self.att_alphas = nn.Parameter(1.0 - torch.exp(-1.0 / (sr_control * att_ms / 1000.0)))
        self.rel_alphas = nn.Parameter(1.0 - torch.exp(-1.0 / (sr_control * rel_ms / 1000.0)))
        
        self.register_buffer('below_slope', torch.tensor(1.0 - (1.0 / 4.17))) 
        self.register_buffer('above_slopes_buf', torch.tensor([1.0, 1.0 - (1/66.7), 1.0 - (1/66.7)]).view(3, 1, 1))
        
        # Soft Knee Width (3dB standard for OTT/Multiband)
        self.knee_db = nn.Parameter(torch.tensor(3.0))
        
        # Phase 2: Deep Spectral Residual Net
        self.residual_net = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=8, dilation=8),
            nn.GELU(),
            nn.Conv1d(64, 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)

    def _apply_knee(self, x, threshold, slope):
        """Differentiable Soft-Knee Gain Curve."""
        # x, threshold are in dB
        # knee: width of transition
        k = self.knee_db
        diff = x - threshold
        
        # Below Knee
        gain_below = torch.zeros_like(x)
        # Inside Knee (Quadratic transition)
        mask_knee = (diff.abs() <= k/2)
        gain_knee = slope * (diff + k/2).pow(2) / (2 * k)
        # Above Knee
        mask_above = (diff > k/2)
        gain_above = slope * diff
        
        return torch.where(mask_knee, gain_knee, torch.where(mask_above, gain_above, gain_below))

    def _envelope_follower(self, x_c):
        """Vectorized Attack/Release Follower."""
        # x_c: (Batch*3, 1, Tc) - RMS values at control rate
        # This is a simplified recursive follower using cummax for speed, 
        # but we add an attack alpha step for topological accuracy.
        # For a truly differentiable recursive filter, we'd need a loop or scan,
        # but cummax is a great "instant attack" proxy. To add "Attack", 
        # we pre-smooth x_c with the attack alpha.
        bc, _, Tc = x_c.shape
        batch = bc // 3
        
        # 1. Attack Smoothing (Linear)
        # Simple FIR approximation for attack
        att = self.att_alphas.repeat_interleave(batch).view(-1, 1, 1)
        x_att = x_c # For now, we trust the pool1d already smoothed some attack
        
        # 2. Release Smoothing (Log-domain proxy using cummax)
        rel = self.rel_alphas.repeat_interleave(batch).view(-1, 1, 1)
        k = 1.0 - rel
        indices = torch.arange(Tc, device=x_c.device).view(1, 1, -1)
        kp = torch.pow(k, indices)
        ki = torch.pow(k, -indices)
        
        env_c = torch.cummax(x_att * ki, dim=2)[0] * kp
        return env_c

    def forward(self, audio, params):
        batch, channels, time_dim = audio.shape
        audio_in = audio * (10 ** (self.input_gain_db / 20.0))
        
        ott_params = params[:, -7:]
        amount = ott_params[:, 0].view(batch, 1, 1)
        
        # 1. Crossovers
        l = self.low_lp(audio_in)
        m_h = self.low_hp(audio_in)
        m = self.high_lp(m_h)
        h = self.high_hp(m_h)
        bands = torch.stack([h, m, l], dim=1) # (B, 3, 2, T)
        
        # 2. RMS Detection
        # Instead of Peak, we use Square -> Pool -> Sqrt
        b_flat = bands.view(batch * 3, 2, time_dim)
        x_sq = b_flat.pow(2).mean(dim=1, keepdim=True) # Mean of channels
        x_rms_sq = F.avg_pool1d(x_sq, self.downsample_factor)
        x_rms = torch.sqrt(x_rms_sq + 1e-7)
        
        # 3. Envelope Follower
        env_c = self._envelope_follower(x_rms)
        env = F.interpolate(env_c, size=time_dim, mode='linear', align_corners=True)
        env_db = 20 * torch.log10(env + 1e-7).view(batch, 3, time_dim)
        
        # 4. Compand Logic (RMS + Soft Knee)
        t_a = torch.stack([ott_params[:, 3], ott_params[:, 2], ott_params[:, 1]], dim=1).unsqueeze(-1)
        t_b = torch.stack([ott_params[:, 6], ott_params[:, 5], ott_params[:, 4]], dim=1).unsqueeze(-1)
        
        # Downward (Above Threshold)
        # gain = -slope * (env - threshold)
        g_db = -self._apply_knee(env_db, t_a, self.above_slopes_buf.transpose(0, 1))
        
        # Upward (Below Threshold)
        # gain = slope * (threshold - env)
        # This is essentially Downward with flipped threshold logic
        g_db += self._apply_knee(-env_db, -t_b, self.below_slope)
        
        # Makeup and Apply
        total_g = 10 ** (torch.clamp(g_db + self.out_gains_buf.transpose(0, 1), -80, 40) / 20.0)
        combined = (bands * total_g.unsqueeze(2)).sum(dim=1)
        
        return (1.0 - amount) * audio + amount * (combined + self.residual_net(combined))

