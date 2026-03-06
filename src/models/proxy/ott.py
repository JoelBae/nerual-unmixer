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

def design_one_pole_ir(time_ms, sr, n_taps):
    """
    Design impulse response for a one-pole filter.
    y[n] = alpha*x[n] + (1-alpha)*y[n-1]
    """
    alpha = 1.0 - np.exp(-1.0 / (sr * time_ms / 1000.0))
    n = np.arange(n_taps)
    ir = alpha * np.power(1.0 - alpha, n)
    return torch.tensor(ir, dtype=torch.float32)

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
        self.num_params = 7
        
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

        # Dynamics
        sr_control = sr / downsample_factor
        
        # Release times are fixed based on known stock values
        rel_ms = torch.tensor([132.0, 282.0, 282.0]) # H, M, L
        rel_alphas_tensor = 1.0 - torch.exp(-1.0 / (sr_control * rel_ms / 1000.0))
        self.register_buffer('rel_alphas', rel_alphas_tensor.view(1, 3, 1))

        # Attack times are fixed via FIR filters based on stock OTT values
        # The bands are processed in order [h, m, l] in the forward pass.
        attack_times_ms = [13.5, 22.4, 47.8] # High, Mid, Low
        attack_n_taps = 512
        attack_filters = []
        for att_ms in attack_times_ms:
            ir = design_one_pole_ir(att_ms, sr_control, attack_n_taps)
            attack_filters.append(FIRFilter(ir))
        self.attack_filters = nn.ModuleList(attack_filters)
        
        self.register_buffer('below_slope', torch.tensor(1.0 - (1.0 / 4.17))) 
        self.register_buffer('above_slopes_buf', torch.tensor([1.0, 1.0 - (1/66.7), 1.0 - (1/66.7)]).view(3, 1, 1))
        
        # Soft Knee Width (3dB standard for OTT/Multiband)
        self.knee_db = nn.Parameter(torch.tensor(3.0))
        
        # Phase 2: Deep Spectral Residual Net
        self.residual_net = nn.Sequential(
            nn.Conv1d(2, 128, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=8, dilation=8),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=16, dilation=16),
            nn.GELU(),
            nn.Conv1d(128, 2, kernel_size=3, padding=1),
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
        bc, _, Tc = x_c.shape
        batch = bc // 3
        
        # 1. Attack Smoothing (Explicit FIR)
        # Reshape from (B*3, 1, Tc) to (B, 3, Tc) to apply per-band filters
        x_c_reshaped = x_c.view(batch, 3, Tc)

        # Apply the specific FIR attack filter to each band.
        # The bands are in order [h, m, l].
        h_rms = x_c_reshaped[:, 0:1, :]
        m_rms = x_c_reshaped[:, 1:2, :]
        l_rms = x_c_reshaped[:, 2:3, :]

        h_att = self.attack_filters[0](h_rms)
        m_att = self.attack_filters[1](m_rms)
        l_att = self.attack_filters[2](l_rms)

        # Combine bands back and reshape to (B*3, 1, Tc) for release stage
        x_att_reshaped = torch.cat([h_att, m_att, l_att], dim=1)
        x_att = x_att_reshaped.view(batch * 3, 1, Tc)

        # 2. Release Smoothing (Log-domain proxy using cummax)
        # Broadcast alphas to match interleaved batch shape (B, 3, 1) -> (B*3, 1, 1)
        rel = self.rel_alphas.repeat(batch, 1, 1).view(-1, 1, 1)
        k = 1.0 - rel
        indices = torch.arange(Tc, device=x_c.device).view(1, 1, -1)
        kp = torch.pow(k, indices)
        ki = torch.pow(k, -indices)
        
        env_c = torch.cummax(x_att * ki, dim=2)[0] * kp
        return env_c

    def forward(self, audio, params):
        batch, channels, time_dim = audio.shape
        audio_in = audio * (10 ** (self.input_gain_db / 20.0))
        
        # Armor: Ensure params are safe
        params = torch.nan_to_num(params, nan=0.0)
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
        g_db = -self._apply_knee(env_db, t_a, self.above_slopes_buf.transpose(0, 1))
        
        # Upward (Below Threshold)
        g_db += self._apply_knee(-env_db, -t_b, self.below_slope)
        
        # --- Apply Amount parameter by scaling the gain reduction/expansion ---
        g_db_scaled = g_db * amount

        # Makeup and Apply
        total_g = 10 ** (torch.clamp(g_db_scaled + self.out_gains_buf.transpose(0, 1), -80, 40) / 20.0)
        combined = (bands * total_g.unsqueeze(2)).sum(dim=1)
        
        # The residual net corrects any spectral differences in the final combined audio
        return combined + self.residual_net(combined)

