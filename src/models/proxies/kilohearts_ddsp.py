import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict

class KHDistortionDSP(nn.Module):
    """
    Differentiable Kilohearts Distortion DSP core.
    Implements multiple differentiable waveshaping types.
    """
    TYPES = ["Soft Clip", "Hard Clip", "Foldback", "Sine", "Saturate"] # Simplified list for PoC

    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(self, 
                x: torch.Tensor, 
                drive: torch.Tensor, 
                bias: torch.Tensor, 
                mix: torch.Tensor,
                type_idx: int) -> torch.Tensor:
        """
        Single-type forward pass for a specific manifold.
        """
        # 1. Map parameters
        drive_gain = torch.pow(10.0, drive * 2.4) 
        dc_offset = (bias * 2.0) - 1.0
        
        # 2. Add Bias and Apply Drive
        processed = (x + dc_offset) * drive_gain
        
        # 3. Waveshaping
        if type_idx == 0: # Soft Clip
            distorted = torch.tanh(processed)
        elif type_idx == 1: # Hard Clip
            distorted = torch.clamp(processed, -1.0, 1.0)
        elif type_idx == 2: # Foldback
            distorted = torch.abs(torch.remainder(processed + 1.0, 4.0) - 2.0) - 1.0
        elif type_idx == 3: # Sine
            distorted = torch.sin(processed * (np.pi / 2))
        else: # Saturate
            distorted = processed / (1.0 + torch.abs(processed))
        
        # 4. DC Filter
        distorted = self._high_pass_filter(distorted, cutoff_hz=20.0)
        
        # 5. Dry/Wet Mix
        output = (1.0 - mix) * x + mix * distorted
        
        return output

    def _high_pass_filter(self, x: torch.Tensor, cutoff_hz: float) -> torch.Tensor:
        batch_size, n_samples = x.shape
        device = x.device
        alpha = np.exp(-2.0 * np.pi * cutoff_hz / self.sample_rate)
        
        last_x = torch.zeros(batch_size, device=device)
        last_y = torch.zeros(batch_size, device=device)
        y = torch.zeros_like(x)
        
        for i in range(n_samples):
            curr_x = x[:, i]
            y[:, i] = curr_x - last_x + alpha * last_y
            last_x = curr_x
            last_y = y[:, i]
            
        return y

class KHDistortionProxy(nn.Module):
    """
    Smart Proxy for Kilohearts Distortion using Gated Multi-Head Regression.
    """
    def __init__(self, latent_dim: int = 128, sample_rate: int = 44100):
        super().__init__()
        self.dsp = KHDistortionDSP(sample_rate)
        self.latent_dim = latent_dim
        self.num_types = len(KHDistortionDSP.TYPES)
        
        # Classification Head: Predicts Distortion Type
        self.classification_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_types)
        )
        
        # Gated Multi-Head Regression: One head per type
        # Each head predicts [Drive, Bias, Mix]
        self.regression_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 3),
                nn.Sigmoid()
            ) for _ in range(self.num_types)
        ])

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Takes input audio and a latent vector to reconstruct audio.
        Uses Gated Multi-Head output to preserve manifold integrity.
        """
        # 1. Classification
        type_logits = self.classification_head(latent)
        type_probs = F.softmax(type_logits, dim=-1) # (batch, num_types)
        
        # 2. Multi-Head Rendering
        # We render the audio for each type and then weight the audio outputs.
        # This is the most stable approach for non-smooth types.
        final_output = torch.zeros_like(x)
        
        for i in range(self.num_types):
            # Predict parameters for this specific type
            params = self.regression_heads[i](latent)
            drive, bias, mix = params[:, 0:1], params[:, 1:2], params[:, 2:3]
            
            # Render audio for this type
            type_audio = self.dsp(x, drive, bias, mix, type_idx=i)
            
            # Weighted sum of audio outputs (Gating)
            # type_probs is (batch, num_types), type_audio is (batch, samples)
            final_output += type_probs[:, i:i+1] * type_audio
            
        return final_output

class KHReverbDSP(nn.Module):
    """
    Differentiable Kilohearts Reverb DSP core.
    Uses a Feedback Delay Network (FDN) for late reflections and 
    a Tapped Delay Line for early reflections.
    """
    def __init__(self, sample_rate: int = 44100, num_delays: int = 8):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_delays = num_delays
        
        # Base delay times in samples (prime numbers or uncorrelated)
        # These are scaled by the 'size' parameter.
        base_delays = torch.tensor([487, 601, 719, 853, 1013, 1153, 1297, 1453], dtype=torch.float32)
        self.register_buffer("base_delays", base_delays)
        
        # Fixed Hadamard-like orthogonal matrix for mixing
        # We use a Householder reflection as a simple orthogonal matrix
        v = torch.ones(num_delays, 1)
        self.register_buffer("mixing_matrix", torch.eye(num_delays) - 2.0 * (v @ v.T) / num_delays)

    def forward(self, 
                x: torch.Tensor, 
                decay: torch.Tensor, 
                dampen: torch.Tensor, 
                size: torch.Tensor, 
                width: torch.Tensor, 
                early: torch.Tensor, 
                mix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, samples) - Mono input for simplicity, will output stereo
            decay, dampen, size, width, early, mix: (batch, 1) parameters [0, 1]
        """
        batch_size, n_samples = x.shape
        device = x.device
        
        # 1. Parameter Mapping
        # Decay: T60 from 0.1s to 10s
        t60 = 0.1 + decay * 9.9
        
        # Size: Scales base delays from 0.5x to 5x
        scale = 0.5 + size * 4.5
        delays = self.base_delays * scale
        
        # Dampen: LP cutoff from 500Hz to 20kHz
        cutoff_hz = 500.0 + (1.0 - dampen) * 19500.0
        alpha = torch.exp(-2.0 * np.pi * cutoff_hz / self.sample_rate)
        
        # Early reflections gain
        early_gain = early * 0.5
        
        # 2. Early Reflections (Simple Tapped Delay)
        # Render a few early taps for character
        early_audio = torch.zeros(batch_size, 2, n_samples, device=device)
        for i in range(4):
            tap_delay = int(100 * (i + 1) * scale[0].item()) # Approx
            if tap_delay < n_samples:
                tap_audio = torch.zeros_like(x)
                tap_audio[:, tap_delay:] = x[:, :-tap_delay]
                gain = early_gain * (0.8 ** i)
                early_audio[:, 0, :] += tap_audio * gain # Left
                early_audio[:, 1, :] += tap_audio * gain * (1.0 - (i % 2) * width) # Right
        
        # 3. Late Reflections (FDN)
        # This is a simplified differentiable FDN core
        # Note: True RNN-style FDN is slow in PyTorch. For PoC/PoA, we use 
        # a block-based approximation or a loop if samples are small.
        # Here we use a sample-by-sample loop for accuracy, but it's slow.
        states = torch.zeros(batch_size, self.num_delays, device=device)
        # g is (batch, num_delays)
        g = torch.pow(10.0, -3.0 * (delays / self.sample_rate) / t60)
        
        # Feedback filter states (one-pole LP)
        filt_states = torch.zeros(batch_size, self.num_delays, device=device)
        
        # Delay buffers (we need to handle fractional delays for 'size' differentiability)
        # For now, we'll use integer delays for stability in this PoC
        max_delay = int(delays.max().item() + 1)
        delay_buffers = torch.zeros(batch_size, self.num_delays, max_delay, device=device)
        ptr = 0
        
        late_audio = torch.zeros(batch_size, 2, n_samples, device=device)
        
        # Optimization: Block-based processing for FDN is complex. 
        # Using a loop for now.
        for i in range(n_samples):
            # Input to FDN: sum of mono input
            in_val = x[:, i:i+1] # (batch, 1)
            
            # Read from delay buffers
            # (batch, num_delays)
            delayed_vals = torch.zeros(batch_size, self.num_delays, device=device)
            for d_idx in range(self.num_delays):
                # We assume batch=1 for simplicity in this indexing or handle all batches
                # For differentiability with 'size', we'd need interpolation. 
                # For now, we take the mean delay across batch if batch > 1, 
                # but ideally we'd handle each batch's delay.
                # In this loop, we'll just use the first batch's delay for the pointer.
                d_samples = int(delays[0, d_idx].item())
                read_ptr = (ptr - d_samples) % max_delay
                delayed_vals[:, d_idx] = delay_buffers[:, d_idx, read_ptr]
            
            # Apply LP Dampening
            # y[n] = (1-alpha)*x[n] + alpha*y[n-1]
            filt_states = (1.0 - alpha) * delayed_vals + alpha * filt_states
            
            # Apply Feedback Gain
            feedback = filt_states * g # (batch, num_delays) * (batch, num_delays)
            
            # Mix via Orthogonal Matrix
            mixed_feedback = feedback @ self.mixing_matrix.T
            
            # Update Delay Buffers
            write_vals = in_val + mixed_feedback
            for d_idx in range(self.num_delays):
                delay_buffers[:, d_idx, ptr] = write_vals[:, d_idx]
                
            # Output: Sum of delay lines for L/R with width
            # Simple panning for now
            late_audio[:, 0, i] = torch.sum(filt_states[:, 0:self.num_delays//2], dim=1)
            late_audio[:, 1, i] = torch.sum(filt_states[:, self.num_delays//2:], dim=1)
            
            ptr = (ptr + 1) % max_delay

        # 4. Width and Stereo Spread
        # Combine L and R based on width
        mid = (late_audio[:, 0, :] + late_audio[:, 1, :]) * 0.5
        side = (late_audio[:, 0, :] - late_audio[:, 1, :]) * 0.5
        late_audio[:, 0, :] = mid + side * width
        late_audio[:, 1, :] = mid - side * width
        
        # 5. Final Mix
        # Proxy currently handles mono-in/stereo-out. If we need to match VST, 
        # we might need to handle stereo-in.
        wet = early_audio + late_audio
        
        final_l = (1.0 - mix) * x + mix * wet[:, 0, :]
        final_r = (1.0 - mix) * x + mix * wet[:, 1, :]
        
        return torch.stack([final_l, final_r], dim=1)

class KHReverbProxy(nn.Module):
    """
    Smart Proxy for Kilohearts Reverb.
    Predicts [Decay, Dampen, Size, Width, Early, Mix].
    """
    def __init__(self, latent_dim: int = 128, sample_rate: int = 44100):
        super().__init__()
        self.dsp = KHReverbDSP(sample_rate)
        self.latent_dim = latent_dim
        
        self.regression_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6), # 6 parameters
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Process audio with predicted parameters.
        """
        params = self.regression_head(latent)
        
        # Unpack parameters
        decay = params[:, 0:1]
        dampen = params[:, 1:2]
        size = params[:, 2:3]
        width = params[:, 3:4]
        early = params[:, 4:5]
        mix = params[:, 5:6]
        
        return self.dsp(x, decay, dampen, size, width, early, mix)

class KHEQ3BandDSP(nn.Module):
    """
    Differentiable Kilohearts 3-Band EQ DSP core.
    Simplified crossover-based implementation matching the real VST.
    
    Parameters:
    - low, mid, high: Gain [0, 1] 
    - low_split, high_split: Split frequencies [0, 1]
    """
    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.sample_rate = sample_rate

    def _lr4_resp(self, freq: torch.Tensor, n_fft: int, device: torch.device):
        """4th order Linkwitz-Riley frequency response."""
        # freq is (batch, 1)
        w = torch.linspace(0, np.pi, n_fft // 2 + 1, device=device)
        w0 = 2.0 * np.pi * freq / self.sample_rate # (batch, 1)
        
        # Normalize frequency: wn is (batch, freq_bins)
        wn = w.unsqueeze(0) / (w0 + 1e-8)
        
        # 2nd order Butterworth (squared for LR4)
        # H_bw(s) = 1 / (s^2 + sqrt(2)s + 1)
        den = (1.0 - wn**2) + 1j * np.sqrt(2) * wn
        H_lp_bw = 1.0 / (den + 1e-8)
        H_hp_bw = (wn**2) / (den + 1e-8)
        
        # LR4 is squared Butterworth
        return H_lp_bw**2, H_hp_bw**2

    def forward(self, 
                x: torch.Tensor, 
                low: torch.Tensor, 
                mid: torch.Tensor, 
                high: torch.Tensor, 
                low_split: torch.Tensor, 
                high_split: torch.Tensor) -> torch.Tensor:
        
        batch_size, n_samples = x.shape
        device = x.device
        
        # 1. Parameter Mapping
        # In KHS 3-Band EQ:
        # low/mid/high are gains. 0.5 is approx 0dB. 
        # Range is approx -inf to +12dB. 
        # We'll use a safer mapping for now.
        l_gain = torch.pow(10.0, (low * 3.0 - 1.5)) # 0.5 -> 1.0
        m_gain = torch.pow(10.0, (mid * 3.0 - 1.5))
        h_gain = torch.pow(10.0, (high * 3.0 - 1.5))
        
        # Splits map 20Hz - 20kHz
        ls_freq = 20.0 * torch.pow(1000.0, low_split)
        hs_freq = 20.0 * torch.pow(1000.0, high_split)
        
        # 2. FFT
        n_fft = 2 ** int(np.ceil(np.log2(n_samples)))
        X = torch.fft.rfft(x, n=n_fft)
        
        # 3. Frequency Responses (LR4)
        L_lp, L_hp = self._lr4_resp(ls_freq, n_fft, device)
        H_lp, H_hp = self._lr4_resp(hs_freq, n_fft, device)
        
        # Band responses:
        # Low: below low_split
        # Mid: above low_split AND below high_split
        # High: above high_split
        H_low_band = L_lp
        H_mid_band = L_hp * H_lp
        H_high_band = L_hp * H_hp
        
        # 4. Apply Gains and Sum
        # (batch, freq) * (batch, 1) -> (batch, freq)
        Y = X * (H_low_band * l_gain + H_mid_band * m_gain + H_high_band * h_gain)
        
        # 5. IFFT
        y = torch.fft.irfft(Y, n=n_fft)
        
        return y[:, :n_samples]

class KHEQ3BandProxy(nn.Module):
    """
    Smart Proxy for Kilohearts 3-Band EQ.
    Predicts [Low, Mid, High, LowSplit, HighSplit].
    """
    def __init__(self, latent_dim: int = 128, sample_rate: int = 44100):
        super().__init__()
        self.dsp = KHEQ3BandDSP(sample_rate)
        self.latent_dim = latent_dim
        
        self.regression_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5), # 5 parameters
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        params = self.regression_head(latent)
        
        # Unpack parameters
        low = params[:, 0:1]
        mid = params[:, 1:2]
        high = params[:, 2:3]
        low_split = params[:, 3:4]
        high_split = params[:, 4:5]
        
        return self.dsp(x, low, mid, high, low_split, high_split)
