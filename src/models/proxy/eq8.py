import torch
import torch.nn as nn
import math

class EQEightProxy(nn.Module):
    """
    Analytical DDSP Proxy for Ableton EQ Eight.
    
    Uses exact biquad filter math (Audio EQ Cookbook) to compute the frequency response.
    No training required — the EQ curve is calculated directly from parameters.
    Fully differentiable w.r.t. frequency, gain, and Q for end-to-end backpropagation.
    
    Supports all 8 bands with 8 filter types:
        0: Low Cut 48dB/oct  (4th order highpass — two cascaded biquads)
        1: Low Cut 12dB/oct  (2nd order highpass)
        2: Low Shelf
        3: Bell (Peaking EQ)
        4: Notch
        5: High Shelf
        6: High Cut 12dB/oct (2nd order lowpass)
        7: High Cut 48dB/oct (4th order lowpass — two cascaded biquads)
    
    Parameters per band: [filter_type, frequency_hz, gain_db, q]
    Total: 8 bands × 4 params = 32 parameters
    """
    
    def __init__(self, sr=44100, n_fft=2048, hop_length=512):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_bins = n_fft // 2 + 1  # 1025 for n_fft=2048
        self.num_params = 32            # 8 bands × 4 params
        
        # STFT window
        self.register_buffer('window', torch.hann_window(n_fft))
        
        # Precompute complex exponentials for frequency response evaluation
        # z^(-1) = e^(-jω) at each STFT bin frequency
        freqs = torch.linspace(0, sr / 2, self.num_bins)
        omega = 2.0 * math.pi * freqs / sr
        self.register_buffer('z_inv', torch.exp(-1j * omega))
        self.register_buffer('z_inv2', torch.exp(-2j * omega))
    
    def _biquad_coeffs(self, freq_hz, gain_db, q):
        """
        Compute biquad filter coefficients for ALL filter types (0-7).
        """
        # Armor: Ensure no NaNs hit the biquad math
        freq_hz = torch.nan_to_num(freq_hz, nan=1000.0)
        q = torch.nan_to_num(q, nan=0.707)
        gain_db = torch.nan_to_num(gain_db, nan=0.0)
        
        # Clamp to safe DSP ranges
        freq_hz = torch.clamp(freq_hz, 20.0, self.sr / 2 - 100)
        q = torch.clamp(q, 0.1, 18.0)
        gain_db = torch.clamp(gain_db, -24.0, 24.0)
        
        # Common intermediate values
        w0 = 2.0 * math.pi * freq_hz / self.sr
        cos_w0 = torch.cos(w0)
        sin_w0 = torch.sin(w0)
        alpha = sin_w0 / (2.0 * q)
        A = torch.pow(10.0, gain_db / 40.0)
        sqrt_A = torch.sqrt(torch.clamp(A, min=1e-6))
        
        batch = freq_hz.shape[0]
        
        # We'll build a (batch, 8 types, 6 coeffs) tensor
        coeffs = torch.zeros(batch, 8, 6, device=freq_hz.device)
        
        # 0 & 1: Highpass (Low Cut)
        coeffs[:, 0:2, 0] = (1 + cos_w0).unsqueeze(1) / 2 # b0
        coeffs[:, 0:2, 1] = -(1 + cos_w0).unsqueeze(1)   # b1
        coeffs[:, 0:2, 2] = (1 + cos_w0).unsqueeze(1) / 2 # b2
        coeffs[:, 0:2, 3] = (1 + alpha).unsqueeze(1)      # a0
        coeffs[:, 0:2, 4] = (-2 * cos_w0).unsqueeze(1)    # a1
        coeffs[:, 0:2, 5] = (1 - alpha).unsqueeze(1)      # a2
        
        # 2: Low Shelf
        coeffs[:, 2, 0] = A * ((A+1) - (A-1)*cos_w0 + 2*sqrt_A*alpha)
        coeffs[:, 2, 1] = 2*A * ((A-1) - (A+1)*cos_w0)
        coeffs[:, 2, 2] = A * ((A+1) - (A-1)*cos_w0 - 2*sqrt_A*alpha)
        coeffs[:, 2, 3] = (A+1) + (A-1)*cos_w0 + 2*sqrt_A*alpha
        coeffs[:, 2, 4] = -2 * ((A-1) + (A+1)*cos_w0)
        coeffs[:, 2, 5] = (A+1) + (A-1)*cos_w0 - 2*sqrt_A*alpha
        
        # 3: Bell
        coeffs[:, 3, 0] = 1 + alpha * A
        coeffs[:, 3, 1] = -2 * cos_w0
        coeffs[:, 3, 2] = 1 - alpha * A
        coeffs[:, 3, 3] = 1 + alpha / A
        coeffs[:, 3, 4] = -2 * cos_w0
        coeffs[:, 3, 5] = 1 - alpha / A
        
        # 4: Notch
        coeffs[:, 4, 0] = 1.0
        coeffs[:, 4, 1] = -2 * cos_w0
        coeffs[:, 4, 2] = 1.0
        coeffs[:, 4, 3] = 1 + alpha
        coeffs[:, 4, 4] = -2 * cos_w0
        coeffs[:, 4, 5] = 1 - alpha

        # 5: High Shelf
        coeffs[:, 5, 0] = A * ((A+1) + (A-1)*cos_w0 + 2*sqrt_A*alpha)
        coeffs[:, 5, 1] = -2*A * ((A-1) + (A+1)*cos_w0)
        coeffs[:, 5, 2] = A * ((A+1) + (A-1)*cos_w0 - 2*sqrt_A*alpha)
        coeffs[:, 5, 3] = (A+1) - (A-1)*cos_w0 + 2*sqrt_A*alpha
        coeffs[:, 5, 4] = 2 * ((A-1) - (A+1)*cos_w0)
        coeffs[:, 5, 5] = (A+1) - (A-1)*cos_w0 - 2*sqrt_A*alpha

        # 6 & 7: Lowpass (High Cut)
        coeffs[:, 6:8, 0] = (1 - cos_w0).unsqueeze(1) / 2
        coeffs[:, 6:8, 1] = (1 - cos_w0).unsqueeze(1)
        coeffs[:, 6:8, 2] = (1 - cos_w0).unsqueeze(1) / 2
        coeffs[:, 6:8, 3] = (1 + alpha).unsqueeze(1)
        coeffs[:, 6:8, 4] = (-2 * cos_w0).unsqueeze(1)
        coeffs[:, 6:8, 5] = (1 - alpha).unsqueeze(1)
        
        return coeffs
    
    def _freq_response(self, coeffs):
        """
        Compute the magnitude frequency response |H(e^jω)| at each STFT bin.
        coeffs: (batch, 8, 6)
        """
        batch = coeffs.shape[0]
        
        # coeffs are (batch, 8 types, 6 values)
        b0 = coeffs[:, :, 0].unsqueeze(-1) # (batch, 8, 1)
        b1 = coeffs[:, :, 1].unsqueeze(-1)
        b2 = coeffs[:, :, 2].unsqueeze(-1)
        a0 = coeffs[:, :, 3].unsqueeze(-1)
        a1 = coeffs[:, :, 4].unsqueeze(-1)
        a2 = coeffs[:, :, 5].unsqueeze(-1)
        
        z1 = self.z_inv.unsqueeze(0).unsqueeze(0)   # (1, 1, num_bins)
        z2 = self.z_inv2.unsqueeze(0).unsqueeze(0)  # (1, 1, num_bins)
        
        # H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (a0 + a1·z⁻¹ + a2·z⁻²)
        numer = b0 + b1 * z1 + b2 * z2
        denom = a0 + a1 * z1 + a2 * z2
        H = torch.abs(numer / (denom + 1e-8)) # (batch, 8, num_bins)
        
        # 48dB/oct = two cascaded identical biquads (Types 0 and 7)
        # Use out-of-place stacking to avoid gradient errors
        H_list = []
        for i in range(8):
            Hi = H[:, i, :]
            if i == 0 or i == 7:
                Hi = Hi * Hi
            H_list.append(Hi)
        
        return torch.stack(H_list, dim=1)

    def forward(self, audio, params):
        """
        Args:
            audio:  (batch, 2, time)  stereo audio
            params: (batch, 32)       8 bands × [type, freq_hz, gain_db, q]
                    Band 1 = params[:, 0:4], Band 2 = params[:, 4:8], ...
        Returns:
            out_audio: (batch, 2, time) filtered audio
        """
        batch, channels, time = audio.shape
        
        # 1. Compute the combined frequency response from all 8 bands
        H_combined = torch.ones(batch, self.num_bins, device=audio.device)
        
        # Determine how many bands we can actually process based on params size
        # Expected: 8 bands * 4 params = 32. 
        # But if we have fewer (like 10 in some datasets), we process what we have.
        num_available = params.shape[1]
        max_bands = min(8, num_available // 4)
        
        for band_idx in range(max_bands):
            i = band_idx * 4
            ftype = params[:, i]            # Float for differentiability
            freq  = params[:, i + 1]
            gain  = params[:, i + 2]
            q     = params[:, i + 3]
            
            # 1. Compute H for all 8 types
            coeffs_all = self._biquad_coeffs(freq, gain, q)
            H_all = self._freq_response(coeffs_all) # (batch, 8, num_bins)
            
            # 2. Soft-selection weights for filter types
            # Narrow softmax centered at ftype
            indices = torch.arange(8, device=audio.device).view(1, 8)
            weights = torch.softmax(-torch.abs(indices - ftype.unsqueeze(1)) / 0.1, dim=1)
            
            # 3. Blend H based on weights
            H_band = torch.sum(H_all * weights.unsqueeze(-1), dim=1) # (batch, num_bins)
            
            H_combined = H_combined * H_band
        
        # If there are trailing parameters that don't make a full band, ignore them.
        # This is common if the parameter vector includes other device settings.
        
        # 2. STFT
        audio_flat = audio.reshape(batch * channels, time)
        stft_complex = torch.stft(
            audio_flat, n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, return_complex=True
        )
        
        # 3. Apply the analytical EQ mask
        H_mask = H_combined.repeat_interleave(channels, dim=0).unsqueeze(-1)
        stft_filtered = stft_complex * H_mask
        
        # 4. ISTFT
        audio_out = torch.istft(
            stft_filtered, n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, length=time
        )
        
        return audio_out.reshape(batch, channels, time)
