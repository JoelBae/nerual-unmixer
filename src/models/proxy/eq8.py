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
    
    def _biquad_coeffs(self, filter_type, freq_hz, gain_db, q):
        """
        Compute biquad filter coefficients using the Audio EQ Cookbook.
        
        Args:
            filter_type: (batch,) int tensor — which filter type (0-7)
            freq_hz:     (batch,) float — center/corner frequency in Hz
            gain_db:     (batch,) float — gain in dB (for shelf/bell only)
            q:           (batch,) float — Q factor
            
        Returns:
            b0, b1, b2, a0, a1, a2: each (batch,) float
        """
        # Clamp to safe DSP ranges
        freq_hz = torch.clamp(freq_hz, 20.0, self.sr / 2 - 100)
        q = torch.clamp(q, 0.1, 18.0)
        gain_db = torch.clamp(gain_db, -24.0, 24.0)
        
        # Common intermediate values
        w0 = 2.0 * math.pi * freq_hz / self.sr
        cos_w0 = torch.cos(w0)
        sin_w0 = torch.sin(w0)
        alpha = sin_w0 / (2.0 * q)
        A = torch.pow(10.0, gain_db / 40.0)     # sqrt(linear gain)
        sqrt_A = torch.sqrt(torch.clamp(A, min=1e-6))
        
        # Default: unity pass-through (b0=1, a0=1, rest=0)
        b0 = torch.ones_like(freq_hz)
        b1 = torch.zeros_like(freq_hz)
        b2 = torch.zeros_like(freq_hz)
        a0 = torch.ones_like(freq_hz)
        a1 = torch.zeros_like(freq_hz)
        a2 = torch.zeros_like(freq_hz)
        
        # ---- Highpass (Low Cut types 0 and 1) ----
        mask = (filter_type == 0) | (filter_type == 1)
        b0 = torch.where(mask, (1 + cos_w0) / 2, b0)
        b1 = torch.where(mask, -(1 + cos_w0), b1)
        b2 = torch.where(mask, (1 + cos_w0) / 2, b2)
        a0 = torch.where(mask, 1 + alpha, a0)
        a1 = torch.where(mask, -2 * cos_w0, a1)
        a2 = torch.where(mask, 1 - alpha, a2)
        
        # ---- Lowpass (High Cut types 6 and 7) ----
        mask = (filter_type == 6) | (filter_type == 7)
        b0 = torch.where(mask, (1 - cos_w0) / 2, b0)
        b1 = torch.where(mask, 1 - cos_w0, b1)
        b2 = torch.where(mask, (1 - cos_w0) / 2, b2)
        a0 = torch.where(mask, 1 + alpha, a0)
        a1 = torch.where(mask, -2 * cos_w0, a1)
        a2 = torch.where(mask, 1 - alpha, a2)
        
        # ---- Low Shelf (type 2) ----
        mask = (filter_type == 2)
        b0 = torch.where(mask, A * ((A+1) - (A-1)*cos_w0 + 2*sqrt_A*alpha), b0)
        b1 = torch.where(mask, 2*A * ((A-1) - (A+1)*cos_w0), b1)
        b2 = torch.where(mask, A * ((A+1) - (A-1)*cos_w0 - 2*sqrt_A*alpha), b2)
        a0 = torch.where(mask, (A+1) + (A-1)*cos_w0 + 2*sqrt_A*alpha, a0)
        a1 = torch.where(mask, -2 * ((A-1) + (A+1)*cos_w0), a1)
        a2 = torch.where(mask, (A+1) + (A-1)*cos_w0 - 2*sqrt_A*alpha, a2)
        
        # ---- Bell / Peaking EQ (type 3) ----
        mask = (filter_type == 3)
        b0 = torch.where(mask, 1 + alpha * A, b0)
        b1 = torch.where(mask, -2 * cos_w0, b1)
        b2 = torch.where(mask, 1 - alpha * A, b2)
        a0 = torch.where(mask, 1 + alpha / A, a0)
        a1 = torch.where(mask, -2 * cos_w0, a1)
        a2 = torch.where(mask, 1 - alpha / A, a2)
        
        # ---- Notch (type 4) ----
        mask = (filter_type == 4)
        b0 = torch.where(mask, torch.ones_like(freq_hz), b0)
        b1 = torch.where(mask, -2 * cos_w0, b1)
        b2 = torch.where(mask, torch.ones_like(freq_hz), b2)
        a0 = torch.where(mask, 1 + alpha, a0)
        a1 = torch.where(mask, -2 * cos_w0, a1)
        a2 = torch.where(mask, 1 - alpha, a2)
        
        # ---- High Shelf (type 5) ----
        mask = (filter_type == 5)
        b0 = torch.where(mask, A * ((A+1) + (A-1)*cos_w0 + 2*sqrt_A*alpha), b0)
        b1 = torch.where(mask, -2*A * ((A-1) + (A+1)*cos_w0), b1)
        b2 = torch.where(mask, A * ((A+1) + (A-1)*cos_w0 - 2*sqrt_A*alpha), b2)
        a0 = torch.where(mask, (A+1) - (A-1)*cos_w0 + 2*sqrt_A*alpha, a0)
        a1 = torch.where(mask, 2 * ((A-1) - (A+1)*cos_w0), a1)
        a2 = torch.where(mask, (A+1) - (A-1)*cos_w0 - 2*sqrt_A*alpha, a2)
        
        return b0, b1, b2, a0, a1, a2
    
    def _freq_response(self, b0, b1, b2, a0, a1, a2, is_48db):
        """
        Compute the magnitude frequency response |H(e^jω)| at each STFT bin.
        
        Args:
            b0..a2: (batch,) biquad coefficients
            is_48db: (batch,) bool — if True, square the response (cascade two biquads)
            
        Returns:
            H: (batch, num_bins) real-valued magnitude response
        """
        # Expand: (batch,) -> (batch, 1) for broadcasting with (1, num_bins)
        b0, b1, b2 = b0.unsqueeze(1), b1.unsqueeze(1), b2.unsqueeze(1)
        a0, a1, a2 = a0.unsqueeze(1), a1.unsqueeze(1), a2.unsqueeze(1)
        
        z1 = self.z_inv.unsqueeze(0)   # (1, num_bins)
        z2 = self.z_inv2.unsqueeze(0)  # (1, num_bins)
        
        # H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (a0 + a1·z⁻¹ + a2·z⁻²)
        numer = b0 + b1 * z1 + b2 * z2
        denom = a0 + a1 * z1 + a2 * z2
        H = torch.abs(numer / (denom + 1e-8))
        
        # 48dB/oct = two cascaded identical biquads → square the response
        H = torch.where(is_48db.unsqueeze(1), H * H, H)
        
        return H

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
            ftype = params[:, i].long()
            freq  = params[:, i + 1]
            gain  = params[:, i + 2]
            q     = params[:, i + 3]
            
            b0, b1, b2, a0, a1, a2 = self._biquad_coeffs(ftype, freq, gain, q)
            is_48db = (ftype == 0) | (ftype == 7)
            H_band = self._freq_response(b0, b1, b2, a0, a1, a2, is_48db)
            
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
