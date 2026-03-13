import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class AudioFeatureExtractor(nn.Module):
    """
    Differentiable audio feature extractor to convert raw audio into Mel-spectrograms.
    Used for technical manifold alignment (alpha_c, epsilon) as prescribed in neural_unmixer-4.pdf.
    """
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 64,
        f_min: float = 20.0,
        f_max: Optional[float] = None
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2
        
        # Initialize Mel Filterbank
        mel_fb = self._build_mel_filterbank()
        self.register_buffer("mel_fb", mel_fb)

    def _build_mel_filterbank(self) -> torch.Tensor:
        """Computes a simple Mel filterbank."""
        # Frequency bins
        freq_bins = torch.linspace(0, self.sample_rate / 2, self.n_fft // 2 + 1)
        
        # Mel scale conversion helpers
        def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700.0)
        def mel_to_hz(mel): return 700 * (10**(mel / 2595.0) - 1)
        
        # Linear mel points
        mel_min = hz_to_mel(self.f_min)
        mel_max = hz_to_mel(self.f_max)
        mel_points = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Filterbank matrix
        fb = torch.zeros((self.n_mels, self.n_fft // 2 + 1))
        for m in range(1, self.n_mels + 1):
            left = hz_points[m-1]
            center = hz_points[m]
            right = hz_points[m+1]
            
            # Simple triangular filters
            # Up-slope
            mask_up = (freq_bins >= left) & (freq_bins <= center)
            fb[m-1, mask_up] = (freq_bins[mask_up] - left) / (center - left)
            
            # Down-slope
            mask_down = (freq_bins >= center) & (freq_bins <= right)
            fb[m-1, mask_down] = (right - freq_bins[mask_down]) / (right - center)
            
        return fb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch, samples)
        Output: (batch, n_mels, time_frames) in log-power domain.
        """
        # 1. STFT
        # We use a Hann window for smoothing as is standard
        window = torch.hann_window(self.n_fft, device=x.device)
        stft = torch.stft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=window,
            return_complex=True,
            center=True
        )
        
        # 2. Magnitude Power
        power = torch.abs(stft)**2 # (batch, freq_bins, frames)
        
        # 3. Mel Filterbank
        mel_spec = torch.matmul(self.mel_fb, power) # (batch, n_mels, frames)
        
        # 4. Log Scaling
        # Add epsilon to avoid log(0)
        log_mel_spec = 10 * torch.log10(mel_spec + 1e-10)
        
        return log_mel_spec

_extractor_cache = {}

def get_mel_spectrogram(audio: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
    """Helper for non-torch code to get a Mel spectrogram (Optimized with caching)."""
    global _extractor_cache
    if sample_rate not in _extractor_cache:
        _extractor_cache[sample_rate] = AudioFeatureExtractor(sample_rate=sample_rate)
    
    extractor = _extractor_cache[sample_rate]
    x = torch.from_numpy(audio)
    if x.ndim == 1: x = x.unsqueeze(0)
    # Move to same device as extractor buffers
    x = x.to(next(extractor.buffers()).device)
    return extractor(x).detach().cpu().numpy()[0]
