import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class SpectralLoss(nn.Module):
    """
    Computes spectral loss using torch.stft directly for maximum speed on MPS.
    """
    def __init__(self, n_fft=1024, hop_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Precompute window
        self.register_buffer('window', torch.hann_window(n_fft))
        
    def forward(self, x, y):
        # x, y: (B, C, T)
        # Combine channels for STFT if stereo (faster)
        B, C, T = x.shape
        x_flat = x.reshape(-1, T)
        y_flat = y.reshape(-1, T)
        
        # 1. STFT
        x_stft = torch.stft(x_flat, n_fft=self.n_fft, hop_length=self.hop_length, 
                           window=self.window, return_complex=True)
        y_stft = torch.stft(y_flat, n_fft=self.n_fft, hop_length=self.hop_length, 
                           window=self.window, return_complex=True)
        
        x_mag = torch.abs(x_stft) + 1e-7
        y_mag = torch.abs(y_stft) + 1e-7
        
        # 2. Spectral Convergence
        loss_sc = torch.norm(y_mag - x_mag, p='fro') / torch.norm(y_mag, p='fro')
        
        # 3. Log-Magnitude
        loss_mag = F.l1_loss(torch.log(x_mag), torch.log(y_mag))
        
        return loss_sc + loss_mag

class VectorizedMultiScaleSpectralLoss(nn.Module):
    """
    Highly optimized multi-scale spectral loss.
    Pre-allocates windows and minimizes Python overhead.
    """
    def __init__(self, fft_sizes=[512, 1024, 2048]):
        super().__init__()
        self.fft_sizes = fft_sizes
        for n in fft_sizes:
            self.register_buffer(f'window_{n}', torch.hann_window(n))
            
    def forward(self, x, y):
        B, C, T = x.shape
        x_f = x.reshape(-1, T)
        y_f = y.reshape(-1, T)
        
        total_loss = 0.0
        for n in self.fft_sizes:
            win = getattr(self, f'window_{n}')
            hop = n // 4
            
            # STFT
            xs = torch.stft(x_f, n, hop, window=win, return_complex=True)
            ys = torch.stft(y_f, n, hop, window=win, return_complex=True)
            
            xm = torch.abs(xs) + 1e-7
            ym = torch.abs(ys) + 1e-7
            
            # SC + Mag
            sc = torch.norm(ym - xm, p='fro') / torch.norm(ym, p='fro')
            mag = F.l1_loss(torch.log(xm), torch.log(ym))
            total_loss += sc + mag
            
        return total_loss / len(self.fft_sizes)

class DynamicsLoss(nn.Module):
    """
    Optimized Dynamics Loss with Vectorized Spectral component.
    """
    def __init__(self, fft_sizes=[512, 1024, 2048]):
        super().__init__()
        self.spectral = VectorizedMultiScaleSpectralLoss(fft_sizes)
        
    def forward(self, x, y):
        loss_spec = self.spectral(x, y)
        
        # Fast Envelope Loss (Downsampled)
        # Using a larger stride to speed up CPU/GPU sync
        env_x = F.avg_pool1d(x.abs(), kernel_size=512, stride=256)
        env_y = F.avg_pool1d(y.abs(), kernel_size=512, stride=256)
        loss_env = F.l1_loss(env_x, env_y) * 5.0
        
        return loss_spec + loss_env
