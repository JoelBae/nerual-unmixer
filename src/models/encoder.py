import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    """
    A reusable 1D Convolutional block for the encoder.
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=4, padding=4):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class AudioEncoder(nn.Module):
    """
    The Core Encoder: Converts raw audio waveforms into a condensed latent vector.
    """
    def __init__(self, in_channels=1, latent_dim=256):
        super().__init__()
        
        # 5 Blocks with stride 4 will downsample 44100 -> 11025 -> 2756 -> 689 -> 172 -> 43
        self.encoder = nn.Sequential(
            ConvBlock1D(in_channels, 32),
            ConvBlock1D(32, 64),
            ConvBlock1D(64, 128),
            ConvBlock1D(128, 256),
            ConvBlock1D(256, 512)
        )
        
        # Project from 512 channels down to the precise latent dimension
        self.projection = nn.Linear(512, latent_dim)

    def forward(self, x):
        """
        Input: (batch_size, channels, time_steps)
        Output: (batch_size, latent_dim)
        """
        # 1. Pass through CNN encoder
        x = self.encoder(x) # Shape: (batch_size, 512, 43)
        
        # 2. Global Average Pooling (collapse the time dimension completely)
        x = torch.mean(x, dim=2) # Shape: (batch_size, 512)
        
        # 3. Project to latent space
        x = self.projection(x) # Shape: (batch_size, latent_dim)
        
        return x
