import torch
import torch.nn as nn
import torchaudio

class AudioEncoder(nn.Module):
    """
    Shared Audio Encoder (Feature Extractor).
    Takes raw audio or spectrogram input and produces a latent embedding.
    """
    def __init__(self, input_channels=1, base_channels=32, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Conv1d(input_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Layer 2
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Layer 3
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(base_channels * 4, latent_dim)

    def forward(self, x):
        """
        x: (batch_size, channels, length)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1) # Add channel dim if missing
            
        features = self.net(x)
        features = features.view(features.size(0), -1)
        embedding = self.fc(features)
        return embedding
