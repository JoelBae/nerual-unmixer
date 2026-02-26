import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    Uses a small Multi-Layer Perceptron (MLP) to learn how to scale (gamma) 
    and shift (beta) audio representations based on external conditioning parameters (e.g., Ableton dials).
    """
    def __init__(self, condition_dim, num_features):
        super().__init__()
        # condition_dim: How many dials the effect has (e.g. 16)
        # num_features: How many audio channels the TCN is currently processing (e.g. 32)
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, num_features * 4),
            nn.PReLU(),
            nn.Linear(num_features * 4, num_features * 2)
        )

    def forward(self, x, condition):
        # x shape: (batch_size, num_features, time)
        # condition shape: (batch_size, condition_dim)
        film_params = self.mlp(condition)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        return (gamma * x) + beta


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network (TCN) Block.
    Main module to approximate the effect's audio output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, condition_dim):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.activation = nn.PReLU()
        self.film = FiLMLayer(condition_dim, out_channels)
        
        # The Bypass Lane (Residual Connection)
        if in_channels != out_channels:
            self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()
            
    def forward(self, x, condition):
        residual = self.res_conv(x)
        x = self.conv(x)
        x = self.activation(x)
        x = self.film(x, condition)
        return x + residual

class ConditionalTCN(nn.Module):
    """
    A Temporal Convolutional Network that processes audio and is conditioned 
    on external parameters (like an Ableton effect device).
    Configured for processing Stereo (2-channel) audio.
    """
    def __init__(self, num_params, num_channels=32, num_blocks=10, kernel_size=15):
        super().__init__()
        
        # 1. Expand the Stereo audio (2 channels) into `num_channels` hidden feature maps
        self.input_layer = nn.Conv1d(2, num_channels, kernel_size=1)
        
        # 2. Build the TCN blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i
            self.blocks.append(
                TCNBlock(
                    in_channels=num_channels, 
                    out_channels=num_channels, 
                    kernel_size=kernel_size, 
                    dilation=dilation, 
                    condition_dim=num_params
                )
            )
            
        # 3. Compress the hidden feature maps back down to a Stereo wave (2 channels)
        self.output_layer = nn.Conv1d(num_channels, 2, kernel_size=1)
        
    def forward(self, audio, params):
        # Audio input might be (batch_size, num_samples) or (batch_size, channels, num_samples)
        # We need to make sure it has the channel dimension.
        if audio.dim() == 2:
            x = audio.unsqueeze(1)
            # If it was actually mono, duplicate it to stereo so the network doesn't crash
            if x.shape[1] == 1:
               x = x.repeat(1, 2, 1) 
        else:
            x = audio
            
        # Push through the network
        x = self.input_layer(x)
        
        for block in self.blocks:
            x = block(x, params)
            
        out_audio = self.output_layer(x)
        
        return out_audio
