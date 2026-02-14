import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class BaseProxyModule(nn.Module):
    """
    Base class for all differentiable effect proxies.
    Takes Audio + Parameters -> Output Audio.
    """
    def __init__(self, param_dim, input_channels=1, output_channels=1, num_channels=[16, 32, 32]):
        super().__init__()
        self.param_dim = param_dim
        
        # TCN Backbone to process audio
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size=3, stride=1, dilation=dilation_size, padding=(3-1) * dilation_size // 2)]
            
        self.audio_net = nn.Sequential(*layers)
        
        # Parameter Conditioning (FiLM-like or simple concatenation)
        # For simplicity v1: We project params and add to audio features
        self.param_proj = nn.Linear(param_dim, num_channels[-1])
        
        # Final output
        self.final_conv = nn.Conv1d(num_channels[-1], output_channels, kernel_size=1)

    def forward(self, audio, params):
        """
        audio: (batch, channels, length)
        params: (batch, param_dim)
        """
        # Process Audio
        feat = self.audio_net(audio) # (batch, ch, len)
        
        # Process Params
        p_emb = self.param_proj(params) # (batch, ch)
        p_emb = p_emb.unsqueeze(-1) # (batch, ch, 1)
        
        # Condition (Simple Addition)
        feat = feat + p_emb
        
        return self.final_conv(feat)
