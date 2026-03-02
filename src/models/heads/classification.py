import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    Categorical Head for predicting discrete, multi-class parameters.
    Given a latent vector (e.g., from the AudioEncoder), this head predicts the 
    logits for a specific categorical knob (e.g., Filter Type, Oscillator Waveform).
    
    The output is raw logits, which should be paired with nn.CrossEntropyLoss
    during training.
    """
    def __init__(self, in_features=256, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Input: (batch, in_features)
        Returns:
            logits: (batch, num_classes) - Unnormalized class scores
        """
        return self.net(x)
