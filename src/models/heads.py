import torch
import torch.nn as nn
import torch.nn.functional as F

class MDNHead(nn.Module):
    """
    Predicts parameters for a Mixture Density Network (MDN).
    Outputs: pi (mixing coefficients), sigma (standard deviations), mu (means).
    """
    def __init__(self, input_dim, output_dim, num_mixtures=5, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_mixtures = num_mixtures
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Heads for MDN parameters
        self.z_pi = nn.Linear(hidden_dim, num_mixtures)
        self.z_sigma = nn.Linear(hidden_dim, num_mixtures * output_dim)
        self.z_mu = nn.Linear(hidden_dim, num_mixtures * output_dim)

    def forward(self, x):
        """
        Returns:
        pi: (batch_size, num_mixtures) - Logits for Softmax
        sigma: (batch_size, num_mixtures, output_dim) - Logits for Exp
        mu: (batch_size, num_mixtures, output_dim)
        """
        hidden = self.net(x)
        
        pi = self.z_pi(hidden)
        pi = F.softmax(pi, dim=-1) # Ensure they sum to 1
        
        sigma_raw = self.z_sigma(hidden)
        sigma = torch.exp(sigma_raw).view(-1, self.num_mixtures, self.output_dim) # Ensure positive
        
        mu = self.z_mu(hidden).view(-1, self.num_mixtures, self.output_dim)
        
        return pi, sigma, mu

class ClassificationHead(nn.Module):
    """
    Predicts categorical parameters (e.g., Wavetable index, Filter type).
    """
    def __init__(self, input_dim, num_classes, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
            # No Softmax here, using CrossEntropyLoss which expects logits
        )

    def forward(self, x):
        return self.net(x)
