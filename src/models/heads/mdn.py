import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MDNHead(nn.Module):
    """
    Mixture Density Network (MDN) Head for predicting continuous audio parameters.
    Given a latent vector (e.g., from the AudioEncoder), this head predicts the 
    parameters (pi, mu, sigma) of a Gaussian Mixture Model for each continuous knob.
    
    This handles the "multi-modal" problem where 2 different synth patches might 
    sound identical, causing a standard MSE regression to predict the "average" 
    setting (which sounds wrong). The MDN instead predicts multiple possible 
    settings with attached confidences (pi).
    """
    def __init__(self, in_features=256, out_features=1, num_gaussians=5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        
        # A small hidden layer to add capacity before splitting to the 3 Gaussian heads
        self.hidden = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.GELU()
        )
        
        # We need num_gaussians for EACH output feature.
        # pi: mixing weights (must sum to 1 per feature)
        self.pi_layer = nn.Linear(128, out_features * num_gaussians)
        # mu: means of the Gaussians
        self.mu_layer = nn.Linear(128, out_features * num_gaussians)
        # sigma: standard deviations of the Gaussians (must be positive)
        self.sigma_layer = nn.Linear(128, out_features * num_gaussians)

    def forward(self, x):
        """
        Input: (batch, in_features)
        Returns:
            pi: (batch, out_features, num_gaussians) - Softmaxed probabilities
            mu: (batch, out_features, num_gaussians) - Means
            sigma: (batch, out_features, num_gaussians) - Exponentiated positive std devs
        """
        batch_size = x.shape[0]
        h = self.hidden(x)
        
        # Shape: (batch, out_features, num_gaussians)
        pi = self.pi_layer(h).view(batch_size, self.out_features, self.num_gaussians)
        mu = self.mu_layer(h).view(batch_size, self.out_features, self.num_gaussians)
        sigma = self.sigma_layer(h).view(batch_size, self.out_features, self.num_gaussians)
        
        # pi must be a probability distribution summing to 1 for each feature
        pi = F.softmax(pi, dim=2)
        
        # sigma must be strictly positive (use ELU + epsilon for stability)
        sigma = F.elu(sigma) + 1.0 + 1e-6
        
        # mu can be raw values, but since our parameters are 0-1, we could sigmoid it.
        # However, it's safer to let the model learn the raw logits and apply bounded sampling later,
        # or we can apply sigmoid here to strongly enforce the 0-1 domain.
        mu = torch.sigmoid(mu)
        
        return pi, mu, sigma

    def sample(self, pi, mu, sigma):
        """
        Given the MDN outputs, samples a specific predicted value for each feature.
        Inference strategy: Pick the mean of the Gaussian with the highest probability (pi).
        """
        batch_size, num_features, _ = pi.shape
        
        # Find the index of the most probable Gaussian for each feature
        # max_indices: (batch, num_features)
        _, max_indices = torch.max(pi, dim=2)
        
        # Gather the mu values corresponding to the highest probability component
        # We need to expand max_indices to (batch, num_features, 1) to match mu shape 
        # for gather
        max_indices = max_indices.unsqueeze(-1)
        
        # selected_mu: (batch, num_features, 1) -> (batch, num_features)
        selected_mu = torch.gather(mu, 2, max_indices).squeeze(-1)
        
        return selected_mu

def mdn_loss(pi, mu, sigma, target):
    """
    Negative Log-Likelihood loss for the Mixture Density Network.
    
    Args:
        pi, mu, sigma: (batch, out_features, num_gaussians)
        target: (batch, out_features) - The true parameter values
    """
    # Expand target to compare against all gaussians: (batch, out_features, 1)
    target = target.unsqueeze(-1)
    
    # Calculate Gaussian probability density function (PDF)
    # N(x | mu, sigma) = (1 / (sigma * sqrt(2pi))) * exp(-0.5 * ((x - mu) / sigma)^2)
    var = sigma ** 2
    # Add a small epsilon to variance to prevent division by zero
    diff = target - mu
    exp_term = torch.exp(-0.5 * (diff ** 2) / (var + 1e-8))
    
    # pdf: (batch, out_features, num_gaussians)
    pdf = (1.0 / (sigma * math.sqrt(2 * math.pi))) * exp_term
    
    # Multiply by mixing weights (pi) and sum across gaussians
    # weighted_pdf: (batch, out_features)
    weighted_pdf = torch.sum(pi * pdf, dim=2)
    
    # Negative log likelihood (add epsilon to prevent log(0))
    nll = -torch.log(weighted_pdf + 1e-8)
    
    # Average across batch and features
    return torch.mean(nll)
