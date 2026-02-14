import torch
import torch.nn as nn
import geoopt

class HyperbolicSequenceDecoder(nn.Module):
    """
    Decodes the latent embedding into a sequence of effect configurations 
    in Hyperbolic space (Poincaré ball).
    """
    def __init__(self, latent_dim, hidden_dim, max_seq_len=8, curvature=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Define the Manifold
        self.manifold = geoopt.PoincareBall(c=curvature)
        
        # GRU Cell - Standard Euclidean GRU for now, but we project outputs to Hyperbolic space
        # Alternatively, we could implement a Mobius GRU if needed.
        self.rnn = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, embedding):
        """
        embedding: (batch_size, latent_dim)
        """
        batch_size = embedding.size(0)
        
        # Initialize hidden state
        h = torch.zeros(1, batch_size, self.hidden_dim).to(embedding.device)
        
        # Input to RNN (repeat embedding for each step or use as initial state)
        # Here we just use embedding as input for simplicity, repeated
        inputs = embedding.unsqueeze(1).repeat(1, self.max_seq_len, 1)
        
        outputs, _ = self.rnn(inputs, h)
        
        # Project outputs to Hyperbolic Space
        # map Euclidean outputs to Poincare Ball
        flat_outputs = outputs.reshape(-1, self.hidden_dim)
        euclidean_proj = self.fc(flat_outputs)
        
        # Exponential map to move from Tangent space (Euclidean) to Manifold (Hyperbolic)
        # We treat the linear output as a vector in the tangent space at the origin
        hyperbolic_points = self.manifold.expmap0(euclidean_proj)
        
        return hyperbolic_points.view(batch_size, self.max_seq_len, -1)

    def dist(self, p1, p2):
        """
        Compute hyperbolic distance between two points on the manifold.
        """
        return self.manifold.dist(p1, p2)
