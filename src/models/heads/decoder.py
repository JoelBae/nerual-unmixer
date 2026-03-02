import torch
import torch.nn as nn

class SequenceDecoder(nn.Module):
    """
    Hyperbolic Sequence Decoder (GRU) for predicting effect routing/chains.
    Given a semantic latent vector (from the AudioEncoder), this decoder predicts 
    a sequence of discrete tokens representing the chain/order of audio effects,
    such as [EQ -> Saturator -> Phaser].
    """
    def __init__(self, latent_dim=256, vocab_size=10, hidden_dim=256, max_seq_len=6):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        # Recurrent layer to decode the sequence
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        # Final projection to vocabulary logits
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        """
        Input: 
            x: (batch, latent_dim)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size = x.shape[0]
        
        # Initialize hidden state directly with the encoded latent vector
        # GRU hidden state requires shape: (num_layers=1, batch, hidden_dim)
        hidden = x.unsqueeze(0)
        
        # To step the sequence, we feed the latent vector as constant context
        # at each timestep. (Standard seq2seq pattern without teacher forcing).
        # shape: (batch, max_seq_len, latent_dim)
        gru_input = x.unsqueeze(1).expand(-1, self.max_seq_len, -1)
        
        # Output contains the hidden states for each timestep
        output, _ = self.gru(gru_input, hidden)
        
        # Project states to effect vocabulary logits
        logits = self.fc_out(output) # (batch, seq_len, vocab_size)
        
        return logits
