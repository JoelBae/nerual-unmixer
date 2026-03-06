import torch
import torch.nn as nn

class AutoregressiveRoutingHead(nn.Module):
    """
    Sequence-to-sequence autoregressive decoder for effect routing.
    Takes the audio latent vector and generates a sequence of tokens.
    Tokens: 0=Saturator, 1=EQ8, 2=OTT, 3=Reverb, 4=EOS
    Max Length: 8
    """
    def __init__(self, latent_dim=256, num_tokens=5, max_len=8, hidden_dim=128):
        super().__init__()
        self.num_tokens = num_tokens
        self.max_len = max_len
        
        # +1 for the START token
        self.token_embedding = nn.Embedding(num_tokens + 1, hidden_dim) 
        
        # Audio Encoder's latent vector initializes the RNN hidden state
        self.rnn = nn.GRU(
            input_size=hidden_dim, 
            hidden_size=latent_dim, 
            batch_first=True
        )
        
        self.fc_out = nn.Linear(latent_dim, num_tokens)
        
        # We start sequence with a special START token index
        self.start_token_idx = num_tokens
        
    def forward(self, latent_context, target_sequence=None):
        """
        latent_context: (Batch, latent_dim) - from AudioEncoder
        target_sequence: (Batch, max_len) - Optional true tokens for Teacher Forcing
        Returns:
            logits: (Batch, max_len, num_tokens)
        """
        batch_size = latent_context.shape[0]
        device = latent_context.device
        
        # Initialize hidden state: (num_layers=1, batch, hidden_size=latent_dim)
        hidden = latent_context.unsqueeze(0).contiguous()
        
        # Initial input token is the START token
        current_token = torch.full((batch_size, 1), self.start_token_idx, dtype=torch.long, device=device)
        
        logits = []
        
        for step in range(self.max_len):
            # Embed current token -> (Batch, 1, hidden_dim)
            embedded = self.token_embedding(current_token) 
            
            # Step RNN -> output: (Batch, 1, latent_dim)
            output, hidden = self.rnn(embedded, hidden) 
            
            # Predict next token -> (Batch, num_tokens)
            step_logits = self.fc_out(output.squeeze(1)) 
            logits.append(step_logits)
            
            # Autoregressive feedback
            if target_sequence is not None:
                # Teacher forcing: use true target token for next step
                current_token = target_sequence[:, step].unsqueeze(1)
            else:
                # Free-running: use predicted token (argmax)
                next_token = torch.argmax(step_logits, dim=-1)
                current_token = next_token.unsqueeze(1)
                
        return torch.stack(logits, dim=1) # (Batch, max_len, num_tokens)
