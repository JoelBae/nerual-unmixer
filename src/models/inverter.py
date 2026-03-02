import torch
import torch.nn as nn
from src.models.encoder import AudioEncoder
from src.models.heads.mdn import MDNHead
from src.models.heads.classification import ClassificationHead
from src.utils.normalization import denormalize_params

class NeuralInverter(nn.Module):
    """
    The "Robot Ears": Takes raw audio and predicts the 62 parameters of the synth/effects chain.
    - AudioEncoder: Extracts a latent vector from raw audio.
    - MDNHead (Continuous): Predicts the 61 continuous knobs (62 total - 1 categorical).
    - ClassificationHead (Wave): Predicts the 'Osc-A Wave' index (0-127).
    - ClassificationHead (Order): Predicts the effect chain permutation (0-23).
    """
    def __init__(self, latent_dim=256, num_gaussians=5):
        super().__init__()
        self.encoder = AudioEncoder(in_channels=2, latent_dim=latent_dim) # Stereo input
        
        # 61 Continuous Knobs:
        # Operator (15), Saturator (4), EQ8 (32), OTT (7), Reverb (3)
        self.mdn_head = MDNHead(
            in_features=latent_dim, 
            out_features=61, 
            num_gaussians=num_gaussians
        )
        
        # 1 Categorical Knob: Osc-A Wave (Indices 0-127)
        self.wave_head = ClassificationHead(
            in_features=latent_dim, 
            num_classes=128
        )

        # Effect Order Permutation (4 effects! = 24 classes)
        # Saturator, EQ8, OTT, Reverb
        self.order_head = ClassificationHead(
            in_features=latent_dim,
            num_classes=24
        )

    def forward(self, audio):
        """
        Input: (batch, 2, time)
        Returns: 
            A dictionary containing pi, mu, sigma, wave_logits, and order_logits.
        """
        latent = self.encoder(audio)
        
        pi, mu, sigma = self.mdn_head(latent)
        wave_logits = self.wave_head(latent)
        order_logits = self.order_head(latent)
        
        return {
            'pi': pi,
            'mu': mu,
            'sigma': sigma,
            'wave_logits': wave_logits,
            'order_logits': order_logits
        }

    def predict(self, audio):
        """
        Inference mode: returns a dictionary with raw parameters and the effect sequence.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(audio)
            
            # 1. Sample from MDN (returns [batch, 61])
            continuous_params = self.mdn_head.sample(
                outputs['pi'], outputs['mu'], outputs['sigma']
            )
            
            # 2. Get highest probability waveform
            wave_indices = torch.argmax(outputs['wave_logits'], dim=1).float().unsqueeze(-1)
            
            # 3. Get highest probability order permutation
            order_idx = torch.argmax(outputs['order_logits'], dim=1)
            
            # 4. Assemble the 62 parameters
            # Map back to the [Operator (16), Saturator (4), EQ8 (32), OTT (7), Reverb (3)] layout
            results = torch.zeros(audio.shape[0], 62, device=audio.device)
            
            # --- Operator (Indices 0:16) ---
            results[:, 0] = continuous_params[:, 0] # Transpose
            results[:, 1] = wave_indices.squeeze(-1) # WAVE (Categorical)
            results[:, 2:16] = continuous_params[:, 1:15] # Filter, Envs...
            
            # --- Saturator (Indices 16:20) ---
            results[:, 16:20] = continuous_params[:, 15:19]
            
            # --- EQ Eight (Indices 20:52) ---
            results[:, 20:52] = continuous_params[:, 19:51]
            
            # --- OTT (Indices 52:59) ---
            results[:, 52:59] = continuous_params[:, 51:58]
            
            # --- Reverb (Indices 59:62) ---
            results[:, 59:62] = continuous_params[:, 58:61]
            
            # 5. Decode Order Index into device sequence
            # Possible effects: saturator, eq8, ott, reverb (4! = 24 permutations)
            import itertools
            effects = ['saturator', 'eq8', 'ott', 'reverb']
            perms = list(itertools.permutations(effects))
            
            # For simplicity in this method, we return the sequence for the FIRST item in the batch
            # In a real un-mixer, you'd likely want a list of sequences per batch item.
            predicted_sequence = ['operator'] + list(perms[order_idx[0]])
            
            # 6. Denormalize MDN outputs back to Ableton ranges
            raw_params = denormalize_params(results)
            
            return {
                'params': raw_params,
                'sequence': predicted_sequence
            }
