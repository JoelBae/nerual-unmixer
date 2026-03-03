import torch
import torch.nn as nn
from src.models.encoder import AudioEncoder
from src.models.heads.mdn import MDNHead
from src.models.heads.classification import ClassificationHead
from src.utils.normalization import denormalize_params

class NeuralInverter(nn.Module):
    """
    The "Robot Ears": Takes raw audio and predicts the 63 parameters of the synth/effects chain.
    - AudioEncoder: Extracts a latent vector from raw audio.
    - MDNHead (Continuous): Predicts the 62 continuous knobs.
    - ClassificationHead (Wave): Predicts the 'Osc-A Wave' index (0-127).
    - ClassificationHead (Order): Predicts the effect chain permutation (0-23).
    """
    def __init__(self, latent_dim=256, num_gaussians=5):
        super().__init__()
        self.encoder = AudioEncoder(in_channels=2, latent_dim=latent_dim) # Stereo input
        
        # 53 Continuous Knobs:
        # Total(63) - Wave(1) - SatType(1) - EQ8Types(8) = 53
        self.mdn_head = MDNHead(
            in_features=latent_dim, 
            out_features=53, 
            num_gaussians=num_gaussians
        )
        
        # Categorical Heads
        self.wave_head = ClassificationHead(in_features=latent_dim, num_classes=128)
        self.sat_type_head = ClassificationHead(in_features=latent_dim, num_classes=8)
        
        # 8 dedicated heads for EQ8 band types (each 8 classes)
        self.eq8_type_heads = nn.ModuleList([
            ClassificationHead(in_features=latent_dim, num_classes=8) for _ in range(8)
        ])

        # Effect Order Permutation
        self.order_head = ClassificationHead(in_features=latent_dim, num_classes=24)

    def forward(self, audio):
        latent = self.encoder(audio)
        
        pi, mu, sigma = self.mdn_head(latent)
        wave_logits = self.wave_head(latent)
        sat_type_logits = self.sat_type_head(latent)
        eq8_type_logits = [head(latent) for head in self.eq8_type_heads]
        order_logits = self.order_head(latent)
        
        return {
            'pi': pi,
            'mu': mu,
            'sigma': sigma,
            'wave_logits': wave_logits,
            'sat_type_logits': sat_type_logits,
            'eq8_type_logits': eq8_type_logits, # List of 8 tensors
            'order_logits': order_logits
        }

    def predict(self, audio):
        """
        Inference mode: returns a dictionary with raw parameters and the effect sequence.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(audio)
            
            # 1. Sample from MDN (returns [batch, 53])
            cont_params = self.mdn_head.sample(
                outputs['pi'], outputs['mu'], outputs['sigma']
            )
            
            # 2. Get highest probability for all categorical heads
            wave_idx = torch.argmax(outputs['wave_logits'], dim=1).float()
            sat_type_idx = torch.argmax(outputs['sat_type_logits'], dim=1).float()
            eq8_type_indices = [torch.argmax(logits, dim=1).float() for logits in outputs['eq8_type_logits']]
            
            # 3. Assemble the 63 parameters
            results = torch.zeros(audio.shape[0], 63, device=audio.device)
            
            # Use a pointer p to track position in cont_params
            p = 0
            
            # --- Operator (Indices 0:16) ---
            results[:, 0] = cont_params[:, p]; p += 1      # Transpose
            results[:, 1] = wave_idx                       # Wave (Cat)
            results[:, 2:16] = cont_params[:, p:p+14]; p += 14 # Filter, Envs...
            
            # --- Saturator (Indices 16:21) ---
            results[:, 16] = cont_params[:, p]; p += 1     # Drive
            results[:, 17] = sat_type_idx                   # Type (Cat)
            results[:, 18:21] = cont_params[:, p:p+3]; p += 3 # WS Curve, Depth, Wet
            
            # --- EQ Eight (Indices 21:53) ---
            for band in range(8):
                idx = 21 + (band * 4)
                results[:, idx] = eq8_type_indices[band]    # Type (Cat)
                results[:, idx+1:idx+4] = cont_params[:, p:p+3]; p += 3 # Freq, Gain, Q
                
            # --- OTT (Indices 53:60) ---
            results[:, 53:60] = cont_params[:, p:p+7]; p += 7
            
            # --- Reverb (Indices 60:63) ---
            results[:, 60:63] = cont_params[:, p:p+3]; p += 3
            
            # 4. Decode Order Index
            import itertools
            effects = ['saturator', 'eq8', 'ott', 'reverb']
            perms = list(itertools.permutations(effects))
            order_idx = torch.argmax(outputs['order_logits'], dim=1)
            # For simplicity, returning sequence for the FIRST item in the batch
            predicted_sequence = ['operator'] + list(perms[order_idx[0]])
            
            # 5. Denormalize MDN outputs back to Ableton ranges
            from src.utils.normalization import denormalize_params
            raw_params = denormalize_params(results)
            
            return {
                'params': raw_params,
                'sequence': predicted_sequence
            }
