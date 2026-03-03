import torch
import torchaudio.functional as F
import random
import random

class AudioAugmentor:
    """
    Applies domain randomization augmentations to audio tensors.
    """
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        
    def __call__(self, audio):
        # audio shape: (Batch, Channels, Length) or (Channels, Length)
        aug_audio = audio.clone()
        
        # If single item (C, L), temporarily add batch dim (1, C, L)
        is_single = False
        if aug_audio.dim() == 2:
            is_single = True
            aug_audio = aug_audio.unsqueeze(0)
            
        batch_size = aug_audio.shape[0]
        device = aug_audio.device
        
        # 1. Polarity Inversion (50% chance per sample)
        # Random boolean mask: (Batch, 1, 1)
        invert_mask = (torch.rand(batch_size, 1, 1, device=device) < 0.5).float()
        # Map 1.0 (invert) to -1.0, and 0.0 (keep) to 1.0
        polarity_multiplier = 1.0 - (2.0 * invert_mask)
        aug_audio = aug_audio * polarity_multiplier
            
        # 2. Random Gain (-6dB to +3dB per sample)
        gain_db = torch.empty(batch_size, 1, 1, device=device).uniform_(-6.0, 3.0)
        gain_linear = 10.0 ** (gain_db / 20.0)
        aug_audio = aug_audio * gain_linear
        
        # 3. White Noise Injection (SNR 40dB to 80dB per sample)
        # Calculate signal power per sample: mean across Channels and Length
        signal_power = torch.mean(aug_audio ** 2, dim=(1, 2), keepdim=True)
        
        # Only add noise if there is an actual signal
        has_signal_mask = (signal_power > 0).float()
        
        target_snr_db = torch.empty(batch_size, 1, 1, device=device).uniform_(40.0, 80.0)
        target_snr_linear = 10.0 ** (target_snr_db / 10.0)
        
        noise_power = signal_power / target_snr_linear
        noise = torch.randn_like(aug_audio) * torch.sqrt(noise_power)
        
        aug_audio = aug_audio + (noise * has_signal_mask)
        
        # 4. Random EQ (50% chance per sample to get a random peak or shelf)
        if hasattr(F, 'biquad'):
            # Generate random eq parameters
            # Type: 0 = peak, 1 = low shelf, 2 = high shelf
            eq_types = torch.randint(0, 3, (batch_size,), device=device)
            eq_freqs = torch.empty(batch_size, device=device).uniform_(100.0, 10000.0)
            eq_gains = torch.empty(batch_size, device=device).uniform_(-6.0, 6.0)
            eq_qs = torch.empty(batch_size, device=device).uniform_(0.5, 2.0)
            apply_eq_mask = torch.rand(batch_size, device=device) < 0.5
            
            # Apply EQ sequentially (biquad doesn't strictly support batched varying coefficients easily)
            for i in range(batch_size):
                if not apply_eq_mask[i]: continue
                
                freq = eq_freqs[i].item()
                gain = eq_gains[i].item()
                q = eq_qs[i].item()
                
                try: # Fallback just in case biquad math fails on extreme random values
                    if eq_types[i] == 0:
                        aug_audio[i] = F.equalizer_biquad(aug_audio[i], self.sample_rate, freq, gain, q)
                    elif eq_types[i] == 1:
                        aug_audio[i] = F.bass_biquad(aug_audio[i], self.sample_rate, gain, freq, q)
                    else:
                        aug_audio[i] = F.treble_biquad(aug_audio[i], self.sample_rate, gain, freq, q)
                except Exception:
                    pass
        
        # Remove batch dim if we added it
        if is_single:
            aug_audio = aug_audio.squeeze(0)
            
        return aug_audio
