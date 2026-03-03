import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def fft_convolve(signal, kernel):
    """
    Robust FFT-based convolution for differentiability and performance.
    Fallback for when torchaudio.functional.fftconvolve is missing.
    """
    try:
        import torchaudio.functional as taf
        return taf.fftconvolve(signal, kernel, mode='full')
    except (ImportError, AttributeError):
        # Manual FFT Convolve
        n_signal = signal.shape[-1]
        n_kernel = kernel.shape[-1]
        n_fft = n_signal + n_kernel - 1
        
        S = torch.fft.rfft(signal, n=n_fft)
        K = torch.fft.rfft(kernel, n=n_fft)
        Y = S * K
        y = torch.fft.irfft(Y, n=n_fft)
        return y

class ReverbProxy(nn.Module):
    """
    Learned DDSP Proxy for the Ableton Reverb effect.
    
    Architecture:
        1. An MLP takes 3 reverb params and predicts:
           - Per-band spectral envelope (32 frequency bands) — room coloring
           - Per-band decay rates (32 bands) — freq-dependent RT60
        2. These are expanded into a full IR via analytical synthesis:
           IR(t,f) = spectral_envelope(f) * noise(t,f) * exp(-decay_rate(f) * t)
        3. The IR is convolved with the input audio via fftconvolve.
        4. Dry/Wet controls the mix.
    
    The MLP only learns ~100 parameters (spectral shape + decay rates),
    not the 264,000 raw samples of a 6-second IR.
    
    Parameters from dataset: 19 total (16 Operator + 3 Reverb)
    Reverb params at indices [-3:]: [decay_time, size, dry_wet]
    """
    
    NUM_BANDS = 32  # Frequency bands for spectral shaping
    
    def __init__(self, sr=44100, max_ir_seconds=3.0):
        super().__init__()
        self.sr = sr
        self.max_ir_length = int(max_ir_seconds * sr)
        
        # Stereo noise buffers for decorrelated reverb tails
        self.register_buffer('noise_buffer', torch.randn(1, self.NUM_BANDS, 2, self.max_ir_length))
        
        self.mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        
        self.spectral_head = nn.Sequential(
            nn.Linear(256, self.NUM_BANDS),
            nn.Softplus()
        )
        
        self.decay_head = nn.Sequential(
            nn.Linear(256, self.NUM_BANDS),
            nn.Softplus()
        )

        # New: Early reflection vs late tail balance (Optional for backward compatibility)
        self.transient_head = nn.Sequential(
            nn.Linear(256, 2), # Stereo early spike weights
            nn.Softplus()
        )

    def load_state_dict(self, state_dict, strict=True):
        # Handle backward compatibility for mono noise_buffer
        if 'noise_buffer' in state_dict:
            ckpt_buffer = state_dict['noise_buffer']
            if ckpt_buffer.dim() == 3: # (1, BANDS, TIME)
                print("ℹ️  Converting mono noise_buffer to stereo...")
                state_dict['noise_buffer'] = ckpt_buffer.unsqueeze(2).repeat(1, 1, 2, 1)
            elif ckpt_buffer.dim() == 4 and ckpt_buffer.shape[2] == 1:
                print("ℹ️  Expanding mono noise_buffer[1, 32, 1, T] to stereo...")
                state_dict['noise_buffer'] = ckpt_buffer.repeat(1, 1, 2, 1)

        # Handle missing transient_head (Phase 1 checkpoints)
        if 'transient_head.0.weight' not in state_dict:
            print("ℹ️  Initializing missing transient_head for backward compatibility...")
            # We'll allow partial loading
            return super().load_state_dict(state_dict, strict=False)

        return super().load_state_dict(state_dict, strict=strict)
    
    def _synthesize_ir(self, spectral_envelope, decay_rates, transients, decay_time, batch, device):
        max_decay = torch.max(decay_time).item()
        ir_length = min(int(max_decay * self.sr) + 1, self.max_ir_length)
        ir_length = max(ir_length, self.sr // 10)
        
        t = torch.linspace(0, max_decay, ir_length, device=device).unsqueeze(0)
        
        # Use stereo noise buffer
        noise = self.noise_buffer[:, :, :, :ir_length].expand(batch, -1, -1, -1)
        
        # Decay: exp(-dr * t / tau)
        tau = decay_time.view(batch, 1, 1, 1) / 6.91
        dr = decay_rates.unsqueeze(-1).unsqueeze(-1) # (batch, NUM_BANDS, 1, 1)
        decay_env = torch.exp(-dr * t.view(1, 1, 1, -1) / (tau + 1e-8))
        
        # Scale by spectral gain
        se = spectral_envelope.unsqueeze(-1).unsqueeze(-1) # (batch, NUM_BANDS, 1, 1)
        
        # Synthesize noisy tail: (batch, NUM_BANDS, stereo, time)
        tail = noise * se * decay_env
        ir_tail = tail.sum(dim=1) # (batch, 2, ir_length)
        
        # New: Add Early Reflection Spike (Transients) at t=0
        # This models the "slap" sounds of the room before the diffusion kicks in
        er_spike = torch.zeros_like(ir_tail)
        er_spike[:, :, 0] = transients # Set weight at t=0
        
        ir = ir_tail + er_spike
        
        # Normalize IR energy
        ir_energy = torch.sqrt(torch.sum(ir.pow(2), dim=(1, 2), keepdim=True) + 1e-8)
        ir = ir / ir_energy
        
        return ir

    def forward(self, audio, params):
        batch, channels, time = audio.shape
        device = audio.device
        
        reverb_params = params[:, -3:]
        decay_time = torch.clamp(reverb_params[:, 0], 0.1, 6.0)
        dry_wet = torch.clamp(reverb_params[:, 2], 0.0, 1.0).view(batch, 1, 1)
        
        h = self.mlp(reverb_params)
        spectral_envelope = self.spectral_head(h)
        decay_rates = self.decay_head(h)
        transients = self.transient_head(h)
        
        ir = self._synthesize_ir(spectral_envelope, decay_rates, transients, decay_time, batch, device)
        
        # Flatten for convolution
        audio_flat = audio.reshape(batch * channels, time)
        ir_flat = ir.reshape(batch * channels, -1)
        
        wet = fft_convolve(audio_flat, ir_flat)
        wet = wet[:, :time]
        wet = wet.reshape(batch, channels, time)
        
        out_audio = (1.0 - dry_wet) * audio + dry_wet * wet
        
        return out_audio
