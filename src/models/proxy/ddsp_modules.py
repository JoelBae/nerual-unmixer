import torch
import torch.nn as nn
import numpy as np
import os

class HarmonicOscillator(nn.Module):
    """
    A Differentiable Additive Synthesizer similar to Ableton's Operator.
    Generates N harmonic sine waves and sums them together based on their amplitudes.
    """
    def __init__(self, sample_rate=44100, num_harmonics=64):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_harmonics = num_harmonics
        harmonics = torch.arange(1, num_harmonics + 1, dtype=torch.float32)
        self.register_buffer("harmonics", harmonics)

        self.filter = DifferentiableAdditiveFilter()
        
    def forward(self, f0, harmonic_amplitudes, cutoff_frequencies=None, filter_resonance=0.707, num_samples=88200):
        """
        f0: Tensor (batch, 1) - The fundamental pitch (e.g. 440 Hz)
        harmonic_amplitudes: Tensor (batch, 64) - The volume of each harmonic overtone
        num_samples: Int - How many audio samples to generate
        """
        batch_size = f0.shape[0]
        t = torch.arange(num_samples, device=f0.device, dtype=torch.float32)
        t = t.view(1, 1, -1).expand(batch_size, 1, -1)
        harmonics_expanded = self.harmonics.view(1, -1, 1)
        all_frequencies = f0 * harmonics_expanded
        phase = 2.0 * np.pi * torch.cumsum(all_frequencies, dim=-1) / self.sample_rate
        
        all_sines = torch.sin(phase)
        amps_expanded = harmonic_amplitudes.unsqueeze(2)

        if cutoff_frequencies is not None:
            filter_gains = self.filter(all_frequencies, cutoff_frequencies, filter_resonance)
            amps_expanded = amps_expanded * filter_gains

        weighted_sines = all_sines * amps_expanded

        audio = torch.sum(weighted_sines, dim=1)
        
        return audio

class DifferentiableADSR(nn.Module):
    """
    A mathematical ADSR Envelope that shapes the volume or filter cutoff of a sound.
    """
    def __init__(self, sample_rate=44100):
        super().__init__()
        self.sample_rate = sample_rate
        
    def forward(self, attack, decay, sustain, release, note_off_time=1.0, num_samples=88200):
        """
        attack, decay, release: Tensors (batch, 1) - Time in seconds
        sustain: Tensor (batch, 1) - Sustain level (0.0 to 1.0)
        note_off_time: Float - When the MIDI note is released in seconds
        num_samples: Int - Total audio length to generate
        """
        batch_size = attack.shape[0]
        t = torch.arange(num_samples, device=attack.device, dtype=attack.dtype)
        # Reshape for broadcasting: (batch, 1) and (1, time) -> (batch, time)
        t = t.view(1, -1) / self.sample_rate

        attack_env = torch.clamp(t / (attack + 1e-8), 0.0, 1.0)
        decay_env = 1.0 - (1.0 - sustain) * torch.clamp((t - attack) / (decay + 1e-8), 0.0, 1.0)
        release_env = 1.0 - torch.clamp((t - note_off_time) / (release + 1e-8), 0.0, 1.0)

        envelope = attack_env * decay_env * release_env
        envelope = torch.clamp(envelope, 0.0, 1.0)
        return envelope

class DifferentiablePitchEnvelope(nn.Module):
    """
    Generates a pitch offset curve (in semitones) that decays over time.
    """ 
    def __init__(self, sample_rate=44100):
        super().__init__()
        self.sample_rate = sample_rate
    
    def forward(self, amount, decay, peak, num_samples=88200):
        """
        amount: (batch, 1) - Overall intensity of the pitch envelope (from -1.0 to 1.0)
        decay:  (batch, 1) - Time in seconds for the pitch to return to normal
        peak:   (batch, 1) - Maximum semi-tones to jump (e.g. 12, 24, 48)
        """
        batch_size = amount.shape[0]
        
        t = torch.arange(num_samples, device=amount.device, dtype=amount.dtype)
        t = t.view(1, -1) / self.sample_rate
        
        # decay_curve: (batch, time)
        decay_curve = 1.0 - torch.clamp(t / (decay + 1e-8), 0.0, 1.0)
        
        max_semitone_jump = peak * amount 
        pitch_offset_semitones = decay_curve * max_semitone_jump.view(batch_size, 1)
        
        return pitch_offset_semitones

class OscWaveMapper(nn.Module):
    """
    A Differentiable Lookup Table that maps 'Osc-A Wave' dial value (0.0 to 1.0)
    into 64 harmonic amplitudes using linear interpolation.
    """
    def __init__(self, num_harmonics=64, num_entries=128):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.num_entries = num_entries
        # Initialize with Sine wave (first harmonic only)
        table = torch.zeros(num_entries, num_harmonics)
        table[:, 0] = 1.0
        self.register_buffer("table", table)

    def forward(self, wave_dial_normalized=None, wave_logits=None):
        """
        Original: wave_dial_normalized (batch, 1) - from 0.0 to 1.0 (for lerp)
        New: wave_logits (batch, 128) - for soft weighted sum (differentiable)
        """
        if wave_logits is not None:
            # Differentiable Discrete-Lookup (Gumbel-Softmax Trick)
            # This forces the network to pick exactly ONE waveform (hard=True)
            # during the forward pass so we don't blend Sine and Square waves (sounds bad),
            # but allows continuous gradients to flow back during the backward pass.
            # Weights: (batch, 128) - One-hot vector functionally
            weights = torch.nn.functional.gumbel_softmax(wave_logits, tau=1.0, hard=True, dim=-1)
            # table: (128, 64)
            # result: (batch, 64)
            return torch.matmul(weights, self.table)
            
        # Fallback to legacy lerp dial if no logits provided
        idx_float = wave_dial_normalized * (self.num_entries - 1)
        idx_lower = torch.floor(idx_float).long()
        idx_upper = torch.ceil(idx_float).long()
        alpha = idx_float - idx_lower.float()
        idx_lower = torch.clamp(idx_lower, 0, self.num_entries - 1)
        idx_upper = torch.clamp(idx_upper, 0, self.num_entries - 1)
        lower_vals = self.table[idx_lower.squeeze(1)]
        upper_vals = self.table[idx_upper.squeeze(1)]
        return lower_vals * (1.0 - alpha) + upper_vals * alpha

class DifferentiableAdditiveFilter(nn.Module):
    """
    DDSP Filtering Trick: Instead of a slow time-domain recursive equation, 
    we calculate the theoretical frequency response of a Low-Pass Filter
    and directly reduce the volume of harmonics that are above the cutoff
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, all_frequencies, cutoff_frequencies, resonance=0.707):
        # resonance: (batch, 1) or float
        if torch.is_tensor(resonance) and resonance.dim() == 2:
            resonance = resonance.unsqueeze(2) # (batch, 1, 1)
            
        f_ratio = all_frequencies / (cutoff_frequencies + 1e-8)

        denominator = torch.sqrt((1.0 - f_ratio**2)**2 + (f_ratio / (resonance + 1e-8))**2 + 1e-8)
        filter_gain = 1.0 / denominator
        
        return filter_gain

class DifferentiableFilterEnvelope(nn.Module):
    """
    Generates an ADSR curve that sweeps the filter cutoff frequency up and down.
    Reuses the DifferentiableADSR class for the core 0-1 shape!
    """
    def __init__(self, sample_rate=44100):
        super().__init__()
        self.sample_rate = sample_rate
        self.adsr = DifferentiableADSR(sample_rate)
        
    def forward(self, amount, attack, decay, sustain, release, note_off_time=1.0, num_samples=88200):
        adsr_curve = self.adsr(attack, decay, sustain, release, note_off_time, num_samples)
        max_octave_sweep = amount * 5.0 
        frequency_multiplier = 2.0 ** (adsr_curve * max_octave_sweep.expand(-1, num_samples))
        
        return frequency_multiplier


class OperatorProxy(nn.Module):
    """
    A Differentiable PyTorch proxy for Ableton's Operator.
    It takes a tensor of 16 normalized parameters (between 0.0 and 1.0) 
    and outputs the generated audio.
    """
    def __init__(self, sample_rate=44100, num_harmonics=64):
        super().__init__()
        self.sample_rate = sample_rate
        
        self.oscillator = HarmonicOscillator(sample_rate, num_harmonics)
        self.amplitude_env = DifferentiableADSR(sample_rate)
        self.pitch_env = DifferentiablePitchEnvelope(sample_rate)
        self.filter_env = DifferentiableFilterEnvelope(sample_rate)
        self.wave_mapper = OscWaveMapper(num_harmonics=num_harmonics)

    def load_wave_table(self, path="checkpoints/wave_table.pt"):
        """Loads analyzed harmonic table."""
        if os.path.exists(path):
            loaded_table = torch.load(path, map_location=self.wave_mapper.table.device)
            self.wave_mapper.table.copy_(loaded_table)
            print(f"✅ Loaded Wave Lookup Table from {path}")
        else:
            print(f"⚠️  Wave Table not found at {path}. Using default sine wave.")

    def forward(self, params, wave_logits=None, note_off_time=1.0, num_samples=88200):
        # Ensure 2D params
        if params.dim() == 1:
            params = params.unsqueeze(0)
        
        batch_size = params.shape[0]

        # Transpose
        transpose_dial = params[:, 0:1] 
        semitones_shift = (transpose_dial - 64.0) * (48.0 / 64.0) 
        f0 = 130.81 * (2.0 ** (semitones_shift / 12.0))

        # ADSR
        attack = (params[:, 12:13] / 127.0) * 20.0
        decay = (params[:, 13:14] / 127.0) * 60.0
        sustain = params[:, 14:15] / 127.0
        release = (params[:, 15:16] / 127.0) * 60.0

        # Pitch Envelope
        pe_amount = (params[:, 9:10] / 64.0) - 1.0
        pe_decay = (params[:, 10:11] / 127.0) * 20.0
        pe_peak = (params[:, 11:12] - 64.0) * (48.0 / 64.0)
        pitch_offset = self.pitch_env(pe_amount, pe_decay, pe_peak, num_samples)
        f0_modulated = f0.unsqueeze(2) * (2.0 ** (pitch_offset.unsqueeze(1) / 12.0))

        # Osc-A Wave Mapping
        if wave_logits is not None:
             harmonic_amplitudes = self.wave_mapper(wave_logits=wave_logits)
        else:
             wave_dial_normalized = params[:, 1:2] / 127.0
             harmonic_amplitudes = self.wave_mapper(wave_dial_normalized=wave_dial_normalized)

        # Filter
        filter_dial = params[:, 2:3] / 127.0
        base_cutoff = 20.0 * (1000.0 ** filter_dial) 

        # Filter Resonance
        filter_res = 0.1 + (params[:, 3:4] / 127.0) * 4.0
        
        # Filter Envelope
        fe_amount = (params[:, 4:5] / 64.0) - 1.0 
        fe_attack = (params[:, 5:6] / 127.0) * 20.0
        fe_decay = (params[:, 6:7] / 127.0) * 60.0
        fe_sustain = params[:, 7:8] / 127.0
        fe_release = (params[:, 8:9] / 127.0) * 60.0

        freq_multiplier = self.filter_env(fe_amount, fe_attack, fe_decay, fe_sustain, fe_release, note_off_time, num_samples)
        cutoff_frequencies = base_cutoff.unsqueeze(2) * freq_multiplier.unsqueeze(1)

        # Generate Audio
        raw_audio = self.oscillator(f0=f0_modulated, harmonic_amplitudes=harmonic_amplitudes, cutoff_frequencies=cutoff_frequencies, filter_resonance=filter_res, num_samples=num_samples)
        volume_envelope = self.amplitude_env(attack, decay, sustain, release, note_off_time, num_samples)

        final_audio = raw_audio * volume_envelope
        
        # Ensure (B, 2, T) for consistency with other proxies
        if final_audio.dim() == 2:
            final_audio = final_audio.unsqueeze(1).repeat(1, 2, 1)
            
        return final_audio