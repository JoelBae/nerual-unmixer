import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Callable
import os
from utils.feature_extractor import AudioFeatureExtractor, get_mel_spectrogram

class AlignmentValidator:
    """
    Validation suite to compute Jacobian alignment (alpha_c) and divergence (epsilon)
    between DDSP proxies and target VSTs.
    """
    def __init__(self, engine: Any, sample_rate: int = 44100):
        self.engine = engine
        self.sample_rate = sample_rate
        self.feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)

    def generate_input_signal(
        self, 
        signal_type: str = "noise", 
        duration_sec: float = 0.1, 
        **kwargs
    ) -> np.ndarray:
        """
        Generates input audio for verification.
        Supported types: 'noise', 'sine_sweep', 'vital'
        """
        num_samples = int(duration_sec * self.sample_rate)
        if signal_type == "noise":
            # Stereo noise for VST
            return np.random.normal(0, 0.1, (2, num_samples)).astype(np.float32)
        
        elif signal_type == "sine_sweep":
            t = np.linspace(0, duration_sec, num_samples)
            sweep = np.sin(2 * np.pi * 20 * (1000 ** (t / duration_sec)))
            return np.tile(sweep, (2, 1)).astype(np.float32)
            
        elif signal_type == "vital":
            # Use Vital as a source of harmonically rich input
            patch_params = kwargs.get("patch_params", {"cutoff": 0.5, "res": 0.0})
            note_freq = kwargs.get("note_freq", 220.0) # A3
            return self.engine.render("vital", patch_params, duration_sec, note_freq=note_freq)
        
        elif signal_type == "full_spectrum":
            t = np.linspace(0, duration_sec, num_samples)
            noise = np.random.normal(0, 0.05, num_samples) 
            sine_l = np.sin(2 * np.pi * 50 * t) * 0.3    # Low
            sine_m = np.sin(2 * np.pi * 1000 * t) * 0.3  # Mid
            sine_h = np.sin(2 * np.pi * 10000 * t) * 0.3 # High
            combined = (noise + sine_l + sine_m + sine_h).astype(np.float32)
            # Normalize to avoid hard clipping
            combined = 0.5 * combined / (np.max(np.abs(combined)) + 1e-8)
            return np.tile(combined, (2, 1))
            
        elif signal_type == "bass_saw":
            t = np.linspace(0, duration_sec, num_samples)
            freq = 55.0 # A1 bass
            # Band-limited sawtooth proxy
            saw = np.zeros_like(t)
            for n in range(1, 12): # 11 harmonics
                saw += (1.0 / n) * np.sin(2 * np.pi * n * freq * t)
            # Normalize and scale
            saw = 0.5 * saw / (np.max(np.abs(saw)) + 1e-8)
            return np.tile(saw.astype(np.float32), (2, 1))
            
        elif signal_type == "mid_saw":
            t = np.linspace(0, duration_sec, num_samples)
            freq = 440.0 # A4 mid
            # Band-limited sawtooth proxy
            saw = np.zeros_like(t)
            for n in range(1, 12): # 11 harmonics
                saw += (1.0 / n) * np.sin(2 * np.pi * n * freq * t)
            # Normalize and scale
            saw = 0.5 * saw / (np.max(np.abs(saw)) + 1e-8)
            return np.tile(saw.astype(np.float32), (2, 1))
        
        else:
            raise ValueError(f"Unsupported signal type: {signal_type}")

    def compute_cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.flatten()
        b_flat = b.flatten()
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a_flat, b_flat) / (norm_a * norm_b + 1e-8)

    def compute_frobenius_divergence(self, h_grad: np.ndarray, h_tilde_grad: np.ndarray) -> float:
        """Computes epsilon (divergence) as \| \nabla h - \nabla \tilde{h} \|_F"""
        error = h_grad - h_tilde_grad
        return np.linalg.norm(error) # L2 norm/Frobenius for these vectors

    def compute_alignment_metrics(
        self,
        plugin_id: str,
        proxy: torch.nn.Module,
        vst_params: Dict[str, float],
        proxy_param_mapping: Dict[str, str],
        input_audio: np.ndarray,
        duration_sec: float = 0.1,
        delta: float = 1e-3,
        representation: str = "mel"
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Computes alpha_c (Jacobian Alignment) for each target parameter.
        Note: Removed epsilon from here as it now refers to input-output divergence.
        """
        j_vst = self.get_jacobian(plugin_id, proxy, vst_params, proxy_param_mapping, input_audio, duration_sec, delta, representation, mode="vst")
        j_proxy = self.get_jacobian(plugin_id, proxy, vst_params, proxy_param_mapping, input_audio, duration_sec, delta, representation, mode="proxy")
        
        results = {}
        for p in proxy_param_mapping.keys():
            results[p] = float(self.compute_cosine_similarity(j_vst[p], j_proxy[p]))
            
        base_audio_vst = self.engine.render(plugin_id, vst_params, duration_sec, input_audio)
        return results, base_audio_vst

    def get_jacobian(
        self,
        plugin_id: str,
        proxy: torch.nn.Module,
        vst_params: Dict[str, float],
        proxy_param_mapping: Dict[str, str],
        input_audio: np.ndarray,
        duration_sec: float = 0.1,
        delta: float = 1e-3,
        representation: str = "mel",
        mode: str = "vst"
    ) -> Dict[str, np.ndarray]:
        """
        Returns the raw Jacobian columns for either VST or Proxy.
        """
        if representation == "mel":
            def h(x: np.ndarray) -> np.ndarray: return get_mel_spectrogram(x, self.sample_rate)
        else:
            def h(x: np.ndarray) -> np.ndarray: return x[0]
            
        jacobian = {}
        
        if mode == "vst":
            for vst_p in proxy_param_mapping.keys():
                p_plus = vst_params.copy(); p_plus[vst_p] = min(1.0, vst_params[vst_p] + delta)
                p_minus = vst_params.copy(); p_minus[vst_p] = max(0.0, vst_params[vst_p] - delta)
                grad = (h(self.engine.render(plugin_id, p_plus, duration_sec, input_audio)) - 
                        h(self.engine.render(plugin_id, p_minus, duration_sec, input_audio))) / (p_plus[vst_p] - p_minus[vst_p])
                jacobian[vst_p] = grad
        else:
            input_audio_torch = torch.from_numpy(input_audio[0]).unsqueeze(0).float()
            for vst_p, proxy_p in proxy_param_mapping.items():
                def get_proxy_rep(val):
                    p_inputs = {pp: torch.tensor([[val if vp == vst_p else vst_params[vp]]]).float() 
                                for vp, pp in proxy_param_mapping.items()}
                    return h(proxy(input_audio_torch, **p_inputs).detach().numpy())

                grad = (get_proxy_rep(min(1.0, vst_params[vst_p] + delta)) - 
                        get_proxy_rep(max(0.0, vst_params[vst_p] - delta))) / (2 * delta)
                jacobian[vst_p] = grad
                
        return jacobian

    def compute_input_output_divergence(
        self,
        plugin_id: str,
        proxy: torch.nn.Module,
        vst_params: Dict[str, float],
        proxy_param_mapping: Dict[str, str],
        input_audio: np.ndarray,
        duration_sec: float = 0.1,
        noise_std: float = 1e-3,
        representation: str = "mel"
    ) -> float:
        """
        Computes epsilon: the Frobenius norm divergence between VST and Proxy input Jacobians.
        Estimated via finite difference perturbation of the input audio.
        """
        if representation == "mel":
            def h(x: np.ndarray) -> np.ndarray: return get_mel_spectrogram(x, self.sample_rate)
        else:
            def h(x: np.ndarray) -> np.ndarray: return x[0]

        # 1. Base Outputs
        base_vst = h(self.engine.render(plugin_id, vst_params, duration_sec, input_audio))
        
        input_audio_torch = torch.from_numpy(input_audio[0]).unsqueeze(0).float()
        p_inputs = {pp: torch.tensor([[vst_params[vp]]]).float() for vp, pp in proxy_param_mapping.items()}
        base_proxy = h(proxy(input_audio_torch, **p_inputs).detach().numpy())

        # 2. Perturbed Outputs
        noise = np.random.normal(0, noise_std, input_audio.shape).astype(np.float32)
        input_perturbed = input_audio + noise
        
        pert_vst = h(self.engine.render(plugin_id, vst_params, duration_sec, input_perturbed))
        
        input_perturbed_torch = torch.from_numpy(input_perturbed[0]).unsqueeze(0).float()
        pert_proxy = h(proxy(input_perturbed_torch, **p_inputs).detach().numpy())

        # 3. Compute Divergence
        # G = (h(x + dx) - h(x)) / ||dx||
        vst_diff = (pert_vst - base_vst)
        proxy_diff = (pert_proxy - base_proxy)
        
        # Epsilon is the distance between these directional derivatives
        epsilon = np.linalg.norm(vst_diff - proxy_diff) / (np.linalg.norm(noise) + 1e-8)
        
        return float(epsilon)

    def plot_alignment(self, vst_grad: np.ndarray, proxy_grad: np.ndarray, param_name: str, save_path: str):
        """Visualizes the alignment of gradients in the time or mel domain."""
        plt.figure(figsize=(12, 4))
        plt.plot(vst_grad, label="VST Jacobian", alpha=0.7)
        plt.plot(proxy_grad, label="Proxy Jacobian", alpha=0.7, linestyle='--')
        plt.title(f"Jacobian Alignment: {param_name}")
        plt.legend()
        plt.grid(True)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
