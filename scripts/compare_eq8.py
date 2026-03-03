import os
import sys
import json
import torch
import torchaudio
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.models.proxy.eq8 import EQEightProxy
from src.models.losses import VectorizedMultiScaleSpectralLoss

def compare():
    print("--- Comparing EQ8 Proxy vs Ableton Output (TEST) ---")
    dataset_dir = "dataset/eq8_restore_test/eq8"
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata not found at {metadata_path}")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    num_to_test = min(len(metadata), 1) # Just test 1 for now for speed
    for sample_idx in range(num_to_test):
        sample = metadata[sample_idx]
        print(f"\n--- Testing Sample Index {sample['index']} ---")
    
    input_path = os.path.join(dataset_dir, sample['input_file'])
    target_path = os.path.join(dataset_dir, sample['output_file'])
    
    # 1. Load Audio
    input_audio, sr = torchaudio.load(input_path)
    target_audio, _ = torchaudio.load(target_path)
    print(f"Sample Rate: {sr} Hz")
    
    # 2. Extract Parameters for the 8 bands
    params_list = sample['settings']
    # Filter for Band A parameters (Device 1)
    eq8_params = [p for p in params_list if p['device'] == 1 and "Band" in p['name']]
    print(f"Found {len(eq8_params)} EQ8 parameters in metadata")
    
    bands_raw = []
    found_bands = 0
    for band_idx in range(8):
        base = 4 + (band_idx * 10)
        t = [p['raw_value'] for p in eq8_params if p['param'] == base + 1]
        f = [p['raw_value'] for p in eq8_params if p['param'] == base + 2]
        g = [p['raw_value'] for p in eq8_params if p['param'] == base + 3]
        q = [p['raw_value'] for p in eq8_params if p['param'] == base + 4]
        
        if len(t)>0 and len(f)>0 and len(g)>0 and len(q)>0:
            bands_raw.extend([t[0], f[0], g[0], q[0]])
            found_bands += 1
        else:
            print(f"⚠️ Band {band_idx+1} missing data! Using defaults.")
            bands_raw.extend([3.0, 1000.0, 0.0, 0.707])
            
    print(f"Extracted {found_bands} active bands.")
    proxy_params = torch.tensor([bands_raw]) # [1, 32]
    
    # 3. Initialize Proxy
    proxy = EQEightProxy(sr=sr)
    
    # 4. Run Proxy
    # input_audio: (2, time) -> (1, 2, time)
    input_audio_batch = input_audio.unsqueeze(0)
    proxy_output = proxy(input_audio_batch, proxy_params).squeeze(0)
    
    # 5. Measure Loss with Time Alignment
    print("Aligning waveforms...")
    align_len = int(sr * 0.5)
    p_sig = proxy_output[0, :align_len].numpy()
    t_sig = target_audio[0, :align_len].numpy()
    
    correlation = np.correlate(t_sig, p_sig, mode='full')
    offset = np.argmax(correlation) - (len(p_sig) - 1)
    print(f"Detected Latency: {offset} samples ({(offset/sr)*1000:.2f}ms)")
    
    # Apply offset
    if offset > 0:
        # Target starts LATER than Proxy (Proxy is leading)
        t_aligned = target_audio[:, offset:]
        p_aligned = proxy_output[:, :-offset]
    elif offset < 0:
        # Proxy starts LATER than Target (Target is leading)
        abs_offset = abs(offset)
        t_aligned = target_audio[:, :-abs_offset]
        p_aligned = proxy_output[:, abs_offset:]
    else:
        t_aligned = target_audio
        p_aligned = proxy_output

    min_len = min(p_aligned.shape[-1], t_aligned.shape[-1])
    p_out = p_aligned[:, :min_len]
    t_out = t_aligned[:, :min_len]
    
    print(f"Proxy Peak: {torch.max(torch.abs(p_out)).item():.4f}")
    print(f"Target Peak: {torch.max(torch.abs(t_out)).item():.4f}")
    
    # Normalize volume for shape comparison if they are clearly different
    # But let's first see the raw loss breakdown
    
    # LOSS BREAKDOWN
    print("--- Loss Breakdown ---")
    fft_sizes = [512, 1024, 2048]
    total_sc = 0
    total_mag = 0
    for n in fft_sizes:
        win = torch.hann_window(n)
        hop = n // 4
        xs = torch.stft(p_out.mean(0, keepdim=True), n, hop, window=win, return_complex=True)
        ys = torch.stft(t_out.mean(0, keepdim=True), n, hop, window=win, return_complex=True)
        xm = torch.abs(xs) + 1e-7
        ym = torch.abs(ys) + 1e-7
        
        sc = torch.norm(ym - xm, p='fro') / torch.norm(ym, p='fro')
        mag = torch.nn.functional.l1_loss(torch.log(xm), torch.log(ym))
        print(f"Size {n}: SC={sc.item():.4f}, Mag={mag.item():.4f}")
        total_sc += sc
        total_mag += mag
        
    print(f"Average SC: {(total_sc/len(fft_sizes)).item():.4f}")
    print(f"Average Mag: {(total_mag/len(fft_sizes)).item():.4f}")
    print(f"Total Combined: {((total_sc + total_mag)/len(fft_sizes)).item():.4f}")

    if HAS_MATPLOTLIB:
        plt.figure(figsize=(15, 12))
        plt.subplot(3, 1, 1)
        plt.title("Waveform (0-20ms)")
        plt.plot(t_out[0, :int(sr*0.02)].numpy(), label="Ableton", alpha=0.6)
        plt.plot(p_out[0, :int(sr*0.02)].numpy(), label="Proxy", alpha=0.6, linestyle='--')
        plt.legend()
        
        # Transfer Function Analysis
        n_fft_plot = 2048
        freqs = np.fft.rfftfreq(n=n_fft_plot, d=1/sr)
        # Take a larger chunk for better average
        chunk_len = min(65536, len(t_out[0]))
        t_chunk = t_out[0, :chunk_len].numpy()
        p_chunk = p_out[0, :chunk_len].numpy()
        i_chunk = input_audio[0, :chunk_len].numpy() # Correct for latency!
        
        # We need to align input_audio too if we want to use it for Transfer Function
        # Actually, let's just use Proxy Out / Input vs Target Out / Input
        # But for H_proxy, we already know the mask H_combined.
        
        # Let's compute them from STFT to be consistent
        spec_i = np.abs(np.fft.rfft(i_chunk, n=n_fft_plot)) 
        spec_t = np.abs(np.fft.rfft(t_chunk, n=n_fft_plot))
        spec_p = np.abs(np.fft.rfft(p_chunk, n=n_fft_plot))
        
        # H = Magnitude Response
        H_target = spec_t / (spec_i + 1e-6)
        H_proxy = spec_p / (spec_i + 1e-6)

        # Direct Magnitude Check
        print("--- Direct Magnitude Comparison (dB) ---")
        for f_check in [100, 1000, 10000]:
            idx_f = np.argmin(np.abs(freqs - f_check))
            db_i = 20 * np.log10(spec_i[idx_f] + 1e-6)
            db_t = 20 * np.log10(spec_t[idx_f] + 1e-6)
            db_p = 20 * np.log10(spec_p[idx_f] + 1e-6)
            print(f"{f_check}Hz: Input={db_i:.2f}, Target={db_t:.2f}, Proxy={db_p:.2f}")
            
        print("--- Transfer Function Comparison (dB) ---")
        for f_check in [100, 1000, 10000]:
            idx_f = np.argmin(np.abs(freqs - f_check))
            val_t = 20 * np.log10(H_target[idx_f] + 1e-6)
            val_p = 20 * np.log10(H_proxy[idx_f] + 1e-6)
            print(f"{f_check}Hz: Target_H={val_t:.2f}dB, Proxy_H={val_p:.2f}dB (Diff={val_t-val_p:.2f}dB)")

        plt.subplot(3, 1, 3)
        plt.title("H_target vs H_proxy (dB)")
        plt.semilogx(freqs, 20*np.log10(H_target + 1e-6), label="Ableton H")
        plt.semilogx(freqs, 20*np.log10(H_proxy + 1e-6), label="Proxy H", linestyle='--')
        plt.grid()
        plt.legend()
        plt.xlim(20, 20000)
        plt.ylim(-30, 30)
        
        plt.tight_layout()
        plt.savefig("eq8_diagnostic.png")
        print("Diagnostic plot saved to eq8_diagnostic.png")

if __name__ == "__main__":
    compare()
