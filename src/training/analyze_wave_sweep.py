import os
import json
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
import matplotlib.pyplot as plt

def extract_harmonics(audio_path, num_harmonics=64):
    """
    Analyzes an audio file to extract the first num_harmonics.
    """
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio[:, 0] # Use mono
        
    # Take middle segment to avoid onsets/offsets
    # 1.0s audio @ 44.1k -> 44100 samples.
    # We take the middle 0.5s.
    start_idx = int(0.25 * len(audio))
    end_idx = int(0.75 * len(audio))
    segment = audio[start_idx:end_idx]
    
    # Hann window
    window = np.hanning(len(segment))
    segment = segment * window
    
    # FFT
    fft = np.fft.rfft(segment)
    mags = np.abs(fft)
    freqs = np.fft.rfftfreq(len(segment), 1/sr)
    
    # 1. Find Fundamental (highest peak in expected range)
    # The sweep is fixed to Transpose 64 (C3 ~130.81Hz)
    f_min, f_max = 50, 500
    search_range = (freqs >= f_min) & (freqs <= f_max)
    idx_f0 = np.argmax(mags[search_range])
    f0 = freqs[search_range][idx_f0]
    
    # 2. Extract Harmonics
    harmonic_amplitudes = []
    for h in range(1, num_harmonics + 1):
        target_f = h * f0
        if target_f > sr/2:
            harmonic_amplitudes.append(0.0)
            continue
            
        # Find closest bin
        idx = np.argmin(np.abs(freqs - target_f))
        
        # Take max in a small window around the target
        win_size = 3 
        val = np.max(mags[idx-win_size:idx+win_size+1])
        harmonic_amplitudes.append(val)
        
    harmonic_amplitudes = np.array(harmonic_amplitudes)
    
    # Normalize: Fundamental is always 1.0
    if harmonic_amplitudes[0] > 1e-6:
        harmonic_amplitudes = harmonic_amplitudes / harmonic_amplitudes[0]
            
    return harmonic_amplitudes

def analyze_sweep(dataset_dir, output_path="checkpoints/wave_table.pt"):
    print(f"--- Analyzing wave sweep in {dataset_dir} ---")
    
    meta_path = os.path.join(dataset_dir, "wave_metadata.json")
    if not os.path.exists(meta_path):
        print(f"Error: metadata not found at {meta_path}. Run generate_wave_sweep.py first.")
        return
        
    with open(meta_path, "r") as f:
        metadata = json.load(f)
        
    # We expect 128 entries
    table = np.zeros((128, 64))
    
    for item in tqdm(metadata, desc="Analyzing Waveforms"):
        wave_val = item["wave_value"]
        audio_path = os.path.join(dataset_dir, item["filename"])
        
        try:
            harmonics = extract_harmonics(audio_path)
            table[wave_val] = harmonics
        except Exception as e:
            print(f"Failed to analyze {audio_path}: {e}")
            
    # Save as Torch Tensor
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    table_tensor = torch.FloatTensor(table)
    torch.save(table_tensor, output_path)
    
    print(f"✅ Successfully saved (128, 64) lookup table to {output_path}")
    
    # --- Quick Plot of Important Waves ---
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate([0, 32, 64, 96, 127]):
        plt.subplot(1, 5, i+1)
        plt.bar(range(64), table[idx])
        plt.title(f"Wave Dial: {idx}")
        plt.ylim(0, 1.1)
        
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/wave_table_analysis.png")
    print("Verification plot saved to results/plots/wave_table_analysis.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./dataset/wave_sweep")
    parser.add_argument("--output", type=str, default="checkpoints/wave_table.pt")
    
    args = parser.parse_args()
    analyze_sweep(args.dataset_dir, args.output)
