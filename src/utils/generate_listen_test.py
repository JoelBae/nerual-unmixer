import os
import sys
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import json
import random
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.proxy.ott import OTTProxy
from src.models.proxy.reverb import ReverbProxy
from src.models.proxy.saturator import SaturatorProxy
from src.models.proxy.eq8 import EQEightProxy
from src.models.proxy.ddsp_modules import OperatorProxy
from src.data.dataset import NeuralProxyDataset

from src.training.train_proxies import get_proxy_model

def generate_comparison(effect_name, checkpoint_path=None, num_samples=3, use_stft=False, use_stft_cond=False):
    print(f"--- Generating Listen Test for {effect_name} ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load Model
    model = get_proxy_model(effect_name, use_stft, use_stft_cond).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
    elif checkpoint_path:
        print(f"❌ Checkpoint not found at {checkpoint_path}!")
        return
    else:
        print(f"ℹ️  No checkpoint provided, using model in its current state (useful for analytical proxies).")
    
    model.eval()
    
    # 2. Setup Dataset
    dataset_dir = f"dataset/{effect_name}"
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
        
    dataset = NeuralProxyDataset(effect_name=effect_name, dataset_dir=dataset_dir, split="val", preload=False)
    
    output_dir = f"results/listen_test/{effect_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        num_to_test = min(num_samples, len(dataset))
        for i in range(num_to_test):
            idx = random.randint(0, len(dataset) - 1)
            dry, params, target, _ = dataset[idx]
            
            dry = dry.unsqueeze(0).to(device) # (1, 2, T)
            params = params.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)
            
            # Predict
            if effect_name.lower() == "operator":
                # Operator takes different arguments
                pred = model(params) 
                if pred.dim() == 2: # Mono to stereo
                    pred = pred.unsqueeze(1).repeat(1, 2, 1)
            else:
                pred = model(dry, params)
            
            # Save individual files
            torchaudio.save(f"{output_dir}/sample_{i}_dry.wav", dry.cpu().squeeze(0), 44100)
            torchaudio.save(f"{output_dir}/sample_{i}_target.wav", target.cpu().squeeze(0), 44100)
            torchaudio.save(f"{output_dir}/sample_{i}_proxy.wav", pred.cpu().squeeze(0), 44100)
            
            # Save a concatenated version for quick horizontal comparison: Dry | Target | Proxy
            silence = torch.zeros(2, 22050)
            concat = torch.cat([dry.cpu().squeeze(0), silence, target.cpu().squeeze(0), silence, pred.cpu().squeeze(0)], dim=1)
            torchaudio.save(f"{output_dir}/sample_{i}_compare.wav", concat, 44100)
            
            print(f"🌟 Saved sample {i} to {output_dir}")
            
            # 3. Spectral Plot
            plt.figure(figsize=(15, 5))
            
            # Plot Target Spectrogram
            plt.subplot(1, 2, 1)
            plt.specgram(target.cpu().squeeze(0)[0].numpy(), Fs=44100, NFFT=2048, noverlap=1024)
            plt.title(f"{effect_name} Sample {i} - Ableton Target")
            plt.colorbar(label='dB')
            
            # Plot Proxy Spectrogram
            plt.subplot(1, 2, 2)
            plt.specgram(pred.cpu().squeeze(0)[0].numpy(), Fs=44100, NFFT=2048, noverlap=1024)
            plt.title(f"{effect_name} Sample {i} - Proxy Reconstructed")
            plt.colorbar(label='dB')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/sample_{i}_spectrum.png")
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--effect", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--stft", action="store_true", help="Use the old STFT-based architecture for OTT")
    parser.add_argument("--stft_cond", action="store_true", help="Use the new Conditioned STFT-based architecture for OTT")
    args = parser.parse_args()
    
    generate_comparison(args.effect, args.checkpoint, args.num_samples, args.stft, args.stft_cond)
