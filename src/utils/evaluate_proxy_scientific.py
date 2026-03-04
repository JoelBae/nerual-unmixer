import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.proxy.ott import OTTProxy
from src.models.proxy.ott_stft import OTTSTFTProxy
from src.data.dataset import NeuralProxyDataset
from src.models.losses import SpectralLoss
from src.training.train_proxies import get_proxy_model

def review_spectral(effect_name, checkpoint_path, use_stft=False, use_stft_cond=False):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Scientific Performance Review: {effect_name.upper()} Proxy ---")
    
    # 1. Load Model dynamically based on flag
    model = get_proxy_model(effect_name, use_stft, use_stft_cond).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    # 2. Setup Data
    dataset = NeuralProxyDataset(effect_name=effect_name, dataset_dir=f"dataset/{effect_name}", split="val", preload=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 3. Individual Loss Components
    # We use our optimized SpectralLoss but track components
    spec_crit = SpectralLoss().to(device)
    
    sc_scores = []
    mag_scores = []
    env_scores = []
    
    print(f"   > Evaluating over {len(dataset)} validation samples...")
    
    with torch.no_grad():
        for dry, params, target, _ in tqdm(loader):
            dry = dry.to(device)
            target = target.to(device)
            params = params.to(device)
            
            # Predict
            pred = model(dry, params)
            
            # Spectral Components
            # SC = L2(diff) / L2(target)
            # Mag = L1(log diff)
            T = dry.shape[2]
            dry_flat = dry.reshape(-1, T)
            pred_flat = pred.reshape(-1, T)
            target_flat = target.reshape(-1, T)
            
            # STFT
            win = torch.hann_window(1024).to(device)
            p_stft = torch.stft(pred_flat, 1024, 256, window=win, return_complex=True)
            t_stft = torch.stft(target_flat, 1024, 256, window=win, return_complex=True)
            
            pm = torch.abs(p_stft) + 1e-7
            tm = torch.abs(t_stft) + 1e-7
            
            sc = torch.norm(tm - pm, p='fro') / torch.norm(tm, p='fro')
            mag = F.l1_loss(torch.log(pm), torch.log(tm))
            
            # Envelope Loss
            ep = F.avg_pool1d(pred.abs(), kernel_size=512, stride=256)
            et = F.avg_pool1d(target.abs(), kernel_size=512, stride=256)
            env = F.l1_loss(ep, et)
            
            sc_scores.append(sc.item())
            mag_scores.append(mag.item())
            env_scores.append(env.item())
            
    avg_sc = np.mean(sc_scores)
    avg_mag = np.mean(mag_scores)
    avg_env = np.mean(env_scores)
    
    print("\n" + "="*40)
    print("📈 FINAL SCIENTIFIC SCORECARD")
    print("="*40)
    print(f"1. Spectral Convergence: {avg_sc:.4f}  (Target: < 0.20)")
    print(f"2. Log-Magnitude Loss:  {avg_mag:.4f}  (Target: < 0.60)")
    print(f"3. Envelope Loss:       {avg_env:.4f}  (Target: < 0.15)")
    print("-" * 40)
    
    ready = (avg_sc < 0.20 and avg_mag < 0.60 and avg_env < 0.15)
    if ready:
        print("✅ STATUS: GOOD ENOUGH. Ready for Strategy B.")
    else:
        print("❌ STATUS: NEEDS IMPROVEMENT. Thresholds not met.")
    print("="*40)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, nargs='?', default="checkpoints/ott_proxy(2).pt")
    parser.add_argument("--effect", type=str, default="ott")
    parser.add_argument("--stft", action="store_true", help="Load the STFT version of the OTT Proxy")
    parser.add_argument("--stft_cond", action="store_true", help="Load the Conditioned STFT version of the OTT Proxy")
    args = parser.parse_args()
    
    review_spectral(args.effect, args.checkpoint, args.stft, args.stft_cond)
