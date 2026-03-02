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
from src.data.dataset import NeuralProxyDataset
from src.models.losses import SpectralLoss

def evaluate_scientific(checkpoint_path="checkpoints/ott_proxy(2).pt", batch_size=32):
    print(f"--- Scientific Performance Review: OTT Proxy ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load Model
    model = OTTProxy().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    # 2. Setup Data
    dataset = NeuralProxyDataset(effect_name="ott", dataset_dir="dataset/ott", split="val", preload=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 3. Individual Loss Components
    # We use our optimized SpectralLoss but track components
    spec_crit = SpectralLoss().to(device)
    
    sc_scores = []
    mag_scores = []
    env_scores = []
    
    print(f"   > Evaluating over {len(dataset)} validation samples...")
    
    with torch.no_grad():
        for dry, params, target in tqdm(loader):
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
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ott_proxy(2).pt"
    evaluate_scientific(checkpoint_path=path)
