import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.inverter import NeuralInverter
from src.data.dataset import NeuralProxyDataset
from src.models.proxy.chain import ProxyChainer
from src.models.losses import DynamicsLoss
from src.utils.normalization import denormalize_params

def assemble_full_params(mdn_out, wave_idx, batch_size, device):
    """Helper to construct the 63-param vector for the chainer."""
    # Create a placeholder for the full parameter vector
    full_params = torch.zeros(batch_size, 63, device=device)
    
    # Operator (15 cont + 1 cat)
    full_params[:, 0] = mdn_out[:, 0]            # Transpose
    full_params[:, 1] = wave_idx.float()         # Ground truth wave index
    full_params[:, 2:16] = mdn_out[:, 1:15]      # Filter, Envs...
    
    # Saturator (5 cont)
    full_params[:, 16:21] = mdn_out[:, 15:20]
    
    # EQ Eight (32 cont)
    full_params[:, 21:53] = mdn_out[:, 20:52]
    
    # OTT (7 cont)
    full_params[:, 53:60] = mdn_out[:, 52:59]
    
    # Reverb (3 cont)
    full_params[:, 60:63] = mdn_out[:, 59:62]
    
    return full_params


def train_inverter_audio(args):
    print(f"--- Training Full Inverter with Composite Audio + Categorical Loss ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataset & Loader
    dataset = NeuralProxyDataset(effect_name=args.effect, dataset_dir=args.dataset_dir, split="train", preload=False)
    val_dataset = NeuralProxyDataset(effect_name=args.effect, dataset_dir=args.dataset_dir, split="val", preload=False)
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Models
    model = NeuralInverter(latent_dim=args.latent_dim).to(device)
    if args.resume_encoder and os.path.exists(args.resume_encoder):
        model.load_state_dict(torch.load(args.resume_encoder, map_location=device))
        print(f"✅ Resumed Encoder from {args.resume_encoder}")
    
    chainer = ProxyChainer().to(device)
    chainer.load_checkpoints()
    chainer.eval()
    for p in chainer.parameters():
        p.requires_grad = False
    
    # 3. Optimizer & Losses
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_audio_fn = DynamicsLoss().to(device)
    loss_wave_fn = nn.CrossEntropyLoss()
    loss_order_fn = nn.CrossEntropyLoss()
    
    # Loss weights
    w_audio = args.w_audio
    w_wave = args.w_wave
    w_order = args.w_order
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_audio_loss, total_wave_loss, total_order_loss = 0, 0, 0, 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for _, true_params, target_audio, true_order_idx in progress_bar:
            target_audio = target_audio.to(device)
            true_params = true_params.to(device)
            true_order_idx = true_order_idx.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(target_audio)
            
            # --- Categorical Losses ---
            true_wave_idx = true_params[:, 1].long()
            loss_wave = loss_wave_fn(outputs['wave_logits'], true_wave_idx)
            loss_order = loss_order_fn(outputs['order_logits'], true_order_idx)

            # --- Audio Reconstruction Loss ---
            pi, mu = outputs['pi'], outputs['mu']
            best_g = torch.argmax(pi, dim=1)
            mu_best = mu[torch.arange(target_audio.shape[0]), best_g]

            full_params_norm = assemble_full_params(mu_best, true_wave_idx, target_audio.shape[0], device)
            pred_params_raw = denormalize_params(full_params_norm)
            
            reconstructed_audio = chainer.forward_flat(pred_params_raw, order_idx=true_order_idx)
            
            loss_audio = loss_audio_fn(reconstructed_audio, target_audio)
            
            # --- Combined Loss ---
            loss = (w_audio * loss_audio) + (w_wave * loss_wave) + (w_order * loss_order)
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_audio_loss += loss_audio.item()
            total_wave_loss += loss_wave.item()
            total_order_loss += loss_order.item()
            
            progress_bar.set_postfix({
                'L': f"{loss.item():.3f}",
                'L_aud': f"{loss_audio.item():.3f}",
                'L_wav': f"{loss_wave.item():.3f}",
                'L_ord': f"{loss_order.item():.3f}"
            })
            
        # 5. Validation
        model.eval()
        val_loss, val_audio_loss, val_wave_loss, val_order_loss = 0, 0, 0, 0
        with torch.no_grad():
            for _, true_params, target_audio, true_order_idx in val_loader:
                target_audio = target_audio.to(device)
                true_params = true_params.to(device)
                true_order_idx = true_order_idx.to(device)

                outputs = model(target_audio)
                
                true_wave_idx = true_params[:, 1].long()
                loss_wave = loss_wave_fn(outputs['wave_logits'], true_wave_idx)
                loss_order = loss_order_fn(outputs['order_logits'], true_order_idx)
                
                pi, mu = outputs['pi'], outputs['mu']
                best_g = torch.argmax(pi, dim=1)
                mu_best = mu[torch.arange(target_audio.shape[0]), best_g]
                full_params_norm = assemble_full_params(mu_best, true_wave_idx, target_audio.shape[0], device)
                pred_params_raw = denormalize_params(full_params_norm)
                reconstructed_audio = chainer.forward_flat(pred_params_raw, order_idx=true_order_idx)
                loss_audio = loss_audio_fn(reconstructed_audio, target_audio)
                
                loss = (w_audio * loss_audio) + (w_wave * loss_wave) + (w_order * loss_order)

                val_loss += loss.item()
                val_audio_loss += loss_audio.item()
                val_wave_loss += loss_wave.item()
                val_order_loss += loss_order.item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}: Train Loss={avg_train:.4f} | Val Loss={avg_val:.4f}")
        print(f"  Components (Val): Audio={val_audio_loss/len(val_loader):.4f}, Wave={val_wave_loss/len(val_loader):.4f}, Order={val_order_loss/len(val_loader):.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"checkpoints/inverter_{args.effect}_audio_best.pt")
            print("🌟 New best audio inverter saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the full audio inverter model.")
    parser.add_argument("--effect", type=str, default="full_chain", help="Must be 'full_chain' for this script.")
    parser.add_argument("--dataset_dir", type=str, default="dataset/full_chain", help="Path to the dataset.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--resume_encoder", type=str, default=None)
    parser.add_argument("--w_audio", type=float, default=1.0, help="Weight for audio reconstruction loss.")
    parser.add_argument("--w_wave", type=float, default=0.5, help="Weight for categorical wave prediction loss.")
    parser.add_argument("--w_order", type=float, default=0.5, help="Weight for categorical order prediction loss.")
    
    args = parser.parse_args()
    if args.effect != "full_chain":
        print("Warning: This script is intended for the 'full_chain' effect. Results may be unexpected.")
        args.dataset_dir = f"dataset/{args.effect}"

    os.makedirs("checkpoints", exist_ok=True)
    train_inverter_audio(args)
