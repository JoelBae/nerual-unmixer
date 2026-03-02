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

def train_inverter_audio(args):
    print(f"--- Strategy B: Training Inverter with Audio Loss for {args.effect} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataset & Loader
    # Using 4 workers and pin_memory for performance
    dataset = NeuralProxyDataset(
        effect_name=args.effect, 
        dataset_dir=args.dataset_dir, 
        split="train",
        preload=False
    )
    val_dataset = NeuralProxyDataset(
        effect_name=args.effect, 
        dataset_dir=args.dataset_dir, 
        split="val",
        preload=False
    )
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Models
    # Encoder
    model = NeuralInverter(latent_dim=args.latent_dim).to(device)
    if args.resume_encoder and os.path.exists(args.resume_encoder):
        model.load_state_dict(torch.load(args.resume_encoder, map_location=device, weights_only=True))
        print(f"✅ Resumed Encoder from {args.resume_encoder}")
    
    # Proxy (Differentiable DSL Chain)
    chainer = ProxyChainer().to(device)
    chainer.load_checkpoints() # Loads wave_table.pt, ott_proxy.pt, etc.
    chainer.eval()
    for p in chainer.parameters():
        p.requires_grad = False
    
    # 3. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = DynamicsLoss().to(device)
    
    # 4. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for dry_audio, true_params, target_audio in progress_bar:
            dry_audio = dry_audio.to(device)
            target_audio = target_audio.to(device)
            # true_params is only used for logging/ref if needed, not for loss in Strategy B
            
            optimizer.zero_grad()
            
            # Step 1: Predict Params (MDN Means)
            outputs = model(target_audio)
            
            # Use the mean of the most likely Gaussian for the most stable "predicted knobs"
            # Shape of mu: (batch, num_gaussians, out_features)
            # Shape of pi: (batch, num_gaussians)
            best_g = torch.argmax(outputs['pi'], dim=1) # (batch,)
            mu_best = outputs['mu'][torch.arange(target_audio.shape[0]), best_g] # (batch, 22)
            
            # Step 2: Denormalize to Ableton Ranges
            # Need to assemble full 23-param vector (Transpose, Wave, ... OTT)
            # Strategy B focuses on OTT for now since we have the proxy.
            # We assume Wave and Transpose are either known or we just use the predicted ones.
            # Note: Wave is index 1, skip it for denorm loop in a simplified way
            
            full_params_norm = torch.zeros(target_audio.shape[0], 23, device=device)
            full_params_norm[:, 0] = mu_best[:, 0] # Transpose
            # Wave logits are categorical, we take argmax but it's not differentiable.
            # However, the OTT proxy doesn't use index 1, so it's fine.
            full_params_norm[:, 1] = torch.argmax(outputs['wave_logits'], dim=1).float()
            full_params_norm[:, 2:23] = mu_best[:, 1:] # Filter, Envs, OTT
            
            pred_params_raw = denormalize_params(full_params_norm)
            
            # Step 3: Differentiable Pass through Proxy Chain (Operator -> OTT)
            # The chainer internally handles Generator (Op) vs Processor (OTT)
            reconstructed_audio = chainer.forward_flat(pred_params_raw, sequence=['operator', 'ott'])
            
            # Step 4: Audio-Domain Loss
            loss = criterion(reconstructed_audio, target_audio)
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'AudioLoss': f"{loss.item():.4f}"})
            
        # 5. Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for dry_audio, true_params, target_audio in val_loader:
                dry_audio = dry_audio.to(device)
                target_audio = target_audio.to(device)
                
                outputs = model(target_audio)
                best_g = torch.argmax(outputs['pi'], dim=1)
                mu_best = outputs['mu'][torch.arange(target_audio.shape[0]), best_g]
                
                full_params_norm = torch.zeros(target_audio.shape[0], 23, device=device)
                full_params_norm[:, 0] = mu_best[:, 0]
                full_params_norm[:, 1] = torch.argmax(outputs['wave_logits'], dim=1).float()
                full_params_norm[:, 2:23] = mu_best[:, 1:]
                pred_params_raw = denormalize_params(full_params_norm)
                
                reconstructed_audio = chainer.forward_flat(pred_params_raw, sequence=['operator', 'ott'])
                loss = criterion(reconstructed_audio, target_audio)
                val_loss += loss.item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Audio Loss = {avg_train:.4f} | Val Audio Loss = {avg_val:.4f}")
        
        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"checkpoints/inverter_{args.effect}_audio_best.pt")
            print("🌟 New best audio inverter saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--effect", type=str, default="ott")
    parser.add_argument("--dataset_dir", type=str, default="dataset/ott")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4) # Lower LR for Strategy B
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--proxy_path", type=str, default="checkpoints/ott_proxy.pt")
    parser.add_argument("--resume_encoder", type=str, default=None)
    
    args = parser.parse_args()
    os.makedirs("checkpoints", exist_ok=True)
    train_inverter_audio(args)
