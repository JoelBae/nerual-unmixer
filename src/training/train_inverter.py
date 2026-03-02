import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.inverter import NeuralInverter
from src.data.dataset import NeuralProxyDataset
from src.models.heads.mdn import mdn_loss
from src.utils.normalization import normalize_params

def train_inverter(args):
    print(f"--- Training Neural Inverter (Ears) for {args.effect} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataset & Loader
    dataset = NeuralProxyDataset(
        effect_name=args.effect, 
        dataset_dir=args.dataset_dir, 
        split="train",
        preload=True
    )
    val_dataset = NeuralProxyDataset(
        effect_name=args.effect, 
        dataset_dir=args.dataset_dir, 
        split="val",
        preload=True
    )
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Model
    model = NeuralInverter(latent_dim=args.latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Categorical Loss
    ce_loss = nn.CrossEntropyLoss()
    
    # 3. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        for dry, params, target in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # For the Inverter:
            # Input = WET Audio (target)
            # Label = Params
            target_audio = target.to(device)
            true_params = params.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(target_audio)
            
            # Normalize labels to 0-1 for MDN (except categorical)
            norm_true_params = normalize_params(true_params)
            
            # Split into Continuous and Categorical
            true_wave = true_params[:, 1].long() # Categorical uses raw index
            true_cont = torch.cat([norm_true_params[:, 0:1], norm_true_params[:, 2:23]], dim=1)
            
            # Losses
            loss_mdn = mdn_loss(outputs['pi'], outputs['mu'], outputs['sigma'], true_cont)
            loss_ce = ce_loss(outputs['wave_logits'], true_wave)
            
            loss = loss_mdn + loss_ce
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # 4. Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for dry, params, target in val_loader:
                target_audio = target.to(device)
                true_params = params.to(device)
                
                outputs = model(target_audio)
                
                norm_true_params = normalize_params(true_params)
                true_wave = true_params[:, 1].long()
                true_cont = torch.cat([norm_true_params[:, 0:1], norm_true_params[:, 2:23]], dim=1)
                
                loss_mdn = mdn_loss(outputs['pi'], outputs['mu'], outputs['sigma'], true_cont)
                loss_ce = ce_loss(outputs['wave_logits'], true_wave)
                val_loss += (loss_mdn + loss_ce).item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train:.4f} | Val Loss = {avg_val:.4f}")
        
        # Save checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"checkpoints/inverter_{args.effect}_best.pt")
            print("  New best model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--effect", type=str, default="ott")
    parser.add_argument("--dataset_dir", type=str, default="dataset/ott")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=512)
    
    args = parser.parse_args()
    os.makedirs("checkpoints", exist_ok=True)
    train_inverter(args)
