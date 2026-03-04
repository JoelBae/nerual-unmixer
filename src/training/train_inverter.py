import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.inverter import NeuralInverter
from src.data.dataset import NeuralProxyDataset
from src.data.proxy_dataset import OnTheFlyProxyDataset
from src.data.augment import AudioAugmentor
from src.models.proxy.chain import ProxyChainer
from src.models.heads.mdn import mdn_loss
from src.utils.normalization import normalize_params

def train_inverter(args):
    print(f"--- Training Neural Inverter (Ears) for {args.effect} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 0. Setup Logging
    writer = SummaryWriter(log_dir=args.log_dir)
    global_step = 0
    
    # 1. Dataset & Loader
    if args.use_proxy_data:
        print(f"--- Using OnTheFlyProxyDataset (Sim-to-Real) ---")
        dataset = OnTheFlyProxyDataset(virtual_length=args.proxy_virtual_size, effect_name=args.effect)
        val_dataset = OnTheFlyProxyDataset(virtual_length=max(args.proxy_virtual_size // 10, 100), effect_name=args.effect)
        augmentor = AudioAugmentor(sample_rate=44100)
    else:
        dataset = NeuralProxyDataset(
            effect_name=args.effect, 
            dataset_dir=args.dataset_dir, 
            split="train",
            preload=True,
            augment=True
        )
        val_dataset = NeuralProxyDataset(
            effect_name=args.effect, 
            dataset_dir=args.dataset_dir, 
            split="val",
            preload=True,
            augment=False
        )
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Proxy Chainer
    chainer = None
    if args.use_proxy_data:
        chainer = ProxyChainer(sr=44100).to(device)
        chainer.load_checkpoints()
        chainer.eval()
        for param in chainer.parameters():
            param.requires_grad = False
    
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
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            dry, params, target = batch[0], batch[1], batch[2]
            order_idx = batch[3].to(device) if len(batch) > 3 else None
            true_params = params.to(device)
            
            if args.use_proxy_data and chainer is not None:
                with torch.no_grad():
                    # Generate audio dynamically from Proxies on the GPU
                    target_audio = chainer.forward_flat(true_params, order_idx=order_idx)
                    target_audio = augmentor(target_audio)
            else:
                target_audio = target.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(target_audio)
            
            # Normalize labels for MDN (skipping categorical)
            norm_true_params = normalize_params(true_params)
            
            # 1. Extract Continuous Label (MDN)
            # Must match the order in NeuralInverter.predict assembly!
            cont_list = []
            cont_list.append(norm_true_params[:, 0:1])   # Transpose
            cont_list.append(norm_true_params[:, 2:17])  # Operator (minus wave)
            cont_list.append(norm_true_params[:, 18:21]) # Sat (minus type)
            for band in range(8):
                idx = 21 + (band * 4)
                cont_list.append(norm_true_params[:, idx+1:idx+4]) # Freq, Gain, Q
            cont_list.append(norm_true_params[:, 53:63]) # OTT, Reverb
            true_cont = torch.cat(cont_list, dim=1)
            
            # 2. Extract Categorical Labels
            true_wave = true_params[:, 1].long()
            true_sat_type = true_params[:, 17].long()
            true_eq8_types = [true_params[:, 21 + i*4].long() for i in range(8)]
            
            # 3. Calculate Losses
            loss_mdn = mdn_loss(outputs['pi'], outputs['mu'], outputs['sigma'], true_cont)
            loss_wave = ce_loss(outputs['wave_logits'], true_wave)
            loss_sat = ce_loss(outputs['sat_type_logits'], true_sat_type)
            loss_eq8 = sum([ce_loss(outputs['eq8_type_logits'][i], true_eq8_types[i]) for i in range(8)])
            
            loss = loss_mdn + loss_wave + loss_sat + loss_eq8
            
            loss.backward()
            optimizer.step()
            
            # Logging
            train_loss += loss.item()
            if global_step % 10 == 0:
                writer.add_scalar("Train/Total_Loss", loss.item(), global_step)
                writer.add_scalar("Train/MDN_Loss", loss_mdn.item(), global_step)
                writer.add_scalar("Train/Wave_Loss", loss_wave.item(), global_step)
            
            global_step += 1
            
        # 4. Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                dry, params, target = batch[0], batch[1], batch[2]
                order_idx = batch[3].to(device) if len(batch) > 3 else None
                true_params = params.to(device)
                
                if args.use_proxy_data and chainer is not None:
                    target_audio = chainer.forward_flat(true_params, order_idx=order_idx)
                else:
                    target_audio = target.to(device)
                    
                outputs = model(target_audio)
                
                norm_true_params = normalize_params(true_params)
                
                # Extract Cont
                cont_list = []
                cont_list.append(norm_true_params[:, 0:1])
                cont_list.append(norm_true_params[:, 2:17])
                cont_list.append(norm_true_params[:, 18:21])
                for band in range(8):
                    idx = 21 + (band * 4)
                    cont_list.append(norm_true_params[:, idx+1:idx+4])
                cont_list.append(norm_true_params[:, 53:63])
                true_cont = torch.cat(cont_list, dim=1)
                
                # Extract Cat
                true_wave = true_params[:, 1].long()
                true_sat_type = true_params[:, 17].long()
                true_eq8_types = [true_params[:, 21 + i*4].long() for i in range(8)]
                
                # Calculate Losses
                loss_mdn = mdn_loss(outputs['pi'], outputs['mu'], outputs['sigma'], true_cont)
                loss_wave = ce_loss(outputs['wave_logits'], true_wave)
                loss_sat = ce_loss(outputs['sat_type_logits'], true_sat_type)
                loss_eq8 = sum([ce_loss(outputs['eq8_type_logits'][i], true_eq8_types[i]) for i in range(8)])
                
                val_loss += (loss_mdn + loss_wave + loss_sat + loss_eq8).item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        writer.add_scalar("Val/Total_Loss", avg_val, epoch)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train:.4f} | Val Loss = {avg_val:.4f}")
        
        # Save checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"checkpoints/inverter_{args.effect}_best.pt")
            print("  New best model saved!")
            
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--effect", type=str, default="ott")
    parser.add_argument("--dataset_dir", type=str, default="dataset/ott")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--use_proxy_data", action="store_true", help="Use infinite proxy-generated data on-the-fly")
    parser.add_argument("--proxy_virtual_size", type=int, default=100000)
    parser.add_argument("--log_dir", type=str, default="runs/inverter_param")
    
    args = parser.parse_args()
    os.makedirs("checkpoints", exist_ok=True)
    train_inverter(args)
