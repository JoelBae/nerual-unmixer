import os
import sys
import warnings
warnings.filterwarnings("ignore", message="An output with one or more elements was resized*", category=UserWarning)

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import NeuralProxyDataset

# Import all our proxies
from src.models.proxy.ddsp_modules import OperatorProxy
from src.models.proxy.saturator import SaturatorProxy
from src.models.proxy.eq8 import EQEightProxy
from src.models.proxy.ott import OTTProxy
from src.models.proxy.reverb import ReverbProxy

def get_proxy_model(effect_name):
    name = effect_name.lower()
    if name == "operator":
        return OperatorProxy()
    elif name == "saturator":
        return SaturatorProxy()
    elif name == "eq8":
        return EQEightProxy()
    elif name == "ott":
        return OTTProxy()
    elif name == "reverb":
        return ReverbProxy()
    else:
        raise ValueError(f"Unknown effect name: {effect_name}")

def train_proxy(effect_name, dataset_dir, batch_size=32, epochs=100, lr=1e-3, device="cpu", patience=15, resume=False):
    print(f"--- Training {effect_name} Proxy ---")
    print(f"Device: {device} | Batch Size: {batch_size} | Epochs: {epochs} | Patience: {patience}")
    
    # 1. Setup Data
    duration = 1.0 if effect_name.lower() == "operator" else 2.0
    dataset_train = NeuralProxyDataset(effect_name=effect_name, dataset_dir=dataset_dir, split="train", preload=False, duration=duration)
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, 
        drop_last=True, pin_memory=(device=="mps" or device=="cuda"), num_workers=4 
    )
    
    dataset_val = NeuralProxyDataset(effect_name=effect_name, dataset_dir=dataset_dir, split="val", preload=False, duration=duration)
    dataloader_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, 
        drop_last=False, pin_memory=(device=="mps" or device=="cuda"), num_workers=4
    )
    
    print(f"Loaded {len(dataset_train)} Training / {len(dataset_val)} Validation samples")
    
    # 2. Setup Model, Loss, and Optimizer
    model = get_proxy_model(effect_name).to(device)
    if device == "cuda":
        model = model.half()
    
    if effect_name.lower() == "ott":
        from src.models.losses import DynamicsLoss
        criterion = DynamicsLoss().to(device)
    elif effect_name.lower() in ["reverb"]:
        from src.models.losses import VectorizedMultiScaleSpectralLoss
        criterion = VectorizedMultiScaleSpectralLoss().to(device)
    else:
        from src.models.losses import SpectralLoss
        criterion = SpectralLoss().to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Phase 2: Frozen Dynamics
    # If training only the spectral character, we freeze the analytical parameters
    if hasattr(args, 'phase2') and args.phase2:
        print("❄️  Phase 2: Freezing analytical dynamics, training ONLY spectral character.")
        for name, param in model.named_parameters():
            if "residual_net" not in name:
                param.requires_grad = False
        # Update optimizer to only see active params
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    if effect_name.lower() in ["ott", "reverb"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        use_cosine = True
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        use_cosine = False
    
    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/{effect_name}_proxy.pt"
    
    if resume and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        print(f"✅ Loaded checkpoint from {save_path}.")
        
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Accelerated Training (FP16/AMP)
    use_amp = (device == "mps" or device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{epochs}")
        
        for input_audio, params, target_audio in progress_bar:
            input_audio = input_audio.to(device)
            params = params.to(device)
            target_audio = target_audio.to(device)
            
            optimizer.zero_grad()
            
            # Use Mixed precision for speed and memory efficiency
            if device == "mps":
                input_audio = input_audio.half()
                # model.half() must be called once outside
                with torch.autocast(device_type="mps", enabled=True, dtype=torch.float16):
                    if effect_name.lower() == "operator":
                        pred_audio = model(params)
                    else:
                        pred_audio = model(input_audio, params)
                    loss = criterion(pred_audio, target_audio)
            elif device == "cuda":
                with torch.autocast(device_type="cuda", enabled=True):
                    if effect_name.lower() == "operator":
                        pred_audio = model(params)
                    else:
                        pred_audio = model(input_audio, params)
                    loss = criterion(pred_audio, target_audio)
            else:
                if effect_name.lower() == "operator":
                    pred_audio = model(params)
                else:
                    pred_audio = model(input_audio, params)
                loss = criterion(pred_audio, target_audio)
            
            if torch.isnan(loss):
                continue
                
            if device == "cuda":
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_loss / len(dataloader_train)
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for input_audio, params, target_audio in dataloader_val:
                input_audio = input_audio.to(device)
                params = params.to(device)
                target_audio = target_audio.to(device)
                
                if device == "mps":
                    input_audio = input_audio.half()
                    with torch.autocast(device_type="mps", enabled=True, dtype=torch.float16):
                        pred_audio = model(input_audio, params)
                        loss = criterion(pred_audio, target_audio)
                elif device == "cuda":
                    with torch.autocast(device_type="cuda", enabled=True):
                        pred_audio = model(input_audio, params)
                        loss = criterion(pred_audio, target_audio)
                else:
                    pred_audio = model(input_audio, params)
                    loss = criterion(pred_audio, target_audio)
                    
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(dataloader_val) if len(dataloader_val) > 0 else 0.0
        print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f} Val={avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"🌟 Saved best model!")
        else:
            patience_counter += 1
            
        if use_cosine:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            if optimizer.param_groups[0]['lr'] > old_lr:
                patience_counter = 0
        else:
            scheduler.step(avg_val_loss)
            
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--effect", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--phase2", action="store_true", help="Freeze dynamics, train only spectral correction.")
    
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # OTT is very memory-intensive, cap batch size
    effective_batch_size = args.batch_size
    if args.effect.lower() == "ott" and effective_batch_size > 32 and device == "cuda":
        print(f"⚠️  Capping OTT batch size to 32 for memory safety (requested {args.batch_size})")
        effective_batch_size = 32

    train_proxy(
        effect_name=args.effect, 
        dataset_dir=f"dataset/{args.effect.lower()}",
        batch_size=effective_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        patience=args.patience,
        resume=args.resume
    )

