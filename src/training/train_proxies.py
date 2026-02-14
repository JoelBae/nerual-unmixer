import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Optional W&B
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not found. Experiment tracking disabled.")

from src.data.dataset import ProxyDataset, get_param_names
from src.training.loss import STFTLoss

# Import Proxy Models
try:
    from src.models.proxy.modules import (
        ProxySaturator, ProxyEQ8, ProxyOTT, ProxyPhaser, ProxyReverb
    )
    # Generic wrapper or mapping
    MODEL_MAP = {
        "saturator": ProxySaturator,
        "eq8": ProxyEQ8,
        "ott": ProxyOTT,
        "phaser": ProxyPhaser,
        "reverb": ProxyReverb,
        # "operator": ProxyOperator # Operator might need a specific proxy if we treat it as Source->Audio
    }
except ImportError as e:
    print(f"Error importing models: {e}")
    MODEL_MAP = {}

# Simple fallback for Operator or missing models
class GenericProxy(nn.Module):
    def __init__(self, param_dim):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
    def forward(self, x, params):
        return self.conv(x)

class MultiScaleSpectralLoss(nn.Module):
    def __init__(self, fft_sizes=[2048, 1024, 512, 256, 128, 64],
                 hop_sizes=[512, 256, 128, 64, 32, 16],
                 win_lengths=[2048, 1024, 512, 256, 128, 64]):
        super().__init__()
        self.stft_losses = nn.ModuleList()
        for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(STFTLoss(fs, hs, wl))

    def forward(self, pred, target):
        loss = 0.0
        for f in self.stft_losses:
            loss += f(pred, target)
        return loss

def train(args):
    # 1. Load Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    effect = args.effect
    # Allow running even if effect not in config map (partial support)
    eff_config = config["effects"].get(effect, {})
    if not eff_config and not args.dry_run:
         print(f"Warning: Effect '{effect}' Config not found. Using defaults.")
         eff_config = {"dataset_path": f"dataset/{effect}", "param_names": None}
         
    train_config = config["training"]

    # 2. Setup W&B
    if HAS_WANDB:
        wandb.init(
            project=config.get("wandb", {}).get("project", "neural-unmixer"),
            config={**train_config, **eff_config, "effect": effect},
            mode="disabled" if args.dry_run else "online"
        )

    # 3. Data
    dataset_path = eff_config.get("dataset_path", f"dataset/{effect}")
    if not os.path.exists(dataset_path):
         print(f"Warning: Dataset path {dataset_path} does not exist.")
         if not args.dry_run:
             # Create dummy dataset/paths or error out
             raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    param_names = get_param_names(effect)
    
    # Create Dataset
    train_loader = None
    val_loader = None
    
    try:
        if os.path.exists(os.path.join(dataset_path, "metadata.json")):
            full_dataset = ProxyDataset(
                dataset_dir=dataset_path, 
                effect_name=effect, 
                param_names=param_names
            )
            
            # Split Train/Val
            val_split = train_config.get("val_split", 0.1)
            val_size = int(len(full_dataset) * val_split)
            train_size = len(full_dataset) - val_size
            
            # handle edge case of small dataset (e.g. 1 sample)
            if val_size == 0 and len(full_dataset) > 1: val_size = 1
            if train_size == 0: train_size = len(full_dataset)
                
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True, num_workers=train_config.get("num_workers", 0))
            val_loader = DataLoader(val_dataset, batch_size=train_config["batch_size"], shuffle=False, num_workers=train_config.get("num_workers", 0))
        else:
            if not args.dry_run: raise FileNotFoundError("metadata.json not found")
            print("Usage: No metadata found. Proceeding with dummy/empty loaders for dry-run check.")
            train_loader = []
            val_loader = []

    except Exception as e:
        if args.dry_run:
            print(f"Dataset load failed ({e}), skipping data setup for dry-run.")
            train_loader = []
            val_loader = []
        else:
            raise e

    # 4. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if effect in MODEL_MAP:
        model = MODEL_MAP[effect]().to(device)
    else:
        # Fallback or generic
        print(f"Model for {effect} not found in map. Using GenericProxy.")
        param_dim = len(param_names) if param_names else 10 # Default
        model = GenericProxy(param_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=float(train_config["learning_rate"]))
    criterion = MultiScaleSpectralLoss().to(device)

    # 5. Training Loop
    epochs = 1 if args.dry_run else train_config["epochs"]
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # If dry-run and no data, skip loop but print "Training..."
        if not train_loader:
            print("No data loaded. Skipping loop.")
            break
            
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for dry_audio, params, wet_audio in pbar:
            dry_audio = dry_audio.to(device)
            params = params.to(device)
            wet_audio = wet_audio.to(device)
            
            optimizer.zero_grad()
            
            # Ensure shapes: Audio [B, 1, L]
            if dry_audio.ndim == 2: dry_audio = dry_audio.unsqueeze(1)
            if wet_audio.ndim == 2: wet_audio = wet_audio.unsqueeze(1)

            pred_audio = model(dry_audio, params)
            
            loss = criterion(pred_audio, wet_audio)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        if HAS_WANDB: wandb.log({"train_loss": avg_train_loss, "epoch": epoch})
        
        # Validation
        model.eval()
        val_loss = 0.0
        if val_loader:
            with torch.no_grad():
                for dry_audio, params, wet_audio in val_loader:
                    dry_audio = dry_audio.to(device)
                    params = params.to(device)
                    wet_audio = wet_audio.to(device)
                    
                    if dry_audio.ndim == 2: dry_audio = dry_audio.unsqueeze(1)
                    if wet_audio.ndim == 2: wet_audio = wet_audio.unsqueeze(1)
                    
                    pred_audio = model(dry_audio, params)
                    loss = criterion(pred_audio, wet_audio)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            if HAS_WANDB: wandb.log({"val_loss": avg_val_loss})
            print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")
        
    # 6. Save Model
    if not args.dry_run or (args.dry_run and train_loader):
        os.makedirs("src/models/weights", exist_ok=True)
        save_path = f"src/models/weights/{effect}_proxy.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        if HAS_WANDB: wandb.save(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--effect", type=str, required=True, help="Effect name to train (e.g. saturator)")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Run a single pass to check pipeline")
    
    args = parser.parse_args()
    train(args)
