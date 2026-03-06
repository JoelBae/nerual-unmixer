import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchaudio
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import io
import numpy as np
import glob
from google.cloud import storage

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.inverter import NeuralInverter
from src.data.dataset import NeuralProxyDataset
from src.data.proxy_dataset import OnTheFlyProxyDataset
from src.data.augment import AudioAugmentor
from src.models.proxy.chain import ProxyChainer
from src.models.losses import DynamicsLoss
from src.models.heads.mdn import MDNHead
from src.utils.normalization import denormalize_params

def assemble_full_params(mdn_out, wave_idx, sat_type_idx, eq8_type_idxs, batch_size, device):
    """
    Reconstruct the 63-param flat vector from the MDN's 53 continuous outputs
    and the categorical head predictions.
    
    MDN layout (53 features, matching train_inverter.py cont_list order):
      [0]      Transpose
      [1:15]   Operator continuous (FilterFreq, FilterRes, Fe*, Pe*, Ae*) = 14
      [15:18]  Saturator continuous (WS Curve, WS Depth, Dry/Wet) = 3
      [18:42]  EQ8 continuous (8 bands x [Freq, Gain, Q]) = 24
      [42:49]  OTT continuous (Amount + 6 Thresholds) = 7
      [49:52]  Reverb continuous (Decay, Size, Dry/Wet) = 3
    Total = 1 + 14 + 3 + 24 + 7 + 3 = 52
    
    Full param layout (63):
      [0]      Transpose
      [1]      Wave (categorical)
      [2:16]   Operator continuous (14)
      [16]     Sat Drive (continuous)  -- wait, let me re-check normalization.py
      [17]     Sat Type (categorical)
      [18:21]  Sat continuous (3)
      [21-52]  EQ8: 8 bands x [Type(cat), Freq, Gain, Q] (32)
      [53:60]  OTT (7)
      [60:63]  Reverb (3)
    """
    full_params = torch.zeros(batch_size, 63, device=device)
    idx = 0
    
    # Operator
    full_params[:, 0] = mdn_out[:, idx]; idx += 1          # Transpose
    full_params[:, 1] = wave_idx.float()                   # Wave (categorical)
    full_params[:, 2:16] = mdn_out[:, idx:idx+14]; idx += 14  # Operator continuous
    
    # Saturator
    full_params[:, 16] = mdn_out[:, idx]; idx += 1         # Drive
    full_params[:, 17] = sat_type_idx.float()               # Type (categorical)
    full_params[:, 18:21] = mdn_out[:, idx:idx+3]; idx += 3  # WS Curve, Depth, DryWet
    
    # EQ Eight (8 bands x 4 params, type is categorical)
    for band in range(8):
        base = 21 + (band * 4)
        full_params[:, base] = eq8_type_idxs[band].float()          # Type (categorical)
        full_params[:, base+1:base+4] = mdn_out[:, idx:idx+3]; idx += 3  # Freq, Gain, Q
    
    # OTT (7 continuous)
    full_params[:, 53:60] = mdn_out[:, idx:idx+7]; idx += 7
    
    # Reverb (3 continuous)
    full_params[:, 60:63] = mdn_out[:, idx:idx+3]; idx += 3
    
    return full_params

def plot_spectrogram(audio, title="Spectrogram"):
    """Helper to create a spectrogram image for TensorBoard."""
    # audio: (channels, time)
    audio = audio.cpu()
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=1024)(audio)
    # Average channels for visualization if stereo
    if spectrogram.shape[0] > 1:
        spectrogram = spectrogram.mean(dim=0)
    else:
        spectrogram = spectrogram.squeeze(0)
        
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(torch.log10(spectrogram + 1e-9), aspect='auto', origin='lower')
    ax.set_title(title)
    plt.colorbar(img, ax=ax)
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    # Convert buffer to tensor
    image = plt.imread(buf)
    # Ensure 3 channels and float32 in [0, 1]
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).float() # (3, H, W)

def sync_to_gcs(local_dir, remote_path):
    """Syncs local tensorboard logs to GCS using the Python SDK."""
    if not remote_path.startswith("gs://"):
        return
        
    try:
        # gs://bucket/path -> bucket, path
        parts = remote_path[5:].split('/', 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Upload every file in the local log dir
        for local_file in glob.glob(f"{local_dir}/**", recursive=True):
            if os.path.isfile(local_file):
                # Calculate relative path for remote
                rel_path = os.path.relpath(local_file, local_dir)
                remote_file = os.path.join(prefix, rel_path)
                
                blob = bucket.blob(remote_file)
                blob.upload_from_filename(local_file)
    except Exception as e:
        print(f"⚠️ GCS Sync Failed: {e}")


def train_inverter_audio(args):
    print(f"--- Training Full Inverter with Composite Audio + Categorical Loss ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 0. Setup Logging
    # Use local logging + periodic GCS sync to avoid clobbering/latency issues
    is_gcs = args.log_dir.startswith("gs://")
    local_log_dir = "/tmp/tb_logs" if is_gcs else args.log_dir
    os.makedirs(local_log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=local_log_dir)
    global_step = 0
    
    # 1. Dataset & Loader
    if args.use_proxy_data:
        print(f"--- Using OnTheFlyProxyDataset (Sim-to-Real) ---")
        dataset = OnTheFlyProxyDataset(virtual_length=args.proxy_virtual_size, effect_name=args.effect)
        val_dataset = OnTheFlyProxyDataset(virtual_length=max(args.proxy_virtual_size // 10, 100), effect_name=args.effect)
        augmentor = AudioAugmentor(sample_rate=44100)
    else:
        dataset = NeuralProxyDataset(effect_name=args.effect, dataset_dir=args.dataset_dir, split="train", preload=False)
        val_dataset = NeuralProxyDataset(effect_name=args.effect, dataset_dir=args.dataset_dir, split="val", preload=False)
        augmentor = None
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0 if args.use_proxy_data else 4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0 if args.use_proxy_data else 4, pin_memory=True)
    
    # 2. Models
    model = NeuralInverter(latent_dim=args.latent_dim).to(device)
    if args.resume_encoder and os.path.exists(args.resume_encoder):
        model.load_state_dict(torch.load(args.resume_encoder, map_location=device))
        print(f"✅ Resumed Encoder from {args.resume_encoder}")
        
    start_epoch = 0
    start_step = 0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"--- 🔄 Resuming from Checkpoint: {args.resume_checkpoint} ---")
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            start_epoch = ckpt.get('epoch', 0)
            start_step = ckpt.get('step', 0)
            print(f"✅ Recovered Model State (Epoch {start_epoch}, Step {start_step})")
        else:
            # Fallback for old checkpoint format
            model.load_state_dict(ckpt)
            print("✅ Recovered Model State (Raw Dict)")
    
    chainer = ProxyChainer().to(device)
    chainer.load_checkpoints()
    chainer.eval()
    for p in chainer.parameters():
        p.requires_grad = False
    
    # 3. Optimizer & Losses
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint) and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f"✅ Recovered Optimizer State")
        
    loss_audio_fn = DynamicsLoss().to(device)
    loss_wave_fn = nn.CrossEntropyLoss()
    loss_order_fn = nn.CrossEntropyLoss()
    
    # Validation helper
    def get_accuracy(logits, targets):
        pred = torch.argmax(logits, dim=1)
        return (pred == targets).float().mean().item()
    
    # Loss weights
    w_audio = args.w_audio
    w_wave = args.w_wave
    w_order = args.w_order
    w_sat_type = getattr(args, 'w_sat_type', 0.5)
    w_eq8_type = getattr(args, 'w_eq8_type', 0.5)
    w_rms = getattr(args, 'w_rms', 0.1)
    
    # Scheduler: Cosine Annealing to zero LR at the last epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize Gradient Scaler for Automatic Mixed Precision (AMP)
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_loss = float('inf')
    
    global_step = start_step # Override global_step with the loaded step
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss, total_audio_loss, total_wave_loss, total_order_loss = 0, 0, 0, 0
        total_sat_type_loss, total_eq8_type_loss, total_rms_loss = 0, 0, 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            true_params = batch[1].to(device)
            true_order_idx = batch[3].to(device) if len(batch) > 3 else None
            
            # V8 Autoregressive Target Audio Generation
            # Convert true_order_idx (Batch, 8) to a string sequence for the ProxyChainer
            target_seq = None
            if true_order_idx is not None:
                token_to_name = {0: 'saturator', 1: 'eq8', 2: 'ott', 3: 'reverb'}
                first_seq = true_order_idx[0].tolist()
                target_seq = ['operator']
                for t in first_seq:
                    if t == 4: break # EOS
                    if t in token_to_name: target_seq.append(token_to_name[t])
            
            # Generate or load target audio
            if args.use_proxy_data:
                with torch.no_grad():
                    # Generate identical batch audio using the target_seq
                    target_audio_clean = chainer.forward_flat(true_params, sequence=target_seq)
                    # Apply Domain Randomization to what the Inverter SEES
                    target_audio = augmentor(target_audio_clean)
            else:
                target_audio = batch[2].to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # V8 Teacher Forcing Forward Pass
                outputs = model(target_audio, target_sequence=true_order_idx)
                
                # --- Categorical Losses ---
                true_wave_idx = true_params[:, 1].long()
                loss_wave = loss_wave_fn(outputs['wave_logits'], true_wave_idx)
                
                # Saturator Type
                true_sat_type_idx = true_params[:, 17].long()
                loss_sat_type = loss_wave_fn(outputs['sat_type_logits'], true_sat_type_idx)
                
                # EQ8 Filter Types (8 bands)
                loss_eq8_type = 0
                for i in range(8):
                    true_type_idx = true_params[:, 21 + i*4].long()
                    loss_eq8_type += loss_wave_fn(outputs['eq8_type_logits'][i], true_type_idx)
                loss_eq8_type = loss_eq8_type / 8.0 # Average across bands
                
                # V8 Sequence-to-Sequence Loss (Flattened from BxLxC to (B*L)xC)
                if true_order_idx is not None:
                    loss_order = loss_order_fn(
                        outputs['order_logits'].view(-1, 5), 
                        true_order_idx.view(-1)
                    )
                else:
                    loss_order = torch.tensor(0.0, device=device)

                # --- Audio Reconstruction Loss ---
                pi, mu, sigma = outputs['pi'], outputs['mu'], outputs['sigma']
                # Use MDN sample: pick the mean of the most probable Gaussian
                # mu: (batch, out_features, num_gaussians) -> mu_best: (batch, out_features)
                _, max_indices = torch.max(pi, dim=2)
                max_indices = max_indices.unsqueeze(-1)
                mu_best = torch.gather(mu, 2, max_indices).squeeze(-1)

                # --- Categorical Curriculum Learning (Teacher Forcing) ---
                if epoch < args.teacher_force_epochs:
                    # PHASE 1: Use true labels to bootstrap continuous parameters safely
                    true_sat_type_idx = true_params[:, 17].long()
                    true_eq8_type_idxs = [true_params[:, 21 + i*4].long() for i in range(8)]
                    
                    full_params_norm = assemble_full_params(mu_best, true_wave_idx, true_sat_type_idx, true_eq8_type_idxs, target_audio.shape[0], device)
                    pred_params_raw = denormalize_params(full_params_norm)
                    
                    # Convert true routing order into synthetic 1-hot logits for the Multiplexer
                    true_order_logits = None
                    if true_order_idx is not None:
                        # [B, 4] -> [B, 4, 5]
                        true_order_logits = F.one_hot(true_order_idx, num_classes=5).float() * 100.0
                    
                    # Bypassing Gumbel and Routing, forcing true discrete paths
                    reconstructed_audio = chainer.forward_flat(
                        pred_params_raw, 
                        order_idx=None, 
                        wave_logits=None, 
                        order_logits=true_order_logits
                    )
                else:
                    # PHASE 2: Fully Unlocked Audio-Guided Routing
                    pred_wave_idx = torch.argmax(outputs['wave_logits'], dim=1)
                    pred_sat_type = torch.argmax(outputs['sat_type_logits'], dim=1)
                    pred_eq8_types = [torch.argmax(logits, dim=1) for logits in outputs['eq8_type_logits']]

                    full_params_norm = assemble_full_params(mu_best, pred_wave_idx, pred_sat_type, pred_eq8_types, target_audio.shape[0], device)
                    pred_params_raw = denormalize_params(full_params_norm)
                    
                    reconstructed_audio = chainer.forward_flat(
                        pred_params_raw, 
                        wave_logits=outputs['wave_logits'],
                        order_logits=outputs['order_logits']
                    )
                
                loss_audio = loss_audio_fn(reconstructed_audio, target_audio)
                
                # --- Loudness Balance Loss ---
                # Ensures predicted parameters lead to target volume
                rms_pred = torch.sqrt(torch.mean(reconstructed_audio**2, dim=-1) + 1e-8)
                rms_target = torch.sqrt(torch.mean(target_audio**2, dim=-1) + 1e-8)
                loss_rms = F.l1_loss(rms_pred, rms_target)
                
                # --- Combined Loss ---
                loss = (w_audio * loss_audio) + (w_wave * loss_wave) + \
                       (w_order * loss_order) + (w_sat_type * loss_sat_type) + \
                       (w_eq8_type * loss_eq8_type) + (w_rms * loss_rms)
                
            if torch.isnan(loss):
                continue
                
            # AMP Backward Pass
            scaler.scale(loss).backward()
            
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            
            # Gradient Clipping: Prevent shattering weights during complex DSP spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step and Scaler update
            scaler.step(optimizer)
            scaler.update()
            
            # Logging
            total_loss += loss.item()
            total_audio_loss += loss_audio.item()
            total_wave_loss += loss_wave.item()
            total_order_loss += loss_order.item()
            total_sat_type_loss += loss_sat_type.item()
            total_eq8_type_loss += loss_eq8_type.item()
            total_rms_loss += loss_rms.item()
            
            if global_step % 10 == 0:
                writer.add_scalar("Train/Total_Loss", loss.item(), global_step)
                writer.add_scalar("Train/Audio_Loss", loss_audio.item(), global_step)
                writer.add_scalar("Train/Wave_Loss", loss_wave.item(), global_step)
                writer.add_scalar("Train/Order_Loss", loss_order.item(), global_step)
                writer.add_scalar("Train/Wave_Acc", get_accuracy(outputs['wave_logits'], true_wave_idx), global_step)
                # Force GCS upload and stdout heartbeat
                writer.flush()
                print(f"--- Step {global_step} | Loss: {loss.item():.4f} ---", flush=True)
                
                # Periodically sync to cloud
                if is_gcs and global_step % 50 == 0:
                    sync_to_gcs(local_log_dir, args.log_dir)
            
            # --- Periodic Checkpoint Save (every 1000 steps) ---
            if global_step > 0 and global_step % 1000 == 0:
                ckpt_path = f"checkpoints/inverter_{args.effect}_step{global_step}.pt"
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, ckpt_path)
                print(f"💾 Checkpoint saved: {ckpt_path}", flush=True)
                
                # Upload to GCS if running in cloud
                if is_gcs and args.log_dir.startswith("gs://"):
                    try:
                        parts = args.log_dir[5:].split('/', 1)
                        bucket_name = parts[0]
                        client = storage.Client()
                        bucket = client.bucket(bucket_name)
                        blob = bucket.blob(f"checkpoints/inverter_{args.effect}_step{global_step}.pt")
                        blob.upload_from_filename(ckpt_path)
                        # Also save as "latest" for easy resume
                        blob_latest = bucket.blob(f"checkpoints/inverter_{args.effect}_latest.pt")
                        blob_latest.upload_from_filename(ckpt_path)
                        print(f"☁️  Checkpoint uploaded to gs://{bucket_name}/checkpoints/", flush=True)
                    except Exception as e:
                        print(f"⚠️ GCS checkpoint upload failed: {e}", flush=True)

            if global_step % 500 == 0:
                # --- Real-time Visual/Audio Dashboard ---
                with torch.no_grad():
                    # Downmix to mono for TensorBoard audio preview
                    writer.add_audio("Train/Target_Audio", target_audio[0].mean(0), global_step, sample_rate=44100)
                    writer.add_audio("Train/Reconstructed_Audio", reconstructed_audio[0].mean(0), global_step, sample_rate=44100)
                    writer.add_image("Train/Target_Spec", plot_spectrogram(target_audio[0], "Target"), global_step)
                    writer.add_image("Train/Recon_Spec", plot_spectrogram(reconstructed_audio[0], "Reconstructed"), global_step)
                    writer.flush()
            
            global_step += 1
            progress_bar.set_postfix({
                "L": f"{loss.item():.3f}", 
                "L_aud": f"{loss_audio.item():.3f}",
                "L_wav": f"{loss_wave.item():.3f}"
            })
            
        # 5. Validation
        model.eval()
        val_loss, val_audio_loss, val_wave_loss, val_order_loss = 0, 0, 0, 0
        val_sat_type_loss, val_eq8_type_loss, val_rms_loss = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                true_params = batch[1].to(device)
                true_order_idx = batch[3].to(device) if len(batch) > 3 else None

                if args.use_proxy_data:
                    # Convert true routing order into synthetic 1-hot logits for the Multiplexer
                    true_order_logits = None
                    if true_order_idx is not None:
                        true_order_logits = F.one_hot(true_order_idx, num_classes=5).float() * 100.0
                        
                    target_audio_clean = chainer.forward_flat(
                        true_params, 
                        order_idx=None, 
                        wave_logits=None, 
                        order_logits=true_order_logits
                    )
                    # No augmentation on validation — measure true proxy performance
                    target_audio = target_audio_clean
                else:
                    target_audio = batch[2].to(device)

                outputs = model(target_audio)
                
                true_wave_idx = true_params[:, 1].long()
                loss_wave = loss_wave_fn(outputs['wave_logits'], true_wave_idx)
                
                # Saturator Type
                true_sat_type_idx = true_params[:, 17].long()
                loss_sat_type = loss_wave_fn(outputs['sat_type_logits'], true_sat_type_idx)
                
                # EQ8 Filter Types (8 bands)
                loss_eq8_type = 0
                for i in range(8):
                    true_type_idx = true_params[:, 21 + i*4].long()
                    loss_eq8_type += loss_wave_fn(outputs['eq8_type_logits'][i], true_type_idx)
                loss_eq8_type = loss_eq8_type / 8.0
                
                if true_order_idx is not None:
                    loss_order = loss_order_fn(outputs['order_logits'].view(-1, 5), true_order_idx.view(-1))
                else:
                    loss_order = torch.tensor(0.0, device=device)
                
                pi, mu, sigma = outputs['pi'], outputs['mu'], outputs['sigma']
                _, max_indices = torch.max(pi, dim=2)
                max_indices = max_indices.unsqueeze(-1)
                mu_best = torch.gather(mu, 2, max_indices).squeeze(-1)
                
                # Validation is ALWAYS unlocked to measure true model performance
                pred_wave_idx = torch.argmax(outputs['wave_logits'], dim=1)
                pred_sat_type = torch.argmax(outputs['sat_type_logits'], dim=1)
                pred_eq8_types = [torch.argmax(logits, dim=1) for logits in outputs['eq8_type_logits']]
                
                full_params_norm = assemble_full_params(mu_best, pred_wave_idx, pred_sat_type, pred_eq8_types, target_audio.shape[0], device)
                pred_params_raw = denormalize_params(full_params_norm)
                
                reconstructed_audio = chainer.forward_flat(
                    pred_params_raw, 
                    wave_logits=outputs['wave_logits'],
                    order_logits=outputs['order_logits']
                )
                loss_audio = loss_audio_fn(reconstructed_audio, target_audio)
                
                # RMS Loss - Tungsten Hardened
                rms_pred = torch.sqrt(torch.mean(reconstructed_audio**2, dim=-1) + 1e-4)
                rms_target = torch.sqrt(torch.mean(target_audio**2, dim=-1) + 1e-4)
                loss_rms = F.l1_loss(rms_pred, rms_target)
                
                loss = (w_audio * loss_audio) + (w_wave * loss_wave) + \
                       (w_order * loss_order) + (w_sat_type * loss_sat_type) + \
                       (w_eq8_type * loss_eq8_type) + (w_rms * loss_rms)

                val_loss += loss.item()
                val_audio_loss += loss_audio.item()
                val_wave_loss += loss_wave.item()
                val_order_loss += loss_order.item()
                val_sat_type_loss += loss_sat_type.item()
                val_eq8_type_loss += loss_eq8_type.item()
                val_rms_loss += loss_rms.item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        writer.add_scalar("Val/Total_Loss", avg_val, epoch)
        writer.add_scalar("Val/Audio_Loss", val_audio_loss / len(val_loader), epoch)
        
        # Validation (Visuals now handled by training heartbeat)

        print(f"\nEpoch {epoch+1}: Train Loss={avg_train:.4f} | Val Loss={avg_val:.4f}")
        print(f"  Components (Val): Audio={val_audio_loss/len(val_loader):.4f}, Wave={val_wave_loss/len(val_loader):.4f}, Order={val_order_loss/len(val_loader):.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"checkpoints/inverter_{args.effect}_audio_best.pt")
            print("🌟 New best audio inverter saved!")
            
        # Update Scheduler
        scheduler.step()
        print(f"  Current LR: {scheduler.get_last_lr()[0]:.6f}")
            
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the full audio inverter model.")
    parser.add_argument("--effect", type=str, default="full_chain", help="Must be 'full_chain' for this script.")
    parser.add_argument("--dataset_dir", type=str, default="dataset/full_chain", help="Path to the dataset.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--resume_encoder", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to full training checkpoint to resume from (.pt)")
    parser.add_argument("--w_audio", type=float, default=1.0, help="Weight for audio reconstruction loss.")
    
    # V8.9 Audio-Guided Architecture: Categorical losses are tracked for metrics but do NOT punish the model 
    # during backprop if the audio sounds identical. (Weights set to 0.0)
    parser.add_argument("--w_wave", type=float, default=0.0, help="Weight for categorical wave prediction loss (0.0 for pure audio routing).")
    parser.add_argument("--w_order", type=float, default=0.0, help="Weight for categorical order prediction loss (0.0 for pure audio routing).")
    parser.add_argument("--w_sat_type", type=float, default=0.0, help="Weight for saturator type loss (0.0 for pure audio routing).")
    parser.add_argument("--w_eq8_type", type=float, default=0.0, help="Weight for EQ8 type loss (0.0 for pure audio routing).")
    
    parser.add_argument("--w_rms", type=float, default=0.1, help="Weight for RMS loudness balance loss.")
    parser.add_argument("--teacher_force_epochs", type=int, default=50, help="Number of epochs to use true categorical labels.")
    parser.add_argument("--use_proxy_data", action="store_true", help="Use infinite proxy-generated data on-the-fly")
    parser.add_argument("--proxy_virtual_size", type=int, default=100000, help="Virtual epoch size for infinite dataset")
    parser.add_argument("--log_dir", type=str, default="runs/inverter_audio")
    
    args = parser.parse_args()
    if args.effect != "full_chain":
        print("Warning: This script is intended for the 'full_chain' effect. Results may be unexpected.")
        args.dataset_dir = f"dataset/{args.effect}"

    os.makedirs("checkpoints", exist_ok=True)
    train_inverter_audio(args)
