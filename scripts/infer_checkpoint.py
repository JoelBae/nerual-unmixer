import torch
import torchaudio
import argparse
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.inverter import NeuralInverter
from src.models.proxy.chain import ProxyChainer
from src.utils.normalization import denormalize_params

def assemble_full_params(mdn_out, wave_idx, sat_type_idx, eq8_type_idxs, batch_size, device):
    full_params = torch.zeros(batch_size, 63, device=device)
    idx = 0
    full_params[:, 0] = mdn_out[:, idx]; idx += 1
    full_params[:, 1] = wave_idx.float()
    full_params[:, 2:16] = mdn_out[:, idx:idx+14]; idx += 14
    full_params[:, 16] = mdn_out[:, idx]; idx += 1
    full_params[:, 17] = sat_type_idx.float()
    full_params[:, 18:21] = mdn_out[:, idx:idx+3]; idx += 3
    for band in range(8):
        base = 21 + (band * 4)
        full_params[:, base] = eq8_type_idxs[band].float()
        full_params[:, base+1:base+4] = mdn_out[:, idx:idx+3]; idx += 3
    full_params[:, 53:60] = mdn_out[:, idx:idx+7]; idx += 7
    full_params[:, 60:63] = mdn_out[:, idx:idx+3]; idx += 3
    return full_params

def generate_samples(checkpoint_path, num_samples=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading checkpoint {checkpoint_path} on {device}...")
    
    # 1. Load Model
    model = NeuralInverter(latent_dim=512).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded step {ckpt.get('step', 'unknown')}, epoch {ckpt.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(ckpt)
    model.eval()
    
    # 2. Load Chainer
    chainer = ProxyChainer().to(device)
    chainer.load_checkpoints()
    chainer.eval()

    # 3. Generate Random Ground Truths
    print(f"Generating {num_samples} random ground truth samples...")
    torch.manual_seed(42) # Fixed seed for reproducibility
    norm_truth = torch.rand(num_samples, 63, device=device)
    raw_truth = denormalize_params(norm_truth)
    
    with torch.no_grad():
        target_audio = chainer.forward_flat(raw_truth)
        
        # 4. Invert
        print("Running Inverter...")
        outputs = model(target_audio)
        
        # Predict params
        pi, mu, sigma = outputs['pi'], outputs['mu'], outputs['sigma']
        _, max_indices = torch.max(pi, dim=2)
        max_indices = max_indices.unsqueeze(-1)
        mu_best = torch.gather(mu, 2, max_indices).squeeze(-1)
        
        pred_wave_idx = torch.argmax(outputs['wave_logits'], dim=1)
        pred_sat_type = torch.argmax(outputs['sat_type_logits'], dim=1)
        pred_eq8_types = [torch.argmax(outputs['eq8_type_logits'][i], dim=1) for i in range(8)]
        
        full_params_norm = assemble_full_params(mu_best, pred_wave_idx, pred_sat_type, pred_eq8_types, num_samples, device)
        pred_params_raw = denormalize_params(full_params_norm)
        
        # 5. Reconstruct
        print("Reconstructing Audio...")
        # Note: At step 1000, Teacher Forcing is active in training, but in inference we just use the predicted routing
        reconstructed_audio = chainer.forward_flat(
            pred_params_raw,
            wave_logits=outputs['wave_logits'],
            order_logits=outputs['order_logits']
        )
        
    # Save outputs
    os.makedirs("results", exist_ok=True)
    for i in range(num_samples):
        # target_audio is (num_samples, L)
        ta = target_audio[i].unsqueeze(0) if target_audio.dim() == 2 else target_audio[i]
        ra = reconstructed_audio[i].unsqueeze(0) if reconstructed_audio.dim() == 2 else reconstructed_audio[i]
        
        torchaudio.save(f"results/sample_{i}_target.wav", ta.cpu(), 44100)
        torchaudio.save(f"results/sample_{i}_recon.wav", ra.cpu(), 44100)
    
    print("✅ Done! Saved to results/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/checkpoints_inverter_full_chain_latest.pt")
    args = parser.parse_args()
    generate_samples(args.ckpt)
