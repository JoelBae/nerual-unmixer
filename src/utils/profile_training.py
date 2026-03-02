import torch
import time
import os
import sys
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.proxy.ott import OTTProxy
from src.models.losses import DynamicsLoss
from src.data.dataset import NeuralProxyDataset

def profile_training_loop(device="mps", num_batches=20):
    print(f"\n--- Profiling OTT Training on {device} ---")
    
    # 1. Setup
    dataset = NeuralProxyDataset(effect_name="ott", dataset_dir="dataset/ott", split="train", preload=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    
    model = OTTProxy().to(device)
    if device == "mps":
        model = model.half()
    criterion = DynamicsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Timers
    t_data = []
    t_forward = []
    t_loss = []
    t_backward = []
    t_step = []
    
    print("   > Warming up...")
    it = iter(loader)
    for _ in range(2):
        input_audio, params, target_audio = next(it)
        input_audio, params, target_audio = input_audio.to(device), params.to(device), target_audio.to(device)
        
        if device == "mps":
            input_audio = input_audio.half()
            with torch.autocast(device_type="mps", enabled=True, dtype=torch.float16):
                pred = model(input_audio, params)
                loss = criterion(pred, target_audio)
        else:
            pred = model(input_audio, params)
            loss = criterion(pred, target_audio)
            
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"   > Profiling {num_batches} batches...")
    start_profiling = time.time()
    
    for i in range(num_batches):
        # 1. Data Loading
        t0 = time.time()
        input_audio, params, target_audio = next(it)
        input_audio, params, target_audio = input_audio.to(device), params.to(device), target_audio.to(device)
        t_data.append(time.time() - t0)
        
        # 2. Forward
        t1 = time.time()
        if device == "mps":
            input_audio = input_audio.half()
            with torch.autocast(device_type="mps", enabled=True, dtype=torch.float16):
                pred = model(input_audio, params)
        else:
            pred = model(input_audio, params)
            
        if device == "mps": torch.mps.synchronize()
        t_forward.append(time.time() - t1)
        
        # 3. Loss
        t2 = time.time()
        if device == "mps":
            with torch.autocast(device_type="mps", enabled=True, dtype=torch.float16):
                loss = criterion(pred, target_audio)
        else:
            loss = criterion(pred, target_audio)
            
        if device == "mps": torch.mps.synchronize()
        t_loss.append(time.time() - t2)
        
        # 4. Backward
        t3 = time.time()
        loss.backward()
        if device == "mps": torch.mps.synchronize()
        t_backward.append(time.time() - t3)
        
        # 5. Optimizer
        t4 = time.time()
        optimizer.step()
        if device == "mps": torch.mps.synchronize()
        t_step.append(time.time() - t4)
        optimizer.zero_grad()

    # Results
    print(f"\nResults per batch (avg of {num_batches}):")
    print(f"   - Data Loading: {sum(t_data)/num_batches*1000:7.2f} ms")
    print(f"   - Forward Pass: {sum(t_forward)/num_batches*1000:7.2f} ms")
    print(f"   - Loss Calc:    {sum(t_loss)/num_batches*1000:7.2f} ms")
    print(f"   - Backward Pass:{sum(t_backward)/num_batches*1000:7.2f} ms")
    print(f"   - Optimizer Step:{sum(t_step)/num_batches*1000:7.2f} ms")
    print(f"   -------------------------------")
    print(f"   - Total Batch:  {(time.time()-start_profiling)/num_batches*1000:7.2f} ms")

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    profile_training_loop(device)
