import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.models.proxy.ddsp_modules import DifferentiableADSR

if __name__ == "__main__":
    sample_rate = 10  # Very low sample rate so we can clearly see the math step-by-step
    adsr = DifferentiableADSR(sample_rate=sample_rate)
    
    # 2-second ADSR config
    attack = torch.tensor([[0.2]], dtype=torch.float32)  # 200ms
    decay = torch.tensor([[0.3]], dtype=torch.float32)   # 300ms
    sustain = torch.tensor([[0.5]], dtype=torch.float32) # 50%
    release = torch.tensor([[0.4]], dtype=torch.float32) # 400ms
    
    note_off_time = 1.5
    num_samples = int(2.0 * sample_rate)  # 20 frames total
    
    # Grab the internal logic variables directly to show the math
    t = torch.arange(num_samples, dtype=torch.float32) / sample_rate
    
    # Emulate the clamping logic
    a = attack.item()
    d = decay.item()
    s = sustain.item()
    r = release.item()
    
    attack_env = torch.clamp(t / (a + 1e-8), 0.0, 1.0)
    decay_env = 1.0 - (1.0 - s) * torch.clamp((t - a) / (d + 1e-8), 0.0, 1.0)
    release_env = 1.0 - torch.clamp((t - note_off_time) / (r + 1e-8), 0.0, 1.0)
    
    envelope = attack_env * decay_env * release_env
    
    print("Time(s) |  Attack  *   Decay  *  Release =  Final Env | Visualization")
    print("-" * 75)
    for i in range(num_samples):
        time_sec = t[i].item()
        att = attack_env[i].item()
        dec = decay_env[i].item()
        rel = release_env[i].item()
        env = envelope[i].item()
        
        # ASCII bar chart (Scale 0.0 to 1.0 -> 0 to 20 blocks)
        bar = "#" * int(env * 20)
        
        print(f" {time_sec:4.1f}   |   {att:4.2f}   *   {dec:4.2f}   *   {rel:4.2f}   =   {env:4.2f}   | {bar}")
