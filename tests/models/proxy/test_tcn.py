import os
import sys
import torch
import torchaudio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.models.proxy.tcn import ConditionalTCN

if __name__ == "__main__":
    print("Testing the Conditional TCN Base Class...")
    
    # 1. Create a dummy audio tensor (e.g. 1 second of stereo white noise)
    sample_rate = 44100
    batch_size = 1
    num_channels = 2
    num_samples = sample_rate
    
    dummy_audio = torch.randn(batch_size, num_channels, num_samples)
    
    # 2. Create dummy parameters (e.g. 4 dials for an effect)
    num_params = 4
    dummy_params = torch.rand(batch_size, num_params)
    
    # 3. Instantiate the TCN
    # We'll use 10 blocks (dilation up to 512). It should have a large receptive field.
    tcn = ConditionalTCN(num_params=num_params, num_channels=32, num_blocks=10, kernel_size=15)
    
    # 4. Pass audio and params through the network
    print("Passing audio through TCN...")
    output_audio = tcn(dummy_audio, dummy_params)
    
    print(f"Input Audio Shape:  {dummy_audio.shape}")
    print(f"Params Shape:       {dummy_params.shape}")
    print(f"Output Audio Shape: {output_audio.shape}")
    
    if output_audio.shape == dummy_audio.shape:
        print("\nSuccess! The TCN preserved the audio shape perfectly.")
    else:
        print("\nError: Output shape does not match input shape.")
