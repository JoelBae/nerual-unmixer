import torch
from src.models.proxy.ott_stft import OTTSTFTProxy

if __name__ == "__main__":
    print("Testing OTTSTFTProxy...")
    model = OTTSTFTProxy(duration=2.0)
    
    # Fake batch of 2 stereo audio files (2 seconds at 44.1kHz)
    dummy_audio = torch.randn(2, 2, 88200)
    
    # Fake params (Amt, Time, In, Out, H, M, L) -> shape (B, 7)
    dummy_params = torch.rand(2, 7) 
    
    out = model(dummy_audio, dummy_params)
    print("Output shape:", out.shape)
    assert out.shape == dummy_audio.shape, "Shape mismatch in IO!"
    print("✅ Forward pass successful!")
