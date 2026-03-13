import torch
import numpy as np
import pytest
from src.models.proxies.vital_ddsp import VitalProxy

def test_vital_proxy_differentiability():
    batch_size = 2
    samples = 4410
    proxy = VitalProxy(n_harmonics=16)
    
    # Parameters [0, 1] as leaf tensors
    f0_leaf = torch.tensor([[440.0], [880.0]], requires_grad=True)
    f0 = f0_leaf.repeat(1, samples).unsqueeze(-1)
    
    wavetable_pos = torch.tensor([[0.5], [0.5]], requires_grad=True)
    cutoff = torch.tensor([[0.8], [0.8]], requires_grad=True)
    res = torch.tensor([[0.5], [0.5]], requires_grad=True)
    attack = torch.tensor([[0.1], [0.1]], requires_grad=True)
    decay = torch.tensor([[0.2], [0.2]], requires_grad=True)
    sustain = torch.tensor([[0.5], [0.5]], requires_grad=True)
    release = torch.tensor([[0.3], [0.3]], requires_grad=True)
    gate = torch.ones((batch_size, samples))
    
    audio = proxy(f0, wavetable_pos, cutoff, res, attack, decay, sustain, release, gate)
    
    loss = torch.mean(audio**2)
    loss.backward()
    
    # Check if gradients exist
    assert f0_leaf.grad is not None
    assert cutoff.grad is not None
    assert res.grad is not None
    assert attack.grad is not None
    assert decay.grad is not None
    assert sustain.grad is not None
    assert release.grad is not None
    
    print("Differentiability test passed!")

def test_vital_proxy_output_shape():
    batch_size = 1
    samples = 1000
    proxy = VitalProxy()
    
    f0 = torch.ones((batch_size, samples, 1)) * 440.0
    wavetable_pos = torch.tensor([[0.5]])
    cutoff = torch.tensor([[0.5]])
    res = torch.tensor([[0.5]])
    attack = torch.tensor([[0.01]])
    decay = torch.tensor([[0.01]])
    sustain = torch.tensor([[1.0]])
    release = torch.tensor([[0.01]])
    gate = torch.ones((batch_size, samples))
    
    audio = proxy(f0, wavetable_pos, cutoff, res, attack, decay, sustain, release, gate)
    
    assert audio.shape == (batch_size, samples)
    print("Output shape test passed!")

if __name__ == "__main__":
    test_vital_proxy_differentiability()
    test_vital_proxy_output_shape()
