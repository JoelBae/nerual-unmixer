import torch
import numpy as np
import pytest
from src.models.proxies.ott_ddsp import OTTProxy

def test_ott_proxy_differentiability():
    batch_size = 2
    samples = 1000
    proxy = OTTProxy()
    
    # Input audio
    x = torch.randn(batch_size, samples, requires_grad=True)
    
    # Parameters [0, 1] as leaf tensors
    depth = torch.tensor([[0.5], [0.5]], requires_grad=True)
    thresh_l = torch.tensor([[0.5], [0.5]], requires_grad=True)
    thresh_m = torch.tensor([[0.5], [0.5]], requires_grad=True)
    thresh_h = torch.tensor([[0.5], [0.5]], requires_grad=True)
    gain_l = torch.tensor([[0.5], [0.5]], requires_grad=True)
    gain_m = torch.tensor([[0.5], [0.5]], requires_grad=True)
    gain_h = torch.tensor([[0.5], [0.5]], requires_grad=True)
    
    audio = proxy(x, depth, thresh_l, thresh_m, thresh_h, gain_l, gain_m, gain_h)
    
    loss = torch.mean(audio**2)
    loss.backward()
    
    # Check if gradients exist
    assert x.grad is not None
    assert depth.grad is not None
    assert thresh_l.grad is not None
    assert thresh_m.grad is not None
    assert thresh_h.grad is not None
    assert gain_l.grad is not None
    assert gain_m.grad is not None
    assert gain_h.grad is not None
    
    print("OTT Differentiability test passed!")

def test_ott_proxy_shape():
    batch_size = 1
    samples = 1000
    proxy = OTTProxy()
    
    x = torch.randn(batch_size, samples)
    
    depth = torch.tensor([[0.5]])
    thresh_l = torch.tensor([[0.5]])
    thresh_m = torch.tensor([[0.5]])
    thresh_h = torch.tensor([[0.5]])
    gain_l = torch.tensor([[0.5]])
    gain_m = torch.tensor([[0.5]])
    gain_h = torch.tensor([[0.5]])
    
    audio = proxy(x, depth, thresh_l, thresh_m, thresh_h, gain_l, gain_m, gain_h)
    
    assert audio.shape == (batch_size, samples)
    print("OTT Output shape test passed!")

if __name__ == "__main__":
    test_ott_proxy_differentiability()
    test_ott_proxy_shape()
