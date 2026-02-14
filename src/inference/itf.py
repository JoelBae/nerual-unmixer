import torch
import torch.nn as nn
import torch.optim as optim

class InferenceTimeFinetuner:
    """
    Implements Inference-Time Finetuning (ITF).
    Refines the predicted parameters theta to minimize Spectral Loss 
    between Proxy(theta) and the specific test sample audio.
    """
    def __init__(self, proxy_model, spectral_loss_fn, steps=50, lr=1e-2):
        self.proxy = proxy_model
        self.loss_fn = spectral_loss_fn
        self.steps = steps
        self.lr = lr
        
        # Freeze proxy weights
        for p in self.proxy.parameters():
            p.requires_grad = False
            
    def finetune(self, initial_params, target_audio):
        """
        initial_params: tensor (batch, param_dim)
        target_audio: tensor (batch, samples)
        """
        # Clone parameters and enable gradients
        theta = initial_params.clone().detach().requires_grad_(True)
        
        optimizer = optim.Adam([theta], lr=self.lr)
        
        for i in range(self.steps):
            optimizer.zero_grad()
            
            # 1. Pass through Proxy
            pred_audio = self.proxy(theta)
            
            # 2. Compute Loss
            # Note: We only use the Spectral component here, not the MDN NLL
            loss = self.loss_fn.spectral_loss(pred_audio, target_audio)
            
            # Add Regularization if needed (L_reg in spec)
            
            loss.backward()
            optimizer.step()
            
            # Optional: Clip parameters to valid range [0, 1]
            theta.data.clamp_(0.0, 1.0)
            
        return theta.detach()
