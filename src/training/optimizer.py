import torch
import torch.optim as optim

class GCWDOptimizer:
    """
    Wrapper for Optimizer with Gradient Clipping and Weight Decay (GCWD).
    Stabilizes training on Spectral Loss landscapes by preventing floating-point overflow.
    """
    def __init__(self, params, lr=1e-4, weight_decay=1e-2, clip_value=1.0):
        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.clip_value = clip_value

    def step(self, closure=None):
        # 1. Clip Gradients
        # We clip by value or norm. The spec mentions "Gradient Clipping".
        # Clipping by norm is generally more stable for deep networks.
        # But if "gradients ranging from 10^40" is the issue, value clipping might be safer 
        # to catch Infs/NaNs before norm calculation if they exist, 
        # though torch.nn.utils.clip_grad_norm_ handles Inf logic too.
        
        # Let's use clip_grad_norm_ for the params in the optimizer
        for group in self.optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], self.clip_value)
            
        # 2. Optimizer Step
        return self.optimizer.step(closure)

    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
