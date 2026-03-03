import torch
from torch.utils.data import Dataset
import random
from src.utils.normalization import PARAM_RANGES, CATEGORICAL_INDICES

class OnTheFlyProxyDataset(Dataset):
    """
    Infinite dataset that generates random parameters on the fly.
    Audio (target) will be generated dynamically on the GPU by the ProxyChainer in the training loop.
    """
    def __init__(self, virtual_length=100000, effect_name="full_chain"):
        self.virtual_length = virtual_length
        self.effect_name = effect_name.lower()
        self.num_params = max(PARAM_RANGES.keys()) + 1 # Should be 63

    def __len__(self):
        return self.virtual_length

    def __getitem__(self, idx):
        # We ignore idx to make it truly infinite
        params = torch.zeros(self.num_params, dtype=torch.float32)
        
        for i, (p_min, p_max) in PARAM_RANGES.items():
            if i in CATEGORICAL_INDICES:
                params[i] = random.randint(int(p_min), int(p_max))
            else:
                params[i] = random.uniform(p_min, p_max)
                
        # To maintain compatibility with NeuralProxyDataset return signature:
        # return input_audio, params_tensor, target_audio, order_idx
        # We'll use dummy zeros for audio because they will be generated on GPU
        # Shape: (1, 88200) for 2 seconds at 44100Hz
        dummy_audio = torch.zeros(1, 88200)
        
        # Order index (for the chainer)
        order_idx = torch.tensor(0, dtype=torch.long)
        
        return dummy_audio, params, dummy_audio, order_idx
