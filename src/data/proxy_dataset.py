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
        
        # Helper to set a parameter to random or neutral
        def set_param(idx, neutral_val, prob_random=0.5):
            p_min, p_max = PARAM_RANGES[idx]
            if random.random() < prob_random:
                if idx in CATEGORICAL_INDICES:
                    params[idx] = random.randint(int(p_min), int(p_max))
                else:
                    params[idx] = random.uniform(p_min, p_max)
            else:
                params[idx] = neutral_val

        # --- 0. Operator (16) ---
        set_param(0, 60.0, prob_random=0.5)  # Transpose (Neutral: 60 = Middle C)
        params[1] = random.randint(0, 127)   # Wave (Always Random)
        
        # Operator Filter (Indices 2:4)
        # Ensure filter is at least partially open (Freq >= 30)
        p_min, p_max = PARAM_RANGES[2]
        params[2] = random.uniform(max(p_min, 30.0), p_max)
        
        if random.random() < 0.6: # 60% Sparse Filter
            set_param(3, 0.0, prob_random=1.0)   # Res
            set_param(4, 0.0, prob_random=1.0)   # Amount
        else: # 40% Static Filter (Bypassed)
            params[2] = 127.0
            params[3] = 0.0
            params[4] = 0.0
            
        # Operator Envelopes (Indices 5:15)
        # 5:8 Filter Env, 9:11 Pitch Env, 12:15 Amp Env
        for i in range(5, 12):
            p_min, p_max = PARAM_RANGES[i]
            params[i] = random.uniform(p_min, p_max)

        # Anti-Silence: Amp Env (Indices 12:15)
        # Ensure either Sustain >= 25 (20%) OR Decay >= 40 (long enough to hear)
        params[12] = random.uniform(PARAM_RANGES[12][0], PARAM_RANGES[12][1]) # Attack
        params[13] = random.uniform(PARAM_RANGES[13][0], PARAM_RANGES[13][1]) # Decay
        params[14] = random.uniform(PARAM_RANGES[14][0], PARAM_RANGES[14][1]) # Sustain
        params[15] = random.uniform(PARAM_RANGES[15][0], PARAM_RANGES[15][1]) # Release
        
        if params[14] < 25.0 and params[13] < 40.0:
            if random.random() < 0.5:
                params[14] = random.uniform(25.0, 127.0) # Boost Sustain
            else:
                params[13] = random.uniform(40.0, 127.0) # Boost Decay

        # --- 1. Saturator (5) ---
        # No device-level bypass; routing handles presence.
        set_param(16, 0.0, prob_random=1.0) # Drive (Always random)
        params[17] = random.randint(0, 7)    # Type
        set_param(18, 0.5, prob_random=1.0)  # WS Curve
        set_param(19, 0.5, prob_random=1.0)  # WS Depth
        set_param(20, 0.5, prob_random=1.0)  # Wet

        # --- 2. EQ Eight (32: 8 bands x 4 params) ---
        for band in range(8):
            base = 21 + (band * 4)
            # Internal sparsity: 50% chance a band is at 0dB even if EQ8 is in chain
            is_band_active = random.random() < 0.5 
            params[base] = random.randint(0, 7)                      # Type
            set_param(base+1, 1000.0, prob_random=1.0)               # Freq
            set_param(base+2, 0.0, prob_random=1.0 if is_band_active else 0.0) # Gain
            set_param(base+3, 0.71, prob_random=1.0)                 # Q

        # --- 3. OTT (7) ---
        set_param(53, 0.5, prob_random=1.0) # Amount
        for i in range(54, 60): # Thresholds
            set_param(i, -20.0, prob_random=1.0)

        # --- 4. Reverb (3) ---
        set_param(60, 0.2, prob_random=1.0) # Decay
        set_param(61, 0.5, prob_random=1.0) # Size
        set_param(62, 0.3, prob_random=1.0) # Wet

        dummy_audio = torch.zeros(1, 88200)
        
        # V8 Autoregressive Routing
        seq_len = random.randint(1, 8)
        sequence = []
        for _ in range(seq_len):
            sequence.append(random.randint(0, 3))
        while len(sequence) < 8:
            sequence.append(4)
        order_idx = torch.tensor(sequence, dtype=torch.long)
        
        return dummy_audio, params, dummy_audio, order_idx
