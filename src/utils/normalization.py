import torch

# Parameter Ranges defined in src/data/generator.py
# Format: {index: (min, max)}
PARAM_RANGES = {
    # --- 0. Operator (16) ---
    0: (58, 70),     # Transpose
    1: (0, 127),    # WAVE (Categorical - not normalized by MDN)
    2: (20, 127),    # Filter Freq
    3: (0, 90),      # Filter Res
    4: (32, 127),    # Fe Amount
    5: (0, 64),      # Fe Attack
    6: (10, 127),    # Fe Decay
    7: (0, 127),     # Fe Sustain
    8: (10, 127),    # Fe Release
    9: (0, 64),      # Pe Amount
    10: (10, 100),   # Pe Decay
    11: (0, 64),     # Pe Peak
    12: (0, 40),     # Ae Attack
    13: (30, 127),   # Ae Decay
    14: (0, 127),    # Ae Sustain
    15: (15, 127),   # Ae Release

    # --- 1. Saturator (5) ---
    16: (0.0, 1.0),  # Drive
    17: (0, 7),      # Type
    18: (0.0, 1.0),  # WS Curve
    19: (0.0, 1.0),  # WS Depth
    20: (0.0, 1.0),  # Dry/Wet

    # --- 2. EQ Eight (32: 8 bands x 4 params) ---
    # Band 1
    21: (0, 7),      # Type
    22: (20, 20000), # Freq
    23: (-15.0, 15.0), # Gain
    24: (0.1, 18.0), # Q
    # Repeating for 8 bands
    **{i: (0, 7) if (i-21) % 4 == 0 else (20, 20000) if (i-21) % 4 == 1 else (-15, 15) if (i-21) % 4 == 2 else (0.1, 18) 
       for i in range(25, 53)},

    # --- 3. OTT (7) ---
    53: (0.0, 1.0),  # Amount
    54: (-60.0, 0.0), # Abv Thresh L
    55: (-60.0, 0.0), # Abv Thresh M
    56: (-60.0, 0.0), # Abv Thresh H
    57: (-60.0, 0.0), # Blw Thresh L
    58: (-60.0, 0.0), # Blw Thresh M
    59: (-60.0, 0.0), # Blw Thresh H

    # --- 4. Reverb (3) ---
    60: (0.05, 0.4), # Decay
    61: (0.0, 1.0),  # Size
    62: (0.0, 1.0),  # Dry/Wet
}

# Categorical indices that should not be normalized/predicted by MDN
CATEGORICAL_INDICES = {1, 17, 21, 25, 29, 33, 37, 41, 45, 49}

def normalize_params(params):
    """Normalize a (batch, 63) param tensor to 0-1 for MDN (skipping categorical)."""
    normalized = params.clone()
    for i, (p_min, p_max) in PARAM_RANGES.items():
        if i in CATEGORICAL_INDICES: continue 
        normalized[:, i] = (params[:, i] - p_min) / (p_max - p_min + 1e-8)
    return normalized

def denormalize_params(norm_params):
    """Convert 0-1 MDN outputs back to raw Ableton values."""
    raw = norm_params.clone()
    for i, (p_min, p_max) in PARAM_RANGES.items():
        if i in CATEGORICAL_INDICES: continue 
        raw[:, i] = norm_params[:, i] * (p_max - p_min) + p_min
    return raw
