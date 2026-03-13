# OTT Proxy Documentation

The `OTTProxy` is a differentiable implementation of Xfer Records' OTT multi-band compressor. It enables parameter inversion for modern "bright" and "heavy" sounds within the Neural Unmixer framework.

## File Overview: `src/models/proxies/ott_ddsp.py`

This file contains the DDSP implementation of OTT, structured into three main components:

1.  **`DifferentiableCrossover3Band`**:
    -   Implements a 3-band divider using 4th-order Linkwitz-Riley filters.
    -   Crossover Frequencies: Fixed at **120 Hz** and **2.5 kHz**.
    -   Ensures perfect reconstruction (summing the bands equals the input signal).

2.  **`DifferentiableCompressor`**:
    -   A dual-action compressor (Upward and Downward).
    -   **Level Detection**: Uses a differentiable leaky integrator (envelope follower) with parameterizable attack and release times.
    -   **Gain Curve**: Implements a differentiable soft-knee transfer function.
    -   Calculates gain in the log (dB) domain for better precision.

3.  **`OTTProxy`**:
    -   The top-level module that coordinates the multi-band processing.
    -   Exposes parameters for:
        -   `depth`: Global dry/wet mix.
        -   `thresh_l/m/h`: Per-band thresholds.
        -   `gain_l/m/h`: Per-band makeup gain.

## Key Parameters

| Parameter | Range | Description |
| :--- | :--- | :--- |
| `depth` | `[0, 1]` | Global intensity of the effect (Dry -> Wet). |
| `thresh_l/m/h` | `[0, 1]` | Threshold for each band (Low, Mid, High). |
| `gain_l/m/h` | `[0, 1]` | Makeup gain for each band. |

## Usage Example

```python
import torch
from src.models.proxies.ott_ddsp import OTTProxy

# Initialize proxy
proxy = OTTProxy(sample_rate=44100)

# Input signal (batch, samples)
x = torch.randn(1, 44100)

# Parameters (batch, 1)
depth = torch.tensor([[1.0]])
params = {p: torch.tensor([[0.5]]) for p in ["thresh_l", "thresh_m", "thresh_h", "gain_l", "gain_m", "gain_h"]}

# Process
output = proxy(x, depth, **params)
```

## Differentiability
The entire signal path is differentiable. You can compute gradients from an audio loss directly back to any of the control parameters:

```python
output.mean().backward()
print(depth.grad) # Gradients for the depth parameter
```
