# Pedalboard Engine Documentation

The `PedalboardEngine` is a core utility designed to bridge Python-based DDSP modeling with industry-standard VST3 plugins using Spotify's `pedalboard` library.

## File Overview: `src/data/pedalboard_engine.py`

This file contains the `PedalboardEngine` class, which handles:
1. **Plugin Management**: Loading and caching VST3 instances (Vital, OTT, and Kilohearts suite).
2. **Deterministic Rendering**: Generating audio given a set of normalized parameters (0 to 1).
3. **Jacobian Estimation**: Computing the sensitivity of audio output relative to parameter changes using finite differences. This is critical for the "Differential Manifold Alignment" strategy.

## Key Methods

### `render(plugin_id, parameters, duration_sec, ...)`
- **`plugin_id`**: String identifier (e.g., `"vital"`, `"ott"`, `"kh_filter"`).
- **`parameters`**: A dictionary where keys are VST parameter names and values are floats in the range `[0, 1]`.
- **Returns**: A NumPy array containing the rendered audio.

### `compute_jacobian(plugin_id, base_params, target_param_names, ...)`
- **`delta`**: The step size for finite differentiation (default: `1e-3`).
- **Returns**: A dictionary mapping each parameter name to its gradient (the change in audio per unit change in parameter).

## VST Path Configuration

The engine assumes VST3 plugins are installed at standard macOS locations:
- Vital: `/Library/Audio/Plug-Ins/VST3/Vital.vst3`
- OTT: `/Library/Audio/Plug-Ins/VST3/OTT.vst3`
- Kilohearts: `/Library/Audio/Plug-Ins/VST3/Kilohearts/kHs Filter.vst3` (and others)

## Usage Example

```python
from src.data.pedalboard_engine import PedalboardEngine

engine = PedalboardEngine()
audio = engine.render("ott", {"depth": 0.8}, duration_sec=2.0)
```
