# Neural Un-Mixer Agent Guide

This repository contains the source code for **Neural Un-Mixer V3**, an AI system designed to deconstruct Ableton Live audio back into its original synthesizer and effect parameters using Differentiable DSP (DDSP).

## 📂 Core Architecture

### 1. Differentiable Proxies (`src/models/proxy/`)
These are digital twins of Ableton devices. They allow gradients to flow from audio loss back to the neural network.
- `ddsp_modules.py`: Contains `OperatorProxy` (Additive Synth) and helper modules (ADSR, Filters).
- `saturator.py`: Analytical implementation of the Saturator device.
- `eq8.py`: Analytical biquad-based implementation of EQ Eight.
- `ott.py`: **Active OTT Proxy**. Gray-box multiband compressor with 128-channel 1D Conv `residual_net` for spectral correction. Checkpoint: `ott_proxy.pt`.
- `ott_stft.py`: Experimental STFT-based variant (processes audio in frequency domain via 2D Conv). Superseded by `ott_stft_conditioned.py`.
- `ott_stft_conditioned.py`: **Alternative OTT Proxy**. Adds FiLM parameter conditioning to the STFT U-Net, injecting OTT knob positions into the spectral correction network. Slower but slightly better Log-Magnitude Loss. Checkpoint: `ott_proxy_cond.pt`. Use `--stft_cond` flag in training/evaluation scripts.
- `reverb.py`: FDN-based proxy for the Reverb device.
- `chain.py`: **The Signal Chain**. Links all proxies together. Supports dynamic reordering of effects.

### 2. Neural Models (`src/models/`)
- `encoder.py`: CNN-based audio feature extractor.
- `inverter.py`: The main model. Combines the Encoder with MDN (Mixture Density Network) and Classification heads.
- `heads/`: Specialized output heads for continuous (MDN) and categorical (Classification) parameters.

### 3. Data & Training (`src/data/`, `src/training/`)
- `generator.py`: OSC-based script that controls Ableton Live to generate random dataset samples.
- `ableton_client.py`: Low-level OSC communication with AbletonOSC.
- `dataset.py`: PyTorch Dataset for loading static Ableton audio/parameter pairs.
- `proxy_dataset.py`: **Infinite Dataset Generator** that creates random Ableton parameters on-the-fly for GPU proxy rendering.
- `augment.py`: Audio Augmentor applying **Domain Randomization** (Gain, Phase, Noise, Random EQ) per-sample.
- `train_inverter.py`: Main Inverter training script. Use `--use_proxy_data` to bypass Ableton data and train directly on the infinite GPU proxy distribution.
- `train_inverter_audio.py`: Main end-to-end training script using Strategy B (Match the sound, not the numbers).

## 🛠 Key Workflows

### Generating New Data
1. Open the Ableton project with the expected track/device setup.
2. Run `python src/data/generator.py --num_samples 1000 --output_dir ./dataset/full_chain`.

### Updating the Operator Wave Table
If you change the waveform selection in Ableton:
1. Run `python src/data/generate_wave_sweep.py`.
2. Run `python src/training/analyze_wave_sweep.py`.
3. The resulting `wave_table.pt` will be used by all agents and proxies.

### Training the Un-Mixer (Inverter)
- **Sim-to-Real Methodology**: We train the Inverter on infinite, on-the-fly data generated entirely by the PyTorch proxies, using Domain Randomization to prevent overfitting to proxy flaws.
- Run `python src/training/train_inverter.py --effect full_chain --use_proxy_data` to begin a massive GPU-bound training pass.
- **Stage 2 (Finetuning)**: `python src/training/train_inverter_audio.py` (Match the audio output of the full proxy chain using realistic Ableton target tracks).

## 📊 Parameter Mapping (63 total)
The system currently targets a 63-parameter vector, plus a categorical prediction for effect order.

1. **Operator (0-15)**: Transpose, Wave, Filter, ADSR, Pitch Env. (16 params)
2. **Saturator (16-20)**: Drive, Type, WS Curve, WS Depth, **Dry/Wet**. (5 params)
3. **EQ Eight (21-52)**: 8 bands × [Type, Freq, Gain, Q]. (32 params)
4. **OTT (53-59)**: Amount, 6 Thresholds. (7 params)
5. **Reverb (60-62)**: Decay, Size, **Dry/Wet**. (3 params) 
   - *Note: The Reverb proxy's existing `Dry/Wet` parameter is now learnable, instead of being fixed at 100%.*

## 🤝 Tips for Other Agents
- **Stereo Consistency**: All audio inside the system is expected to be `(Batch, 2, Time)`.
- **Normalization**: Use `src/utils/normalization.py` to convert between raw Ableton values and 0.1 neural space.
- **Proxy Weights**: Always call `chainer.load_checkpoints()` before training to ensure the proxies are high-fidelity.
- **Order Prediction**: The `NeuralInverter` now includes a `permutation_head` to predict the effect processing order.
