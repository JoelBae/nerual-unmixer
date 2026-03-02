# Neural Un-Mixer Agent Guide

This repository contains the source code for **Neural Un-Mixer V3**, an AI system designed to deconstruct Ableton Live audio back into its original synthesizer and effect parameters using Differentiable DSP (DDSP).

## 📂 Core Architecture

### 1. Differentiable Proxies (`src/models/proxy/`)
These are digital twins of Ableton devices. They allow gradients to flow from audio loss back to the neural network.
- `ddsp_modules.py`: Contains `OperatorProxy` (Additive Synth) and helper modules (ADSR, Filters).
- `saturator.py`: Analytical implementation of the Saturator device.
- `eq8.py`: Analytical biquad-based implementation of EQ Eight.
- `ott.py`: TCN-based proxy for the Multiband Dynamics (OTT) device.
- `reverb.py`: FDN-based proxy for the Reverb device.
- `chain.py`: **The Signal Chain**. Links all proxies together. Supports dynamic reordering of effects.

### 2. Neural Models (`src/models/`)
- `encoder.py`: CNN-based audio feature extractor.
- `inverter.py`: The main model. Combines the Encoder with MDN (Mixture Density Network) and Classification heads.
- `heads/`: Specialized output heads for continuous (MDN) and categorical (Classification) parameters.

### 3. Data & Training (`src/data/`, `src/training/`)
- `generator.py`: OSC-based script that controls Ableton Live to generate random dataset samples.
- `ableton_client.py`: Low-level OSC communication with AbletonOSC.
- `dataset.py`: PyTorch Dataset for loading audio/parameter pairs.
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

### Training the Un-Mixer
- **Stage 1 (Optional)**: `python src/training/train_inverter.py` (Predict parameters directly).
- **Stage 2 (Primary)**: `python src/training/train_inverter_audio.py` (Match the audio output of the full proxy chain).

## 📊 Parameter Mapping (62 total)
The system currently targets a 62-parameter vector:
1. **Operator (0-15)**: Transpose, Wave, Filter, ADSR, Pitch Env.
2. **Saturator (16-19)**: Drive, Type, WS Curve, WS Depth.
3. **EQ Eight (20-51)**: 8 bands × [Type, Freq, Gain, Q].
4. **OTT (52-58)**: Amount, 6 Thresholds.
5. **Reverb (59-61)**: Decay, Size, Dry/Wet.

## 🤝 Tips for Other Agents
- **Stereo Consistency**: All audio inside the system is expected to be `(Batch, 2, Time)`.
- **Normalization**: Use `src/utils/normalization.py` to convert between raw Ableton values and 0.1 neural space.
- **Proxy Weights**: Always call `chainer.load_checkpoints()` before training to ensure the proxies are high-fidelity.
- **Order Prediction**: The `NeuralInverter` now includes a `permutation_head` to predict the effect processing order.
