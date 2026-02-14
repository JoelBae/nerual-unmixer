# Neural Un-Mixer V3

**Inverse Signal Chain Estimation for Analyis-by-Synthesis**

The Neural Un-Mixer V3 is a deep learning system designed to reverse-engineer synthesizer parameters from raw audio. It listens to a sound and predicts the exact configuration (oscillator, filters, effects) used to create it.

## 🏗 Architecture Overview

The system employs a **Hybrid Analysis-by-Synthesis** approach. It combines a direct prediction network with a differentiable DSP proxy to enable end-to-end self-supervised learning.

### 1. The Encoder
*   **Input**: Log-Mel Spectrogram of the effected audio.
*   **Source Signal**: **White Noise** (or Chirp/Impulse). The system now assumes the input to the chain is a known, broadband signal, making the spectral characteristics of the output purely a result of the effects.
*   **Model**: 1D CNN with Global Average Pooling.
*   **Output**: A compact latent embedding vector $z$ representing the timbre and texture of the sound.

### 2. The Heads
The embedding $z$ is split into three specialized prediction heads:
*   **MDN Head**: Mixture Density Network for continuous parameters (Knobs), handling the multimodal uncertainty where multiple settings (e.g., specific Filter Cutoff vs. Waveform shape) can sound similar.
*   **Classification Head**: Standard classifier for discrete choices (Switches/Modes).
*   **Hyperbolic Sequence Decoder**: A GRU operating in **Poincaré Ball** space. This is critical for predicting the *ordered* chain of effects (e.g., `Distortion -> Reverb` vs `Reverb -> Distortion`), as hyperbolic geometry efficiently embeds hierarchical structures.

### 3. The Modular Proxy System (New in V3)
Instead of a monolithic "Black Box" synthesizer, V3 uses a library of **Independent Differentiable Neural Proxies**:
*   **Training Strategy**: Each proxy is trained **separately** on its own dataset. This ensures high-fidelity simulation of each specific effect without interference from others.
*   **Concept**: We train a small, specialized TCN (Temporal Convolutional Network) for each individual audio effect. The currently supported effects are:
    *   **Saturator** (Non-linear distortion)
    *   **EQ Eight** (4-band parametric EQ)
    *   **OTT** (Multiband compression)
    *   **Phaser** (Modulation)
    *   **Reverb** (Spatial processing)
*   **Dynamic Chaining**: At runtime, these modules are dynamically assembled into a computation graph that matches the effect chain predicted by the Decoder.
    *   *Example*: If the decoder predicts `[Saturator, Reverb]`, the signal flows `Oscillator -> ProxySaturator -> ProxyReverb -> Output`.
*   **Benefits**: This allows gradients to flow from the output audio, through the specific effect chain, back to the encoder, enabling the model to learn the nuances of how effects interact.


## 🚀 Optimization Strategy

### Hybrid Loss Function
$$ \mathcal{L}_{total} = \lambda_{MDN} \cdot \mathcal{L}_{NLL} + \lambda_{Spectral} \cdot \mathcal{L}_{Spectral} $$
*   **NLL (Negative Log-Likelihood)**: Ensures the parameter distribution matches the ground truth.
*   **Multi-Scale STFT Loss**: Self-supervised loss ensuring audio fidelity. It computes the spectral distance between the predicted and target audio across multiple FFT resolutions (`[2048, 1024, 512, 256, 128, 64]`), combining:
    *   **Spectral Convergence Loss**: Magnitude distance.
    *   **Log Magnitude Loss**: L1 distance in log-domain.

### Training Pipeline
New training infrastructure has been added:
*   **W&B Integration**: Automatic experiment tracking (Loss curves, Audio samples).
*   **Dockerized**: Reproducible environment via `Dockerfile`.
*   **Usage**: `python -m src.training.train_proxies --effect <effect_name>`

### Inference-Time Finetuning (ITF)
At inference time, the model actively refines its prediction:
1.  **Predict**: Generating initial parameters $\theta_{init}$.
2.  **Freeze**: Freezing the Proxy weights.
3.  **Optimize**: Running Gradient Descent on $\theta$ to minimize the Spectral Error for the specific input sample.

