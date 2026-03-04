# Audio-Chain Estimation Network (ACEN)

**Inverse Signal Chain Estimation for Analysis-by-Synthesis**

The Audio-Chain Estimation Network (ACEN) is a deep learning system designed to reverse-engineer synthesizer parameters from raw audio. It listens to a sound and predicts the exact configuration (oscillator, filters, effects) used to create it.

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

### 3. The Modular Proxy System
Instead of a monolithic "Black Box" synthesizer, V3 uses a library of **Independent Differentiable Neural Proxies**:
*   **Training Strategy**: Each proxy is trained **separately** on its own dataset. This ensures high-fidelity simulation of each specific effect without interference from others.
*   **Concept**: We use specialized Differentiable Digital Signal Processing (DDSP) and Neural Network architectures tailored to each specific device:
    *   **Operator (Instrument Proxy)**: A fully mathematical, differentiable Additive Synthesizer. 
    *   **Audio Effects Proxies**: We use **Gray-Box DDSP** techniques wherever possible:
        *   **EQ Eight (Analytical DDSP)**: A pure-math biquad filter engine. Identical to Ableton's internal DSP.
        *   **Reverb (LTI)**: An MLP predicts the Impulse Response decay curve, applied via FFT Convolution.
        *   **Saturator (Non-Linear)**: A differentiable waveshaper or lightweight TCN.
        *   **OTT (Dynamics)**: A differentiable multiband compressor.
*   **Benefits**: This guarantees math-perfect audio processing with zero neural-hallucination artifacts, while still allowing gradients to flow end-to-end to train the Encoder.


## 🚀 Optimization Strategy

### Hybrid Loss Function
$$ \mathcal{L}_{total} = \lambda_{MDN} \cdot \mathcal{L}_{NLL} + \lambda_{Spectral} \cdot \mathcal{L}_{Spectral} $$
*   **NLL (Negative Log-Likelihood)**: Ensures the parameter distribution matches the ground truth.
*   **Multi-Scale STFT Loss**: Self-supervised loss ensuring audio fidelity.

### Sim-to-Real Domain Randomization (New in V3)
To bridge the gap between our PyTorch Proxies (Simulator) and Ableton Live (Reality), the Inverter is trained on an **Infinite On-The-Fly Proxy Dataset**.
*   **Infinite Data**: Random Ableton parameters are generated on the CPU, and the audio is rendered dynamically on the GPU by the `ProxyChainer`.
*   **Domain Randomization**: Every single generated audio sample is uniquely augmented (Random EQ, Random Gain, Phase Inversion, White Noise) before hitting the Inverter. This forces the Inverter to learn robust, structural DSP behaviors rather than overfitting to specific proxy frequency coloration.

### Training Pipeline
```bash
python -m src.training.train_proxies --effect <effect_name>
```

### Inference-Time Finetuning (ITF)
At inference time, the model actively refines its prediction by running Gradient Descent on the predicted parameters $\theta$ to minimize the Spectral Error for the specific input sample.

