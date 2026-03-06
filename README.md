# Audio-Chain Estimation Network (ACEN) 🎛️

**Inverse Signal Chain Estimation for Analysis-by-Synthesis**

The Audio-Chain Estimation Network (ACEN) is a deep learning system designed to reverse-engineer synthesizer parameters from raw audio. It listens to a sound and predicts the exact configuration (oscillator waveform, synth ADSR, filters, and audio effect routing) used to create it.

---

## 🏗 Architecture Overview (V9)

The system employs a **Hybrid Analysis-by-Synthesis** approach. It combines a direct prediction network with a differentiable DSP proxy graph to enable end-to-end self-supervised learning without rigid parameter datasets.

### 1. The Encoder & Heads
*   **Input**: Raw Stereo Audio Waveform.
*   **Model**: 1D Convolutional Neural Network operating purely in the time-domain, allowing the model to capture geometric saturation clipping and microscopic phase-shifts that are typically erased by standard 2D Spectrogram representations. This feeds into a Transformer Decoder.
*   **Outputs**: ACEN predicts **63 exact parameters** separated into:
    *   **Continuous Knobs**: Mixture Density Networks (MDN) map uncertainty for parameters like Reverb Decay, OTT Thresholds, and EQ Frequencies.
    *   **Categorical Switches**: Classifiers predict Oscillator Waveform types, Saturator routing shapes, and EQ filter modes.
    *   **Autoregressive Routing**: A sequence-to-sequence decoder paired with Differentiable Gumbel-Softmax multiplexing predicts the exact **order of the effects chain**. 

### 2. The Differentiable DSP Proxies
Instead of treating Ableton Live as a "Black Box", ACEN utilizes a library of **Independent Differentiable Neural Proxies**. The predicted parameters are routed through these proxies to generate audio dynamically on the GPU:
*   **Operator**: A mathematical, differentiable Additive Synthesizer. 
*   **EQ Eight**: A pure-math biquad filter engine, identical to Ableton's DSP.
*   **Reverb (LTI)**: An MLP predicting Impulse Response decay curves applied via FFT Convolution.
*   **Saturator**: A differentiable waveshaper.
*   **OTT (Dynamics)**: A differentiable multiband downward/upward compressor.

All components are perfectly wrapped in PyTorch graphs, allowing loss gradients to flow from the final audio output backwards through the DSP chain to correct the CNN's parameter choices.

---

## 🚀 Optimization & Training Strategy

### Sim-to-Real Domain Randomization
ACEN utilizes a pure **Sim-to-Real** pipeline.
1.   An infinite number of random, valid parameter configurations are generated on the CPU.
2.   `OnTheFlyProxyDataset` passes these configurations through the DSP Proxies to render the target audio directly in GPU memory.
3.   An `AudioAugmentor` applies intense Domain Randomization (up to +3dB Gain, Phase-inversion, White Noise injection) to the target audio.
This forces the model to learn structural acoustic reasoning instead of simply memorizing the exact math of the proxies.

### Curriculum Learning (Teacher Forcing)
To stabilize the massive 63-parameter graph, the training loop utilizes **Teacher Forcing**. For the first several epochs, the true categorical labels (like the correct signal chain order and waveform shape) are injected directly into the Proxy Chainer. This anchors the continuous parameters (like EQ frequencies) to reality before the routing training wheels are removed.

### Multi-Resolution STFT Audio Loss
Because categorical metrics (like guessing the wrong waveform) are highly discontinuous, ACEN's loss curve relies primarily on a Multi-Scale Spectrogram Distance function. Categorical accuracy metrics are tracked but mathematically silenced from the backpropagation, allowing the model to freely search for the best *sonic* matching parameter set regardless of the numerical path.

---

## ☁️ Google Cloud & MLOps

ACEN is designed for scale. Given the massive memory footprint of calculating gradients for 64 simultaneous dynamically routed DSP chains, training runs are packaged via Docker and deployed natively to **NVIDIA L4 GPUs on Google Cloud Vertex AI**.

*   **Mixed Precision (AMP)**: Full-scope `torch.cuda.amp.autocast()` wraps both the neural model inference and the massive DSP track generations, utilizing Nvidia TensorCores to run the chain in memory-efficient `FP16` while preserving mastering weights in `FP32`.
*   **Experiment Tracking**: TensorBoard is deeply integrated to stream synchronous validation losses, accuracy curves, interactive Reconstructed Audio players, and live Spectrograms directly from the cloud to your local machine via `gcsfs` and Google Cloud Storage.

---

### Command Execution
```bash
# 1. Build and push the latest architecture to GCR
gcloud builds submit --tag gcr.io/[PROJECT_ID]/inverter-audio:latest .

# 2. Launch the Vertex AI Custom Batch Job
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=inverter-v9-master \
  ...
```
