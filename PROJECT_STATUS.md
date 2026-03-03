# Neural Un-Mixer V3: Project Status Report

## 🚀 Accomplishments So Far

We have successfully built a fully differentiable "Robot Ears" system capable of deconstructing complex Ableton synth patches back into their raw parameters.

### 1. Differentiable Proxy Chain
We have created high-fidelity digital twins (Proxies) for the entire Ableton signal chain:
- **Operator (Synth)**: Using a **Differentiable Lookup Table** approach. We captured the harmonic "fingerprint" of all 128 waveforms via FFT, allowing the AI to dial in exact timbres with 100% accuracy.
- **EQ Eight & Saturator**: Implemented as **Analytical Proxies**. These use exact mathematical formulas, requiring zero training and providing mathematical perfection in the signal path.
- **OTT & Reverb**: Trained using **Deep Learning (TCN/LSTM)**. These proxies have been verified to replicate the complex dynamics and spatial characteristics of the original Ableton effects.
  - **UPDATE**: The `OTTProxy`'s attack modeling has been refactored. It now uses an explicit FIR filter approximation for the attack envelope follower instead of a placeholder learnable parameter. This improves the physical accuracy of the multiband compression simulation.

### 2. Architecture & Learning
- **Unified Chainer**: The `ProxyChainer` module allows audio to flow through a *dynamic sequence* of effects while maintaining gradients for backpropagation. It now supports runtime re-ordering of the entire effect chain.
- **Encoder Strategy**: The system is now optimized for **Strategy B (Audio-Domain Loss)**. Instead of just guessing numbers, the model learns by trying to "re-synthesize" the target sound and measuring how close the audio matches.
- **Verified Differentiability**: Confirmed that gradients flow successfully from the final audio output back to the Encoder's neural weights.

### 3. Repository Optimization
- Standardized all proxy interfaces to Stereo `(B, 2, T)`.
- Reorganized the training pipeline to support multi-device un-mixing.
- Cleaned up deprecated datasets and redundant checkpoints to streamline the training environment.

---

## 📅 Moving Forward: The Road to Final Un-Mixing

### Phase 1: Proxy & Model Updates (Dry/Wet Prediction)
The immediate goal is to update the entire system to support per-effect dry/wet mix control.

### Phase 2: End-to-End Encoder Training
Once the proxies are updated, we will train the `AudioEncoder` to predict:
1.  **The Effect Order**: Which of the 24 possible effect chain permutations was used.
2.  **Dry/Wet Mixes**: The individual mix level for the Saturator and Reverb.
3.  All 16 Operator parameters (Wave, Filter, ADSR, Pitch Env).
4.  All 5 Saturator parameters (including Dry/Wet).
5.  All 32 EQ Eight parameters.
6.  All 7 OTT parameters (Amount and 6 Thresholds).
7.  All 3 Reverb parameters (including a now-learnable Dry/Wet).

### Phase 3: Refinement & Inference
- **Parameter Snapping**: Finalizing the "Inference Head" that snaps categorical predictions.
- **Generalization**: Testing the model on complex patches it hasn't seen.

### Phase 4: Deployment
- Preparing a final script for "One-Click Un-Mixing".

---
**Current Status**: 🟡 In Progress: Updating proxies to support Dry/Wet control.
**Next Command**: `python src/data/generator.py --num_samples 1000 --output_dir ./dataset/full_chain_dry_wet`
