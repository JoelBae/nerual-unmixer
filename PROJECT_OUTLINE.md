# Project Outline: Neural Unmixer

## 1. Core Objective: Differential Manifold Alignment (DMA)
The goal is to reverse-engineer complex audio effects chains (Vital, Kilohearts, OTT) by aligning differentiable software proxies with the real VST3 outputs. We don't just "guess" parameters; we build a system that can accurately simulate the effects and use gradients to converge on the perfect patch.

---

## 2. System Architecture

### A. The "Smart Proxy" Design (Decentralized)
Instead of a single "brain" predicting everything, the system is composed of autonomous, effect-specific modules.

*   **DSP Core**: A differentiable implementation of the effect (e.g., a Tanh waveshaper for distortion, a Biquad for filter).
*   **Classification Head**: Predicts discrete parameters/modes (e.g., Distortion Type, Filter Shape).
*   **Gated Multi-Head Regression**: 
    *   Contains multiple "expert" regression heads (one for each discrete mode).
    *   Prevents interference between different mathematical manifolds (e.g., Hard Clip logic vs. Sine logic).
    *   The final output is an audio-weighted sum based on the Classification Head's probabilities.

### B. The Neural Backbone (Inverter)
*   **Role**: A neural network (CNN/Transformer) that extracts latent trajectories $z(t)$ from the audio.
*   **Locality**: Relies on the **Local Jacobian Assumption**, where each proxy is responsible for identifying its own influence in the latent space.

### C. The Feature Extractor (Alignment Measurement)
*   **Role**: Converts raw audio into phase-insensitive representations like **Mel-spectrograms**.
*   **Verification Assumptions (@[neural_unmixer-4.pdf])**:
    *   **Assumption 2 ($\alpha_c$)**: Jacobian Directional Alignment. Proxies must move "spectral" in the same direction as the VST for each parameter knob.
    *   **Assumption 3 ($\epsilon$)**: Input-Output Stability. The proxy's response to audio transients and frequency shifts must match the VST's topological behavior.
*   **Purpose**: Used to calibrate the technical manifold without being misled by phase offsets.

### D. The Effect Sequencer
*   **Role**: Predicts the topological order of the effect chain (e.g., `Vital -> KH Distortion -> OTT`).
*   **Implementation**: A pointer-network or autoregressive head that outputs a permutation of the active proxies.
*   **Importance**: Critical for the Sequential Solver, as the "Peeling" order depends on the physical signal path.

### E. The Calibration Matrix $K(\theta)$
*   **Role**: A diagonal matrix $K(\theta) = \text{diag}(k_1, \dots, k_K)$ that acts as the "exchange rate" between the Proxy manifold and the VST manifold.
*   **Mathematical Basis**: Each scaling factor $k_c$ is calculated by projecting the proxy Jacobian onto the VST Jacobian:
    $$k_c(\theta) = \frac{\langle j_{vst}, j_{proxy} \rangle_2}{\| j_{vst} \|_2^2}$$
*   **Purpose**: Resolves non-linear sensitivity warping, ensuring that a "10% turn" of a knob in the proxy results in the perceptually correct "X% turn" in the VST.

### F. Differential Engine (`PedalboardEngine`)
*   Uses Spotify's `pedalboard` to host real VST3 plugins.
*   Generates **Ground Truth Audio** and **Parameter Jacobians** (via finite differences) used to compute $K(\theta)$.

---

## 3. Inversion Strategy: "The Peeling Method"

We do not perform **Signal Inversion** ($f^{-1}$), which is lossy and often impossible for nonlinear effects. Instead, we perform **Parameter Inversion** via **Optimization** aiming for **Functional Alignment** (re-creating the sound rather than the exact parameter values).

### Step 1: One-Shot Inference (Fast)
1.  **Encoder** extracts latent $z(t)$.
2.  **Smart Proxies** instantly predict modes $\hat{c}$ and parameters $\hat{\theta}$.
3.  **Result**: A high-speed "best guess" for the entire patch.

### Step 2: Manifold Mapping & $K(\theta)$ Calibration
1.  **LHS Sampling**: Generate optimal parameter clusters using Latin Hypercube Sampling.
2.  **Jacobian Calibration**: Compute the scaling factors $k_c(\theta)$ at each cluster to construct the $K(\theta)$ matrix map.

### Step 3: Path Integration (Calibration)
1.  **Continuous Path Integral**: Instead of a "one-jump" prediction, we integrate the scaling matrix $K(\tau)$ along the path from a reference state $\theta_{ref}$ to the predicted state $\hat{\theta}_P$.
    $$\hat{\theta}_A = \theta_{ref} + \int_{\theta_{ref}}^{\hat{\theta}_P} K(\tau) d\tau$$
2.  **Result**: Maps the proxy's latent prediction reliably into the final target command $\hat{\theta}_A$.

### Step 4: Iterative Refinement (Final Polish)
1.  Run the **Full Proxy Chain** forward and compare $\hat{y}$ with target $y$ in the Mel domain.
2.  **Backpropagate** gradients through the Smart Proxies via the calibrated $K(\theta)$ map to perform final sub-epsilon tweaks.

---

## 4. Why This Works
*   **Structural Prior**: By using the **Vital Oscillator** as the source, we satisfy the fundamental condition of having "infinite resolution" generators to fill in the gaps created by destructive/nonlinear effects.
*   **Modular Scaling**: Adding a new VST only requires training a new "Smart Proxy" and fine-tuning the feature extractor, rather than rebuilding the entire unmixer.

---

## 5. Roadmap
*   **Phase 1 (Current)**: Proof of Concept. Calibration of Vital, KH Distortion, KH 3-Band EQ, and OTT proxies + unified unmixer. Modulation is restricted to Vital parameters for simplicity.
*   **Phase 2**: External Modulation. Handling LFOs and Envelopes for effect proxies (KH EQ, Distortion).
*   **Phase 3**: Context Awareness & Large-scale chains. Handling noise and complex topologies.
