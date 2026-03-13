# Neural Unmixer: DDSP-Driven Manifold Alignment

This plan implements the "Differential Manifold Alignment Strategy" from @[neural_unmixer-4.pdf]. We aim to create differentiable proxies for Vital, Kilohearts, and OTT that satisfy the structural conditions required for sound-matching inversion.

> [!NOTE]
> **Functional Inversion Goal**: We are not seeking exact parameter identity (recovering the absolute original values). Instead, we seek **functional alignment**—finding *any* parameter set that produces a perceptually identical sound.

## User Review Required

> [!IMPORTANT]
> **Phase 1 Effect Bundle**: Based on the 85% sound-recreation goal, we will prioritize:
> 1. **Vital (Generator)**: Oscillator + Primary Filter.
> 2. **Kilohearts Distortion**: For harmonic saturation.
> 3. **Kilohearts 3-Band EQ**: For spectral shaping and balancing.
> 4. **Xfer OTT**: Multi-band compression (essential for modern "bright" sounds).
> 5. **Kilohearts Reverb**: For spatial tails (handled as a late-stage 'peeling' step).
> *Note: We are prioritizing the KH 3-Band EQ over the Filter module for initial spectral matching.*
> [!IMPORTANT]
> **Decentralized Gated Multi-Head Architecture**: Each Proxy encapsulates its own **Classification** and **Regression** heads. To handle non-smooth parameters (Distortion Type):
> 1. **Classification Head**: Picks the mode/type.
> 2. **Gated Multi-Head Regression**: Individual heads predict parameters for *each* mode independently, which are then weighted by the classification probabilities. This prevents interference between different mathematical manifolds.
> 3. **Jacobian Locality**: We assume parameter sensitivity is localized to the effect (diagonal or block-diagonal global Jacobian). The encoder extracts features, and the "Smart Proxies" handle the inversion of their specific local manifolds.


> [!NOTE]
> **DAW Integration via Presets**: To ensure DAW-independence, the system will export predictions as standard `.vstpreset` files. Users can simply drag and drop these onto their DAW tracks (Ableton, Logic, FL Studio, etc.) to recreate the sound using the real VSTs.

> [!TIP]
> **Phased Approach**: We will first implement a **Context-Unaware Unmixer** (training on clean synthesized audio). Context-aware unmixing (using Multi-Stem Augmentation to handle background noise) will be considered in Phase 2 once the core alignment is verified.

## Proposed Changes

### [Core Infrastructure]

#### [NEW] `src/data/pedalboard_engine.py`
A high-performance rendering engine using Spotify's `pedalboard` to generate ground-truth audio and parameter Jacobians (via finite differences) for Vital, Kilohearts, and OTT.

#### [NEW] `src/utils/alignment_validator.py`
A validation suite to compute $\alpha_c$ (cosine similarity) and $\epsilon$ (divergence) between the DDSP proxies and the target VSTs.

### [DDSP Proxies]

#### [MODIFY] `src/models/proxies/vital_ddsp.py`
A differentiable oscillator and filter bank matching Vital's "Basic Shapes" wavetable.
- **Categorical 7-Shape Oscillator**: Treats `oscillator_1_wave_frame` as a categorical selector for 7 unique waveforms.
- **Discovered Shapes (from .vitaltable)**:
    - **Shape 0 (Sine)**: Pos 0.0 (Threshold < 0.063)
    - **Shape 1 (Flatter Sine)**: Pos 0.125 (Threshold 0.063-0.188)
    - **Shape 2 (Triangle)**: Pos 0.251 (Threshold 0.188-0.314)
    - **Shape 3 (Saw)**: Pos 0.376 (Threshold 0.314-0.439)
    - **Shape 4 (Square)**: Pos 0.502 (Threshold 0.439-0.627)
    - **Shape 5 (Pulse 1)**: Pos 0.753 (Threshold 0.627-0.816)
    - **Shape 6 (Pulse 2)**: Pos 0.878 (Threshold 0.816-1.0)
- **Gated Multi-Head Architecture**:
    - **Classification Head**: Maps estimated frame to soft-probabilities over the 7 kernels.
    - **Kernel Summation**: Weighted sum of hardcoded bins (FFT'd from .vitaltable) for gradient flow.

#### [NEW] `src/models/proxies/kilohearts_ddsp.py`
Collection of differentiable modules for Focus effects:
- **KH 3-Band EQ**: Differentiable shelving/bell filters for Low, Mid, and High bands.
- **KH Distortion**: Differentiable waveshaper (Drive/Bias).
- **KH Reverb**: Differentiable delay-line/all-pass network (simplified tail proxy).

#### [NEW] `src/models/proxies/ott_ddsp.py`
A multi-band differentiable compressor proxy matching Xfer OTT's 3-band topology.

### [Inversion]

#### [NEW] `src/models/unmixer.py`
The core inverter model featuring a **Neural Backbone**:
- **Inverter Backbone**: A CNN/Transformer that extracts latent trajectories $z(t)$ from the audio.
- **Proxy Selector**: Predicts which effects are active.
- **Effect Sequencer**: Predicts the topological order/sequence of the active effects.
- **Parameter Delegation**: Passes $z(t)$ to the active Proxy's internal heads to predict $\hat{\theta}(t)$.

> [!NOTE]
> **Alignment via Phase-Insensitive Features**: We use a **Feature Extractor** (e.g., Mel-spectrogram) to measure the difference between VST renders and Proxy renders. This ensures we align the "sound" on a technical manifold without being misled by phase offsets.

### [Verification Infrastructure]

#### [NEW] `src/utils/feature_extractor.py`
A differentiable module that converts raw audio into Mel-spectrograms.
- Uses `torch.stft` with appropriate windowing.
- Implements a logarithmic Mel filterbank.
- Provides a consistent representation for functional alignment tests.

#### [MODIFY] `src/utils/alignment_validator.py`
Refactor to support representation-domain Jacobians.
- Add `representation_fn` argument to `compute_alignment_metrics`.
- Compute $\nabla_\theta h(\theta)$ where $h$ is the transformation to Mel domain.
- Support both Time-domain (legacy) and Mel-domain verification.

## Manifold Mapping (Step 2: LHS)

To calibrate the proxy across the entire parameter space, we must map the manifold's curvature and cross-parameter interactions.

### Latin Hypercube Sampling
We will use **Latin Hypercube Sampling (LHS)** to generate optimal parameter clusters. LHS ensures that each parameter is sampled uniformly across its range while testing highly interactive combinations.

1.  **Cluster Generation**: Sample $N$ clusters in the proxy parameter space $\Theta$.
2.  **Jacobian Empirical Measurement**: For each cluster, render audio through the VST and compute the empirical Jacobian $\nabla_\theta h(\theta)$ in the Mel domain.
3.  **$K(\theta)$ Calibration**: Compute the scalar scaling factors $k_c(\theta)$ for each parameter column by minimizing the $L_2$ norm of the residual projection (Equation 7 in @[neural_unmixer-4.pdf]).
4.  **Exchange Rate Map**: Store these $k_c$ values to construct the $K(\theta)$ "exchange rate" map across the parameter space.

### Path Integration
To resolve non-linear sensitivity warping, the calibrated VST command $\hat{\theta}_A$ is derived by integrating the scaling matrix $K(\tau)$ along the prediction path:
$$\hat{\theta}_A = \theta_{ref} + \int_{\theta_{ref}}^{\hat{\theta}_P} K(\tau) d\tau$$

### Implementation Details
- **`src/data/sampler.py` [DONE]**: A utility to handle LHS generation using `scipy.stats.qmc.LatinHypercube`.
- **`scripts/generate_jacobian_clusters.py` [DONE]**: A script to iterate through LHS samples, render VST audio, and store the resulting Jacobian columns.
- **`src/data/manifold_mapper.py` [NEXT]**: A module to compute and store the $K(\theta)$ scaling factors for each cluster.
- **`src/utils/path_integrator.py` [NEW]**: A numerical solver to perform the path integration from $\theta_{ref}$ to $\hat{\theta}_P$.

## Verification Plan

### Step 1: Functional Alignment (Verified)
- [x] **Parameter Alignment ($\alpha_c$)**: Verify individual parameter Jacobians in the Mel domain.
- [x] **Input Stability ($\epsilon$)**: Verify proxy response to audio perturbations matches VST.

### Step 2: Manifold Mapping (In Progress)
- [x] **LHS Coverage**: Verify cluster uniformity across the parameter range.
- [ ] **Empirical Jacobian Accuracy**: Verify measured partials $\nabla_\theta h$ against finite-difference VST ground truth at cluster centers.

### Manual Verification
1. **Representational Sweep**: Plot Mel-spectrogram Jacobians to visually confirm functional alignment.
2.  **Cluster Calibration Plots**: Visualize how the Jacobian direction $(\alpha_c)$ changes across different LHS clusters.
