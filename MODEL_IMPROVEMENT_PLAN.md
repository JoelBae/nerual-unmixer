# OTT Proxy Model — Architecture History & Status

## ✅ Current Active Version: `ott.py` + `ott_proxy.pt`

The active OTT proxy uses a **gray-box architecture**:
1. **Analytical DSP** (frozen): LR4 crossover filters, RMS envelope followers, soft-knee compander logic — all implemented as differentiable PyTorch operations.
2. **Learned Residual Net** (trainable): A 128-channel 1D CNN that operates on raw audio to correct spectral differences between the analytical model and real Ableton OTT.

### Training
- **Phase 1**: Train the full model end-to-end with `DynamicsLoss` to learn envelope/gain behavior.
- **Phase 2**: Freeze the analytical DSP, train only the `residual_net` with `SpectralLoss` to refine frequency response.

```bash
# Phase 2 training (Colab)
python src/training/train_proxies.py --effect ott --phase2 --device cuda --dataset_dir dataset/ott_retrain/ott
```

### Evaluation Results
| Metric               | Score | Target | Status |
| -------------------- | ----- | ------ | ------ |
| Envelope Loss        | 0.040 | < 0.15 | ✅      |
| Spectral Convergence | 0.46  | < 0.20 | ❌      |
| Log-Magnitude Loss   | 1.91  | < 0.60 | ❌      |

> [!NOTE]
> The spectral metrics remain high despite good subjective audio quality.
> This is likely caused by phase misalignment from Ableton OTT's crossover filters
> introducing a small latency not present in our proxy.

---

## 🧪 Alternative: `ott_stft_conditioned.py` + `ott_proxy_cond.pt`

An experimental architecture that processes audio in the **frequency domain**:
1. Same gray-box DSP as above.
2. Converts output to STFT spectrogram → runs through a **2D Conv U-Net with FiLM conditioning** (the 7 OTT parameters are injected into the bottleneck) → converts back via ISTFT.
3. Trained with `VectorizedMultiScaleSpectralLoss` for balanced frequency coverage.

### Evaluation Results
| Metric               | Score | Target | Status |
| -------------------- | ----- | ------ | ------ |
| Envelope Loss        | 0.041 | < 0.15 | ✅      |
| Spectral Convergence | 0.47  | < 0.20 | ❌      |
| Log-Magnitude Loss   | 1.54  | < 0.60 | ❌      |

### Tradeoffs
- **Pro**: Better Log-Magnitude Loss (1.54 vs 1.91).
- **Con**: Significantly slower due to STFT/ISTFT and 2D convolutions. Not recommended for Encoder training where the proxy runs thousands of times per epoch.

```bash
# Training (Colab)
python src/training/train_proxies.py --effect ott --phase2 --stft_cond --device cuda --dataset_dir dataset/ott_retrain/ott

# Evaluation
python src/utils/evaluate_proxy_scientific.py checkpoints/ott_proxy_cond.pt --stft_cond
python src/utils/generate_listen_test.py --effect ott --checkpoint checkpoints/ott_proxy_cond.pt --stft_cond
```

---

## Decision
We are using `ott.py` + `ott_proxy.pt` for Encoder training (Strategy B) due to its speed advantage. The conditioned STFT variant is preserved as `ott_proxy_cond.pt` for potential future use.
