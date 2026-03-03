# OTT Proxy Model Improvement Plan

This document outlines the planned next steps for improving the spectral performance of the OTT proxy model, based on recent analysis.

## Current Status:
The OTT proxy model exhibits good envelope matching but struggles with spectral accuracy. The `DynamicsLoss` function has been adjusted to give equal weighting to spectral and envelope components (by changing the `loss_env` multiplier from `5.0` to `1.0`). The model is currently undergoing "Phase 2" training, where only the `residual_net` is being optimized to learn spectral corrections.

## Next Steps:

1.  **Complete Phase 2 Training:**
    *   Allow the current Phase 2 training (with the adjusted `DynamicsLoss` weights) to run to completion or until early stopping is triggered.
    *   The command to run this training is:
        ```bash
        python src/training/train_proxies.py --effect ott --phase2 --resume --device mps --dataset_dir dataset/ott_retrain/ott
        ```
        *(Note: Ensure `checkpoints/ott_proxy.pt` exists by renaming `ott_proxy_best_so_far.pt` to `ott_proxy.pt` before starting the training, if not already done.)*

2.  **Evaluate the Improved Model:**
    *   Once Phase 2 training is complete, re-evaluate the model's performance using both the scientific benchmarks and the listen test.
    *   **Scientific Evaluation:**
        ```bash
        python src/utils/evaluate_proxy_scientific.py checkpoints/ott_proxy.pt
        ```
    *   **Listen Test:**
        ```bash
        python src/utils/generate_listen_test.py --effect ott --checkpoint checkpoints/ott_proxy.pt
        ```
    *   Compare the new Spectral Convergence and Log-Magnitude Loss scores against the targets, and perform subjective listening tests on the generated audio samples.

3.  **If Spectral Performance Remains Insufficient (Conditional Step):**
    *   **Increase `residual_net` Capacity:** If the evaluation in Step 2 shows that spectral performance is still not meeting targets, the `residual_net` might need more capacity.
        *   Modify `src/models/proxy/ott.py` to add more convolutional layers to the `residual_net`, increase the number of channels (e.g., from 64 to 128), or experiment with different kernel sizes/dilations.
        *   After modifying the architecture, restart Phase 2 training.

4.  **Adjust Learning Rate (Conditional Step):**
    *   If the model struggles to converge or improves very slowly even after increasing capacity, experiment with different learning rates (e.g., `1e-3`, `5e-5`) in the `train_proxies.py` script.
        *   Modify the `--lr` argument in the training command.

This iterative approach should help systematically improve the OTT proxy's spectral accuracy.
