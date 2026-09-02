# Remaining Work

This document consolidates the remaining tasks and experiments from the v2 dataset and model upgrade plans that have not yet been implemented. The core pipeline (group-aware splitting, Kabsch Cα distance filtering, row-level provenance, deterministic pair caps, sequence-level evaluation, and CLI entry points) is complete.

## Genuine Remaining Improvements

1. **Symmetric x↔y training**:
   - Compute the training objective in both directions (`loss_xy` and `loss_yx`) to reflect the symmetric nature of substitution matrices.
   - Enforces the intended invariance at the objective level and reduces dependence on data construction details. Lowest risk, best aligned with substitution-matrix logic.
2. **Raw and Balanced Validation Views**:
   - Create two validation views: `val_raw` (natural held-out distribution) and `val_balanced` (diagnostic group/contact-balanced subset).
   - Use `val_raw` for final model selection and `val_balanced` to detect failure modes hidden by dominant groups.
3. **State-Margin Metric**:
   - For VQ, compute `margin = d_sorted[:, 1] - d_sorted[:, 0]` (distance to second closest centroid minus distance to closest).
   - For FSQ, report distance to nearest quantization boundary.
4. **Validation-Time State-Transition Metrics**:
   - Accumulate counts `[state_x, state_y]`.
   - Report state entropy, state perplexity, minimum state frequency, dead-state fraction, joint entropy, aligned-state mutual information, and transition-adjusted mutual information.
   - Use these for early stopping or model selection instead of just reconstruction loss.

## Genuinely Interesting Experiments (In Recommended Order)

1. **State-Transition Auxiliary Head**:
   - Predict the aligned partner's discrete state from the source quantized state to directly train the discrete states to be predictive of aligned partner states. Most directly tests whether the model can learn states that improve substitution statistics. (Priority: High)
2. **K-Means Centroid Initialization**:
   - Experiment with initializing the VQ codebook using k-means from encoder latents rather than random sampling. Test only if VQ shows unstable code usage across seeds or has dead/rare states. (Priority: Medium)
3. **Context-Window Encoder**:
   - Allow the encoder to see neighboring residue features. (Caution: may increase local sequential dependency between letters). Do this only after you have stable sequence-level evaluation, to see whether it improves `MI_tot` or just reconstruction loss. (Priority: Medium-Low)
4. **Rotation-Trick VQ**:
   - Experiment with rotation-based VQ. Test only if standard EMA VQ underperforms and you want to keep improving VQ rather than switching to FSQ. (Priority: Low-Medium)

## Useful Engineering and Scaling Work

1. **Resolved Run Configs**:
   - Save configuration artifacts: `run_config.resolved.json`, `preprocessing_config.json`, `model_config.json`, and `train_config.json`.
2. **Pair-Table Schema Validation**:
   - Validate pair tables (`pairs.parquet`) explicitly before training.
3. **Lazy Large-Corpus Storage**:
   - Support lazy datasets with Zarr/Arrow for when the dataset no longer fits in RAM.
4. **Mixed Precision**:
   - Enable mixed precision for faster training, after numerical stability is fully confirmed.
5. **torch.compile**:
   - Enable for faster inference speeds.
6. **Product Quantization**:
   - Only relevant if exploring much larger compositional code spaces (e.g., 64, 128, 256+ states). Unnecessary for 20-32 states.
