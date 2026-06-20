# Experiments quarantine

The v2 simplification removed several auxiliary objectives and training-framework features from the
core path. Rather than maintain dead `.py` files, this directory points at the pre-refactor snapshot
in git history, where the removed code still runs.

## Pre-refactor snapshot

Commit **`308fb40`** ("deleted v1") is the last state before the single-path simplification. Recover
any removed component from there, e.g.:

```bash
git show 308fb40:src/tdi/v2/model.py              # Lightning module, GaussianNLL, contrastive head
git show 308fb40:src/tdi/v2/quantizer_gradients.py # rotation-trick gradient estimator
git checkout 308fb40 -- src/tdi/v2/model.py        # restore a file into the working tree
```

## What lives only in history now

- **GaussianNLL** reconstruction (`var_*` heads) and **self-reconstruction** (`mu_self`).
- **Contrastive** learning (source/target projectors, learnable `logit_scale`).
- **Usage-entropy** regularizer and the **quantizer warmup / aux-ramp** curriculum.
- The **rotation-trick** gradient estimator (`quantizer_gradients.py`); the core now uses the
  straight-through estimator only.
- **PyTorch Lightning** training loop, bf16/AMP autocast, and the cosine+warmup LR schedule.
- Coordinate / descriptor **augmentation** (jitter) and the alignment-balanced batch sampler.

The core path keeps the quantizer performance machinery (EMA-VQ with dead-code replacement, k-means
init, FSQ comparator) and the data-robustness/audit trail.
