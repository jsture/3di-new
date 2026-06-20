# Experiments quarantine

The v2 simplification removed several auxiliary objectives and training-framework features from the
core path. The pieces that are **self-contained** (no coupling to the slimmed model/loop) live here
as runnable snapshots; the rest are coupled to the old model/training loop and stay in git history.

## Reintroduced runnable snapshots

Self-contained, depend only on numpy/torch, and covered by `tests/test_experiments.py`:

- `rotation_trick.py` — the rotation-trick (Householder) and straight-through gradient estimators
  via `apply_quantizer_gradient(z, z_q, mode=...)`.
- `augmentation.py` — `jitter_coords`, coordinate-level Gaussian augmentation.
- `alignment_batch_sampler.py` — `AlignmentBatchSampler`, several distinct alignments per batch.

These are not imported by the core `tdi.v2`; they exist only to keep the removed mechanisms
reproducible in isolation.

## Pre-refactor snapshot

Commit **`308fb40`** ("deleted v1") is the last state before the single-path simplification. Recover
any removed component from there, e.g.:

```bash
git show 308fb40:src/tdi/v2/model.py              # Lightning module, GaussianNLL, contrastive head
git show 308fb40:src/tdi/v2/quantizer_gradients.py # rotation-trick gradient estimator
git checkout 308fb40 -- src/tdi/v2/model.py        # restore a file into the working tree
```

## What lives only in history now

These are coupled to the old model/training loop, so they are not reproducible in isolation and
stay in git history rather than this directory:

- **GaussianNLL** reconstruction (`var_*` heads) and **self-reconstruction** (`mu_self`).
- **Contrastive** learning (source/target projectors, learnable `logit_scale`).
- **Usage-entropy** regularizer and the **quantizer warmup / aux-ramp** curriculum.
- **PyTorch Lightning** training loop, bf16/AMP autocast, and the cosine+warmup LR schedule.

(The rotation-trick gradient and the jitter/batch-sampler pieces *were* self-contained and are
reintroduced above rather than left in history.)

The core path keeps the quantizer performance machinery (EMA-VQ with dead-code replacement, k-means
init, FSQ comparator) and the data-robustness/audit trail.
