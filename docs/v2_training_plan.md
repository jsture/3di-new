# v2 Training — Implementation Plan (Claude Code)

Goal: improve training mechanics and data curriculum, not architecture. **No new input
features** — the 10 geometric descriptors (7 angle cosines, Cα distance, clipped + log signed
sequence separation) stay as-is to keep the alphabet purely structural and keep improvements
attributable to the discretizer.

Targets: `src/tdi/v2/model.py`, `scripts/train_v2.py`, `src/tdi/v2/training_data.py`.
This plan **supersedes Task 1 of `v2_tier1_plan.md`** (first-batch k-means): k-means now runs
*after* a continuous warmup, not on an untrained encoder.

---

## Phase 0 — Already implemented, verify only (no code change)

- **Train-only frozen scaler.** `fit_standardizer` floors std at 1e-6; `PairDataset` reuses
  train mean/std for val; buffers are exported with the model. Confirm this path is untouched.
- **Held-out checkpointing / early stopping.** `train_v2.py` already early-stops and saves
  top-k on `val_score`. Confirm it stays; do not add a fixed-epoch path.

---

## Phase 1 — Staged discretization curriculum (the centerpiece)

### Task 1.1 — Continuous warmup (quantizer bypass)
**File:** `src/tdi/v2/model.py` (`TdiV2Model`)
**Why:** let encoder→decoder reach a useful latent manifold before any hard assignment; this is
the precondition that makes k-means init meaningful and reduces early collapse.
**Do:**
- Add hyperparameters `quantizer_warmup_epochs: int = 2` and `aux_ramp_epochs: int = 2`
  (save via `save_hyperparameters`).
- Give `forward` a `quantize: bool = True` arg. When `quantize=False`: `z = encoder(x)`,
  `z_q = z`, `vq_loss = 0`, skip codebook stats (return placeholder `indices`/`usage`/
  `perplexity`); decode `mu_partner`/`mu_self` from `z` as usual.
- In `training_step`, set `quantize = self.current_epoch >= self.quantizer_warmup_epochs` and
  call `self(x, quantize=quantize)`. **Keep `validation_step` always quantized** so val
  reflects real discrete behaviour.
**Accept:** during warmup epochs, training runs with no codebook updates and decreasing
reconstruction loss; quantization activates exactly at `quantizer_warmup_epochs`.

### Task 1.2 — k-means codebook init after warmup (EMA-VQ only)
**File:** `src/tdi/v2/model.py`
**Why:** seed the codebook from *warmed-up* latents, not noise.
**Do:**
- Add `EMAVectorQuantizer.init_codebook(z)` (k-means, `n_clusters = n_states`, fixed seed; run
  on `F.normalize(z)` when `l2_normalize`; set `embedding`, `ema_sum`, `ema_count=1`,
  `initialized=True`). No-op for FSQ.
- In Lightning hook `on_train_epoch_start`: when
  `self.current_epoch == self.quantizer_warmup_epochs` and the quantizer is an
  `EMAVectorQuantizer` and not yet initialized, run the **encoder** over a sample of
  `self.trainer.train_dataloader` (several batches, `torch.no_grad`), collect `z`, call
  `init_codebook`.
**Accept:** codebook is non-random at the moment quantization begins; early post-warmup
`val_dead_states` is lower than a random-init run. FSQ runs unaffected.

### Task 1.3 — Ramp the auxiliary-loss weights
**File:** `src/tdi/v2/model.py` (`training_step`)
**Why:** forcing hard discretization and auxiliary pressure at full strength from step 1 locks
in poor code usage.
**Do:**
- Add a helper returning a ramp factor `r = clamp((epoch - warmup) / max(1, aux_ramp_epochs),
  0, 1)` (0 during warmup, rising to 1 over `aux_ramp_epochs` after quantization begins).
- Multiply the commitment/VQ term, `lambda_usage`, and `lambda_contrast` by `r` when forming
  `total_loss`. Partner-prediction loss is always on at full weight.
**Accept:** logged `train_vq_loss`, usage, and contrastive weights are ~0 during warmup and
reach full strength `aux_ramp_epochs` later.

---

## Phase 2 — Batch composition

### Task 2.1 — Per-alignment cap at build time
**File:** data build (`scripts/create_training_data.py` / `tdi.data`)
**Why:** long alignments emit disproportionately many residue pairs.
**Do:** set `max_pairs` (already supported by `align_features`) to a conservative default
(512–1024) in the build config. No new model code.
**Accept:** no single alignment contributes more than the cap to the processed array.

### Task 2.2 — Alignment-aware batch sampler
**Files:** `src/tdi/v2/training_data.py`, `scripts/train_v2.py`
**Why:** flat random sampling fills batches with correlated residues from one structure pair,
degrading in-batch contrastive negatives and code-usage statistics.
**Do:**
- Requires an `alignment_id` per training row (from the data-layer metadata — see
  `v2_data_layer_plan.md` Task 1.7). Load it alongside `data.npy`.
- Implement a `torch.utils.data.Sampler`/`BatchSampler` that draws several distinct alignments
  per batch, then pairs within them, so each batch spans many alignments. Pass via
  `DataLoader(batch_sampler=...)`.
**Accept:** sampled batches contain residues from many alignments (assert distinct-alignment
count per batch ≥ a threshold); training is reproducible under a fixed seed.

### Task 2.3 — Gradient accumulation
**File:** `scripts/train_v2.py`
**Do:** add CLI arg `--accumulate_grad_batches` (default e.g. 4) and pass to `L.Trainer`.
**Why:** larger effective batch → better contrastive negatives and code-usage stats without
more memory.
**Accept:** effective batch size scales with the flag; metrics stable.

---

## Phase 3 — Logging & augmentation hygiene

### Task 3.1 — Decompose and log every objective term + per-feature-group recon
**File:** `src/tdi/v2/model.py` (`training_step`)
**Why:** catch the insidious failure where loss looks fine because the model fits the easy
sequence-distance dims and ignores angular geometry.
**Do:**
- `self.log` each component separately: `loss_partner`, `loss_self`, `loss_commitment`/`vq`,
  `loss_usage`, `loss_contrast`, plus `loss_total`.
- Log per-group reconstruction (smooth_l1 of `mu_partner` vs `y`) using the descriptor layout:
  angles `[0:7]`, Cα distance `[7:8]`, sequence `[8:10]` →
  `recon_angles`, `recon_ca_distance`, `recon_sequence`.
**Accept:** all components appear in logs each epoch; the three group reconstructions sum-weight
consistently with the total partner loss.

### Task 3.2 — Prefer coordinate jitter; descriptor jitter off by default
**File:** `src/tdi/v2/training_data.py` (overlaps `v2_data_layer_plan.md` Task 0.3 — do once)
**Why:** coordinate-level noise preserves geometric consistency; descriptor-space noise can
fabricate impossible angle/distance combinations.
**Do:** expose `coordinate_jitter_std` (small, e.g. 0.05–0.20 Å, default 0.0) applied in
`extract_features`; keep `descriptor_jitter_std` default 0.0 and labelled experimental.
**Accept:** default run uses no augmentation; enabling coordinate jitter recomputes descriptors
from perturbed coordinates rather than perturbing standardized features.

---

## Verification (final)
1. `make` lint/type/test green; add unit tests: warmup epoch bypasses the quantizer; k-means
   fires exactly once at the warmup boundary and is a no-op for FSQ; ramp factor is 0 in warmup
   and 1 after `aux_ramp_epochs`; alignment-aware batches span ≥N alignments; all loss
   components are logged.
2. Smoke train a handful of epochs: confirm warmup → quantization transition is visible in the
   logs (codebook activates, `dead_codes_replaced`/usage behave), no NaNs.
3. Judge any before/after by validation aligned-state MI and substitution-matrix / retrieval
   quality — not reconstruction loss alone.

## Implementation order
1.1 → 1.2 → 1.3  (curriculum) → 3.1 (logging, cheap, do early in practice) → 2.1 → 2.3 → 2.2
(needs metadata) → 3.2.

## Deferred (sound, but gate on diagnostics — do NOT build now)
- Inverse-frequency **sampling weights** and per-domain / per-fold-pair caps (`WeightedRandom
  Sampler` by fold/superfamily/seq-sep). Store metadata now (data-layer plan); balance only if
  `report.md` histograms show real skew — structural states are not naturally uniform.
- Second validation view (`val_balanced` alongside `val_raw`).
- Any **new input features** (AA identity, SS, exposure, pLDDT, PLM/DSSP, graph features) — out
  of scope; would confound the discretizer and break the Foldseek-style design.
