# Overhaul v2: model training upgrades, reproducible data pipeline, and discretization curriculum

## Summary
Reworks the v2 VQ-VAE training stack across three fronts — modern optimizer/quantizer mechanics, a reproducible and inspectable preprocessing pipeline, and a staged discretization curriculum — without changing the architecture or the 10 geometric input descriptors. Improvements stay attributable to the discretizer. Implements the three plans under `docs/` (tier-1, data-layer, training).

## Highlights

### 1. Tier-1 model upgrades (`src/tdi/v2/model.py`, `scripts/train_v2.py`)
- k-means codebook init for EMA-VQ (sklearn), seeded from real encoder outputs (no-op for FSQ).
- LR schedule tied to `estimated_stepping_batches`: linear warmup (`warmup_ratio`) → cosine, stepped per optimizer step.
- AdamW parameter groups: weight decay only on ≥2-D weights, none on bias/LayerNorm.
- Modernized contrastive head: learnable `logit_scale` clamped ≤ log(100), symmetric CLIP-style loss (guarded by `lambda_contrast>0`).
- bf16 mixed precision via `--precision`, with fp32-safe quantizer internals (distance/argmin/EMA + gaussian NLL forced fp32).

### 2. Phase-0 correctness fixes (`src/tdi/v2/training_data.py`)
- Feature cache key now includes `virt_cb` + version/convention tags (was path-only → stale features).
- `move_CB` no longer mutates cached coords (move a copy; cache raw parsed coords).
- Deterministic per-item dataset jitter seeded from `(seed, idx, epoch)` — identical batches across `num_workers`. Split knobs: `coordinate_jitter_std` (extract) vs `descriptor_jitter_std` (dataset, experimental, default off).

### 3. Reproducible preprocessing pipeline (`src/tdi/data/`, `configs/data/scop.yaml`)
- `python -m tdi.data <build-features|validate|report> --config ...`; reuses `align_features`/`fit_standardizer`.
- Declarative YAML config + typed loader + stable config hash.
- Immutable, versioned output layout (refuses to overwrite a populated dir); `manifest.json` records input/output sha256, params, git commit, config hash; sorted iteration + stable `alignment_id`/`row_id` ⇒ record-identical reruns.
- First-class `report.json`/`report.md` (stage-drop reconciliation, feature stats, sequence-separation + Cα-distance histograms, per fold/superfamily/alignment).
- `structures.parquet` QC table; CIGAR-semantics validation; pair-metadata parquet with SCOP fold/superfamily/family joins; generated `DATACARD.md`.
- Adds `pyyaml`; legacy `create_training_data.py` gets a pointer to the new pipeline.

### 4. Staged discretization curriculum + batch composition (`src/tdi/v2/model.py`, `scripts/train_v2.py`)
- Continuous warmup: `forward(quantize=False)` bypasses the codebook for `quantizer_warmup_epochs`; validation stays always quantized. Supersedes the tier-1 first-batch k-means (which ran on an untrained encoder) — k-means now fires once at the warmup boundary via `on_train_epoch_start`, seeded from warmed-up latents.
- Auxiliary losses (commitment/VQ, usage entropy, contrastive) ramp 0→1 over `aux_ramp_epochs`; partner prediction always full strength.
- Full objective decomposition in logs + per-feature-group reconstruction (`recon_angles` [0:7], `recon_ca_distance` [7:8], `recon_sequence` [8:10]).
- `AlignmentBatchSampler` so each batch spans many distinct alignments (better contrastive negatives / usage stats); wired via `--alignments_per_batch`. Added `--accumulate_grad_batches`; per-alignment cap default (`max_pairs_per_alignment: 768`).

## Testing
- `68` tests passing (`tests/test_v2_training.py`, `tests/test_v2_detailed.py`, `tests/test_v2_data_layer.py`); ruff + pyright clean.
- Coverage includes: k-means init + FSQ no-op, optimizer param groups, warmup→cosine schedule, symmetric contrastive no-op, cache-key disambiguation, `move_CB` non-mutation, worker-invariant jitter, manifest determinism, metadata/array row-count match, CIGAR rejection, warmup bypass, k-means at boundary, ramp factor, sampler alignment spread, component logging.
- Smoke trains under both precisions and across the warmup→quantization transition: codebook activates at the boundary, aux ramp climbs 0→1, no NaNs.

## Notes for reviewers
- Codebook changes apply to the EMA-VQ backend only; FSQ is a no-op throughout.
- Judge before/after by validation aligned-state MI and substitution-matrix / retrieval quality — not reconstruction loss alone.
- Full SCOPe pipeline run needs the real `data/external/...` PDBs + pairfiles (validated here on synthetic structures).
- Deferred by design: sampling weights / balancing, second validation view, DVC, any new input features.
