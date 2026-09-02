# v2 Simplification Plan Verification Report

This report summarizes the verification of Steps 1 through 7 of the [v2_simplification_plan.md](file:///Users/skn506/Documents/Claude/Projects/3di-new/docs/v2_simplification_plan.md) conducted on June 20, 2026.

---

## Executive Summary

- **Ruff & Pyright Checks**: **Passed**. `uv run ruff check` and `uv run pyright src/tdi` return zero errors, warnings, or informationals.
- **Phases 0–4**: **Successfully Implemented**. The core VQ-VAE model, custom quantizers, plain training loop, dataset pruning, self-describing export config, evaluation pipeline, and canonical parsing are fully aligned with the simplification plan.
- **Phase 5 (Metadata & Reports)**: **Correctly Implemented in Source, Outdated Test Invariants**. The data pipeline in `src/tdi/data` correctly produces the lean, metadata-parquet and joint `report.json` artifacts. However, three tests in `tests/test_v2_data_layer.py` fail because they are still asserting the old data layout structure (expecting the deleted `row_id` column, the separate split reports, and histograms by default).
- **Phase 6 (Delete Legacy Code & Dead Entrypoints)**: **Mostly Implemented, Outdated Documentation**. The `src/tdi/v1` directory and all legacy scripts have been deleted from the repository. However, `README.md` still contains outdated references to the v1 module, PyTorch Lightning, and old loss/regularization objectives.
- **Phase 7 (Standalone Comparison Driver)**: **Fully Implemented and Isolated**. `scripts/compare_quantizers.py` runs VQ and FSQ pipelines and compiles a comparison report. The script is isolated from the core package and is verified by its own test suite. We identified one reporting bug in how it reads the final validation loss.

---

## Detailed Step-by-Step Status

### Phase 0 — Behavior-Lock Golden Tests
* **Status**: **Complete & Green**
* **Verification File**: [test_v2_golden.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/tests/test_v2_golden.py)
* **Implemented Invariants**:
  - `parse_cigar` rejects unsupported ops and converts perfect matches `P` into aligned pair indices.
  - `extract_features` returns `(N, 10)` arrays and maps invalid sequence borders correctly.
  - `filter_ca_distance` superposes structures using Cα columns `0:3` only, and removes bad alignments correctly.
  - `align_features` returns bidirectional outputs in the correct forward-then-reverse order.
  - Train-only feature standardization scaler prevents leakage to validation.
  - Exports correctly enforce state indices within `[0, n_states)`.
  - `evaluate` writes `sequences.txt`, `submat.txt`, and `evaluation_report.json`.

---

### Phase 1 — Carve `model.py` to One Path
* **Status**: **Complete & Compliant**
* **Verification Files**: [model.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/src/tdi/v2/model.py), [quantizers.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/src/tdi/v2/quantizers.py)
* **Key Achievements**:
  - Lightning dependency, BF16/AMP, and non-deterministic autocast logic have been removed. `AlphabetModel` is a plain `nn.Module`.
  - MLP layers are structured as `ResidualMLP` using standard LayerNorm + SiLU blocks.
  - Quantization backends `EMAVectorQuantizer` and `FSQQuantizer` inherit from a unified signature and are instantiated using `make_quantizer`.
  - EMA-VQ enforces cosine lookup (`l2_normalize`), EMA codebook updates, commitment cost loss, and mandatory codebook-replacement to prevent collapse.
  - FSQ quantizes to a grid via `tanh` bounds and a basis index mapping.
  - Single gradient path implemented using Straight-Through Estimators (STE). `gradient_mode` and the rotation trick have been deleted.
  - Alphabet capacity constraint is validated; `n_states > len(letters)` raises a `ValueError` at model initialization.

---

### Phase 2 — Plain Training Loop & Config Slimming
* **Status**: **Complete & Compliant**
* **Verification Files**: [train.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/src/tdi/v2/train.py), [train_config.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/src/tdi/v2/train_config.py)
* **Key Achievements**:
  - Training loop operates without Lightning using raw PyTorch utilities.
  - Standardizer fits on training arrays only and is applied identically to validation.
  - EMA-VQ codebooks can be initialized using a one-shot k-means fit on latents via `init_codebook_from_loader`.
  - Optimizer uses `AdamW` with weight decay disabled on biases and LayerNorm gains. Schedulers default to fixed LR (with optional `CosineAnnealingLR` opt-in, no linear warmup).
  - Validation metrics select the best checkpoint on `val_loss`, and the run directory receives the self-describing export, resolved configuration `run_config.resolved.json`, and training metrics `train_log.csv`.
  - Config schema is pruned of contrastive, temperature, precision, warmup ratio, and jitter parameters.

---

### Phase 3 — Slim Dataset
* **Status**: **Complete & Compliant**
* **Verification File**: [training_data.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/src/tdi/v2/training_data.py)
* **Key Achievements**:
  - Core alignment and extraction functions are kept.
  - Deleted code includes coordinate/descriptor jittering, `AlignmentBatchSampler`, epoch-setting logic, and the per-item dataset RNG.
  - `PairDataset.__getitem__` returns a basic tuple of `(x_scaled[i], y_scaled[i])`. Bidirectional pairs are prepared on the data-side (`make_bidirectional_pairs`), and no symmetric loss double-counting is used in the model or trainer.

---

### Phase 4 — Validation Metrics & Self-Describing Export/Eval
* **Status**: **Complete & Compliant**
* **Verification Files**: [cli.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/src/tdi/v2/cli.py), [encode.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/src/tdi/v2/encode.py), [submat.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/src/tdi/v2/submat.py), [util.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/src/tdi/v2/util.py)
* **Key Achievements**:
  - Validation loop evaluates `val_loss`, `state_counts`, `dead_state_count`, `perplexity`, and VQ-only `vq_margin`.
  - `config.json` stores architectural parameters, the alphabet (`letters`), and `invalid_state` to make the export self-describing.
  - `encode.py` reads `letters` and `invalid_state` from the model's loaded properties rather than module constants, ensuring decoupling.
  - `evaluate` command calculates sequences, counts transitions, builds log-odds substitution scoring matrices, and outputs `sequences.txt`, `submat.txt`, and `evaluation_report.json`.
  - Centralized `util.parse_pairfile_line` is used throughout the build and evaluate pipeline.
  - *CLI Command Syntax*: Argument names in the code use standard pythonic underscores (e.g. `--model_dir`, `--pdb_dir`, `--out_dir`) instead of hyphens.

---

### Phase 5 — Lean Preprocessing Metadata & Report
* **Status**: **Implemented in Source; Unit Tests Need Updating**
* **Verification Files**: [pipeline.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/src/tdi/data/pipeline.py), [report.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/src/tdi/data/report.py)
* **Key Achievements**:
  - `metadata.parquet` columns match the plan exactly (with source/target sids, indices, splits, folds, superfamilies, raw and superposed Cα distances).
  - `alignment_id` is enriched to `{pairfile_stem}:{source_row}:{sid1}:{sid2}`. Standalone `row_id` and `source_pairfile_row` have been dropped.
  - `split_group_*` is mapped to the superfamily grouping used.
  - Duplicated split reports are dropped; only a single `report.json` and `report.md` are written.
  - histograms are gated behind the `--full-report` CLI flag.
* **Test Discrepancies**:
  Running `uv run pytest` yields **3 failures** in `tests/test_v2_data_layer.py` due to outdated test assumptions:
  1. `test_build_features_metadata_row_count_matches_arrays`: Fails because it expects `row_id` (dropped in Phase 5).
  2. `test_build_writes_expected_artifacts`: Fails because it expects `train_report.json` and `val_report.json` (dropped in Phase 5).
  3. `test_report_json_uses_labeled_strict_ca_bins`: Fails because it expects `ca_distance_histogram` without enabling `--full-report`.

---

### Phase 6 — Delete Legacy Code & Dead Entrypoints
* **Status**: **Mostly Complete; Documentation Needs Cleanup**
* **Verification**:
  - The legacy `src/tdi/v1` directory and associated module tests (`tests/test_training.py`) have been entirely deleted from the filesystem.
  - Legacy standalone scripts under `scripts/` (`train.py`, `train_v2.py`, `encode_pdbs.py`, `create_submat.py`, `create_training_data.py`) have been successfully removed.
  - Necessary scripts (`fetch_scop40_structures.py`, `make_splits.py`, `split_folds.py`) have been retained.
  - `CLAUDE.md` has been cleaned up.
  - *Discrepancy*: `README.md` still contains outdated descriptions of the legacy v1 codebase location, and references PyTorch Lightning, Contrastive Learning, and the Rotation Trick (which were all excised from the core v2 codebase in earlier phases).

---

### Phase 7 — Optional Standalone Quantizer-Comparison Driver
* **Status**: **Fully Complete & Isolated**
* **Verification Files**: [compare_quantizers.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/scripts/compare_quantizers.py), [test_compare_quantizers.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/tests/test_compare_quantizers.py)
* **Key Achievements**:
  - `compare_quantizers.py` trains and evaluates VQ and FSQ models side-by-side using the canonical `train_model` and `run_evaluate` entrypoints.
  - It correctly generates and saves `comparison_report.json` and `comparison.md` into the specified root directory.
  - Absolute code-level isolation is maintained; `test_compare_quantizers.py` verifies that no files under `src/tdi/v2/` import `compare_quantizers` or related driver code.
  - The script's logic is fully tested and verified by passing `test_build_comparison_tabulates_two_runs` and `test_core_does_not_import_comparison_driver`.

---

## Architectural & Robustness Review

While the code changes match the requested simplification plan accurately, we identified several opportunities to improve robustness and correctness:

### 1. EMA-VQ L2-Normalization and Centroid Accumulation Mismatch
In `quantizers.py` (:line 175-179):
```python
counts = encodings.sum(dim=0)
sums = encodings.t() @ z.float().detach()

# Update moving averages
self.ema_count.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
self.ema_sum.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)
```
- **The Issue**: When `l2_normalize=True` is enabled, the distance calculations and neighbor lookups are computed in a normalized cosine space. However, `sums` is updated with the *unnormalized* encoder output `z`. Because `z` has arbitrary scale and magnitude, accumulating unnormalized latents into `ema_sum` and subsequently normalizing the centroids at the end of the update step creates a scale mismatch. It mixes historical moving averages (which are scaled unit-vectors) with incoming vectors of arbitrary magnitudes.
- **Robust Approach**: Normalize the latents *before* adding to the EMA accumulator when `l2_normalize` is active:
  ```python
  normalized_z = F.normalize(z.float().detach(), dim=-1) if self.l2_normalize else z.float().detach()
  sums = encodings.t() @ normalized_z
  ```

### 2. Lack of Device-Agnostic Setup in the Training Loop
In `train.py` (lines 187-193):
- **The Issue**: The training loop runs entirely on CPU. The code does not detect if a GPU (`cuda` or `mps`) is available, and it does not move the model or dataset loader batches to a target device. Although the dataset is small and training on CPU is currently fast enough, this limits the codebase's scalability when training on larger alignments or structure databases.
- **Robust Approach**: Auto-detect the execution device or allow configuring it via the configuration/CLI, and transfer models and tensors:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
  model = model.to(device)
  # Inside training loop:
  x, y = x.to(device), y.to(device)
  ```

### 3. Early Stopping Patience during VQ Warmup
In `train.py` (lines 180-216):
- **The Issue**: During the initial epochs of EMA-VQ training, codebook centroid replacement is warming up (governed by `replacement_warmup_steps`, which defaults to 500 steps, or ~1-2 epochs). While centroids are cold and unused codes are being reset, the validation loss might not improve or can remain unstable. Because the patience decrement is active from epoch 0, the model risks triggering early stopping prematurely before the codebook has fully stabilized.
- **Robust Approach**: Bypass decrementing the `patience_left` counter for the first 1-2 epochs (or during the initial `replacement_warmup_steps` steps) to allow the codebook collapse prevention mechanism to settle.

### 4. FSQ Latent Magnitude Drift
In `quantizers.py` (`FSQQuantizer.quantize`):
- **The Issue**: FSQ applies a bounding `tanh` to constrain continuous latents into the `[-1, 1]` grid. Since there is no VQ commitment loss constraint on the encoder outputs `z`, and the straight-through estimator (STE) passes the gradient through `tanh` as a constant `1.0`, the unquantized outputs `z` can grow arbitrarily large. Once `z` is highly saturated (e.g. >10.0), shifting the state assignment requires a massive gradient update, effectively locking the encoder.
- **Robust Approach**: Monitor or apply a small L2 regularization penalty on the pre-quantized latent outputs `z` to prevent magnitude drift.

### 5. Final Validation Loss Mismatch in Comparison Driver (Phase 7)
In `compare_quantizers.py` (`_final_val_loss`):
- **The Issue**: When reading `train_log.csv` to report the final `val_loss`, the script takes the value of the last epoch (`rows[-1]["val_loss"]`). However, if early stopping is triggered, the last logged epoch is **not** the best epoch (the patience decrement steps will have higher validation losses). Because the training loop reloads the **best** weights before exporting the model, the evaluated codebook corresponds to the minimum validation loss, but the comparison report displays the final (worse) epoch's validation loss.
- **Robust Approach**: Update `_final_val_loss` to compute the minimum validation loss across all logged epochs:
  ```python
  return min(float(row["val_loss"]) for row in rows) if rows else None
  ```

### 6. Documentation Sync for excising legacy v1/Lightning modules (Phase 6)
In `README.md`:
- **The Issue**: Although the legacy directory and scripts are deleted, `README.md` still contains documentation and instructions referring to `src/tdi/v1`, PyTorch Lightning, Contrastive Learning, and the Rotation Trick, which are no longer supported.
- **Robust Approach**: Sync the `README.md` to remove these stale references, aligning the user-facing documentation with the single-path v2 architecture as described in Phase 8 of the plan.

### 7. Performance Bottleneck: Redundant Double-Parsing of PDB files
In `structures.py` and `training_data.py`:
- **The Issue**: The preprocessing pipeline parses every PDB file twice. First, `build_structures_table` parses the PDBs to create structural metadata/QC tables. Later, during features generation, `extract_features` parses the same PDB files again to compute structural angles. Since Biopython's PDB parser is CPU-bound and slow, this redundant parsing doubles the data preparation overhead.
- **Robust Approach**: Share the parsed Biopython `Structure` object or combine the QC step and features extraction step into a single PDB parsing pass.

### 8. Memory Footprint of Global Feature Cache
In `training_data.py` (`FEATURE_CACHE`):
- **The Issue**: `FEATURE_CACHE` is a standard Python dict that caches computed coordinate features without any size limits or eviction policy. For massive datasets of structural alignments, storing coordinates for all files in RAM simultaneously can trigger high memory consumption or Out-of-Memory (OOM) errors.
- **Robust Approach**: Use `functools.lru_cache` or a custom sliding-window cache, or clear the cache programmatically after finishing each alignment split.

### 9. Performance Overhead in Dataset Getitem
In `training_data.py` (`PairDataset.__getitem__`):
- **The Issue**: `PairDataset.__getitem__` creates new PyTorch tensors on every batch retrieval via `torch.tensor(numpy_slice)`. Repeatedly converting NumPy array slices to PyTorch tensors in dataloader workers adds significant runtime overhead.
- **Robust Approach**: Convert the entire `self.x_scaled` and `self.y_scaled` arrays to PyTorch tensors once during the dataset's `__init__` constructor using `torch.from_numpy`, and return slices directly in `__getitem__`.

### 10. Security Warnings on Weights Loading
In `model.py` (`AlphabetModel.load`):
- **The Issue**: Model state dict files are loaded with `torch.load(..., map_location="cpu")` without specifying `weights_only=True`. Modern versions of PyTorch flag this as a warning, and it leaves the loader vulnerable to arbitrary code execution if loading untrusted files.
- **Robust Approach**: Pass `weights_only=True` to standard `torch.load` calls:
  ```python
  model.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu", weights_only=True))
  ```

### 11. Robustness in SCOP Lookup Parsing (tdi.data)
In `scop.py` (`load_scop_lookup`):
- **The Issue**: Every line in the SCOP lookup file is split and parsed without checking for headers or comment lines (e.g. lines starting with `#`). If a header line is present, it will be added as a lookup entry (e.g., mapping column headers like `"domain"` to `"classification"`), which is redundant.
- **Robust Approach**: Add a standard check to ignore comments and column header strings:
  ```python
  if line.startswith("#") or "classification" in line:
      continue
  ```

---

## Code Wiring, Redundancy, & Stragglers Review

### 1. Verification of Model Changes
The primary changes to `AlphabetModel`, `EMAVectorQuantizer`, `FSQQuantizer`, and `ResidualMLP` are correctly implemented. The model configuration, quantizer factory setup, scaling logic, and training forward pass align with the plan.

### 2. Analysis of Lost Wiring
- No active training or inference wiring has been lost. The connection between model parameters, features scaling, quantizer codebooks/grids, and partner-prediction is correct.
- **Improved Wiring**: The export mechanism stores critical metadata such as `letters` and `invalid_state` in the model's `config.json`. `encode.py` and `cli.py` correctly load this configuration dynamically, decoupling inference from module-level constants.

### 3. Leftover Stragglers & Redundancies
- **`lightning_logs/` Directory**: A leftover directory remains in the project root containing old tensorboard logs from pre-simplification Lightning training runs. This directory can be safely deleted or ignored.
- **Obsolete Documentation**: The `README.md` file still lists instructions and details about the deleted `src/tdi/v1` package, PyTorch Lightning, Contrastive Learning, and the Rotation Trick. These references must be removed.
- **Test File Invariants**: The unit tests in `tests/test_v2_data_layer.py` still assert older pipeline parameters and outputs (such as expecting `row_id` column, separate `train_report.json`/`val_report.json`, and un-gated histograms by default). They must be updated to expect Phase 5 layout conventions.

### 4. Location of Removed Experiments (Rotation Trick, Jitter, Sampler, etc.)
- **Current State**: The experiments (including the Householder rotation trick, staged discretization curriculum, alignment-aware batch sampler, and coordinate jittering) have been completely removed from the active codebase. They live **exclusively in the repository's git history** (before the simplification refactor commenced).
- **Commit Reference**: The pre-refactor state can be recovered or referenced at commit `20f6f7b` (or branches like `simplify` / `v2-wings` in the repository).
- **Suggestion**: In alignment with Phase 8 of the simplification plan, create an `experiments/README.md` file pointing to these commit hashes/tags so future developers can easily find the legacy files in the Git history.
