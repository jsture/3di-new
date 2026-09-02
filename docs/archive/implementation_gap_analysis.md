# 3Di v2 Implementation Gap Analysis

This document details the missing implementations and unresolved decisions identified across the v2 planning documents. While the core PyTorch model architecture (VQ/FSQ, layers, tests) is mostly complete and functional, the data pipeline has critical structural gaps that will compromise model validity and reproducibility if not addressed before benchmarking.

## 1. Critical Data Leakage: Splitting Strategy
**The Problem:** Currently, `train_v2.py` loads a single massive `.npy` array of paired features and applies PyTorch's `random_split` to create training and validation datasets.
**Why it matters:** Residue pairs from the same protein or structural alignment are highly correlated. Randomly splitting at the residue-pair level guarantees that training and validation datasets will contain nearly identical local structural environments. The validation loss will artificially look good, but the model's true generalization ability will be unknown.
**Required Action:**
- Implement a group-aware split (e.g., by SCOP fold/superfamily or structure ID) *before* pair extraction.
- Generate completely disjoint train and validation `.npy` pairfiles.
- Remove `random_split` from `train_v2.py`.

## 2. Critical Data Leakage: Standardization & Augmentation
**The Problem:** The `PairDataset` in `train_v2.py` calculates feature standardization statistics (mean/std) on the entire dataset *before* `random_split` separates out the validation set. Additionally, coordinate jittering (augmentation) is applied within `PairDataset` without distinguishing between train and validation modes.
**Why it matters:** The model is currently learning scaling parameters derived from the validation set, and the validation set is being artificially perturbed by jittering, making validation metrics noisy and biased.
**Required Action:**
- Fit the standardizer strictly on the isolated training split.
- Ensure that `jitter_std` is set to `0.0` when constructing the validation `PairDataset` or handled explicitly to bypass augmentation during evaluation.

## 3. Data Provenance & Metadata (Parquet)
**The Problem:** The preprocessing pipeline dumps raw coordinate features into `.npy` files without saving any structural metadata (e.g., Structure IDs, residue indices, CIGAR strings, alignment scores).
**Why it matters:** If the model learns a collapsed state or exhibits strange behavior, it is currently impossible to trace an anomalous feature pair back to its source PDB file or sequence alignment.
**Required Action:**
- Modify `create_training_data.py` to output a tabular metadata file (e.g., `.parquet` or `.tsv`) alongside the `.npy` arrays, preserving exact provenance for every training row.

## 4. Missing SCOP Baseline Contract
**The Problem:** The planned `scripts/check_scop_baseline_data.sh` and `scripts/create_scop_pairfile_partitions.sh` do not exist.
**Why it matters:** The model relies on raw structural alignment outputs (`tmaln-06.out`). Without explicit partitioning scripts, unassigned rows, stale SIDs, and cross-split domain pairs remain in the training pool, breaking the reproducible baseline.
**Required Action:**
- Implement the bash scripts outlined in `scop_baseline_data_contract.md` to safely partition pairfiles before downstream processing.

## 5. C-Alpha Distance Filter (Unresolved Decision)
**The Problem:** The `filter_ca_distance` function in `training_data.py` currently raises a `NotImplementedError` because filtering raw PDB coordinates by distance is meaningless unless they have been structurally superposed.
**Why it matters:** The Foldseek paper recommends removing aligned pairs that are structurally divergent (> 5Å). Without this, the VAE is penalized for failing to predict structural states that are not genuinely homologous.
**Required Decision & Action:**
- Decide whether to parse the affine transformation matrices from the upstream TM-align outputs to perform the superposition in Python, OR trust that upstream alignment already filtered these out.
- Implement or explicitly remove the filter.

## 6. Dataset Balancing & Reporting
**The Problem:** Long proteins and overrepresented folds contribute vastly more residue pairs than short or rare proteins. No capping or weighted sampling is implemented. Furthermore, no JSON preprocessing reports are generated.
**Why it matters:** The learned alphabet will overfit to common folds and dense alignments.
**Required Action:**
- Add a maximum pairs-per-alignment cap during dataset generation.
- Implement preprocessing scripts that write `training_data_report.json` to audit input counts, NaNs, and split distributions.
