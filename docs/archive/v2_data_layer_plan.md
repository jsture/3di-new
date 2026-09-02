# v2 Data Layer — Implementation Plan (Claude Code)

Goal: make preprocessing reproducible, inspectable, and not silently wrong, around the current
SCOPe baseline. **No new corpus.** Model code (`model.py`) is out of scope.

The collaborator's 21 suggestions are sound in direction. This plan keeps the concrete,
non-speculative ones and **defers** the items the collaborator themselves gated on diagnostics
or marked experimental (see "Deferred" at the end). Two items are real correctness bugs,
verified in code, and go first.

---

## Phase 0 — Correctness fixes (do first; small, surgical)

### Task 0.1 — Fix the feature cache key  *(verified bug)*
**File:** `src/tdi/v2/training_data.py`, `extract_features` (lines ~63–92)
**Problem:** cache is keyed on `pdb_path` only. Two runs in one process with different
`virt_cb` return stale features.
**Do:**
- Build an explicit key and use it for get/set:
  ```python
  cache_key = (
      os.path.abspath(pdb_path),
      tuple(float(x) for x in virt_cb),
      "features_v2",          # feature-definition version tag
      "seq_delta_j_minus_i",  # convention tag
  )
  ```
- Keep the existing rule: only read/write the cache when `jitter_std == 0.0`.
**Accept:** calling `extract_features` on the same PDB with two different `virt_cb` returns
two different feature arrays.

### Task 0.2 — Stop in-place coordinate mutation  *(verified)*
**File:** `src/tdi/v2/training_data.py`, `extract_features` (line ~76); `features.move_CB`
**Problem:** `move_CB` mutates its input and returns the same object, so the array cached as
"coords" (line 92) is actually the CB-moved array, aliased with `coords_moved`.
**Do:**
- `coords_moved = features.move_CB(coords.copy(), virt_cb=virt_cb)`.
- Cache the **raw parsed** coords, not the moved ones: `FEATURE_CACHE[cache_key] = (vae_features, valid_mask2, coords)` where `coords` is the pre-move array. (CA columns 0:3 used downstream by `filter_ca_distance` are unaffected by the move, so this is safe and now correct.)
**Accept:** the cached coords equal the parser output for CA columns; no caller observes a
mutated input array.

### Task 0.3 — Reproducible, worker-safe jitter + split the two jitter types
**File:** `src/tdi/v2/training_data.py`, `PairDataset`; `scripts/train_v2.py`
**Problem:** `PairDataset` draws noise from a stateful `self.rng` inside `__getitem__`. With
`num_workers > 0` this duplicates/forks RNG state and is non-reproducible. Coordinate jitter and
descriptor-space jitter are also conflated under one `jitter_std`.
**Do:**
- Replace stateful sampling with deterministic per-item seeding, e.g. derive a generator from
  `(base_seed, idx, epoch)` so noise is identical regardless of worker count.
- Split the knobs: `coordinate_jitter_std` (applied at coordinate level in `extract_features`)
  vs `descriptor_jitter_std` (the `PairDataset` path). Default `descriptor_jitter_std = 0.0`
  and label it experimental in the docstring/config.
**Accept:** two runs with the same seed and `num_workers ∈ {0, 4}` produce identical batches.

---

## Phase 1 — Reproducible preprocessing pipeline (the core upgrade)

### Task 1.1 — Declarative YAML config
**New:** `configs/data/scop.yaml`
```yaml
dataset:
  name: scop_baseline
  pdb_dir: data/external/foldseek_scop40/pdb_by_sid
  train_pairfile: data/derived/pairfiles/tmaln-06.train.out
  val_pairfile: data/derived/pairfiles/tmaln-06.val.out
features:
  virtual_center: [270.0, 0.0, 2.0]
  sequence_delta_convention: j_minus_i
  max_ca_dist: 5.0
sampling:
  max_pairs_per_alignment: null
  seed: 123
outputs:
  out_dir: data/processed/scop_ca5_v1
```
All preprocessing reads this config; CLI args may override but the resolved config is recorded.

### Task 1.2 — `tdi.data` pipeline CLI
**New module:** `src/tdi/data/` with `python -m tdi.data <subcommand> --config ...`:
- `build-features` — extract features, apply filters, write arrays + metadata + scaler.
- `validate` — structure QC + CIGAR-semantics checks (Tasks 1.5, 1.6).
- `report` — emit `report.json` / `report.md` (Task 1.4).
Reuse existing `align_features` / `fit_standardizer`; do not duplicate logic.
(Existing `scripts/create_training_data.py` becomes a thin wrapper or is folded in.)

### Task 1.3 — Immutable, versioned processed-dataset layout + manifest with hashes
**Output dir** (immutable once written; refuse to overwrite a populated `out_dir`):
```
data/processed/<name>/
  manifest.json   train_pairs.npy   train_targets.npy   val_pairs.npy   val_targets.npy
  scaler.npz      train_metadata.parquet   val_metadata.parquet
  structures.parquet   report.json   report.md   DATACARD.md
```
`manifest.json` records: dataset name; each **raw input** path + sha256 (pdbs, scop_lookup,
tmaln, pairfiles); resolved `preprocessing` params (max_ca_dist, max_pairs, standardization,
sequence_delta_convention, virtual_center); and for **every output array** sha256, shape,
dtype, n_rows; plus `git_commit`, `config_hash`, `created_at`.
**Determinism contract:** sort pairfile iteration; stable IDs
```python
alignment_id = f"{sid1}:{sid2}:{source_row}"
row_id = sha256(f"{alignment_id}:{idx1}:{idx2}:{direction}".encode()).hexdigest()
```
so identical inputs+config produce record-identical outputs.
**Accept:** running the pipeline twice on the same inputs yields identical output hashes in
`manifest.json`.

### Task 1.4 — Preprocessing report (first-class artifact)
**Output:** `report.json` + `report.md`. Pull stage counts straight from the `meta` dict that
`align_features` already returns, aggregated across alignments:
- rows read / rows skipped; pairs before filters → after descriptor-validity → after Cα filter
  → after max-pair cap → final bidirectional examples;
- feature mean/std/min/max;
- **sequence-separation histogram** (bins: 1, 2–4, 5–12, 13–24, 25–64, >64) and
  **Cα-distance histogram** (these are the cheap diagnostics the collaborator wanted first);
- examples per fold / per superfamily / per alignment.
**Accept:** the sum of per-stage drops reconciles with raw rows in minus final examples out.

### Task 1.5 — Structure-level QC table
**Output:** `structures.parquet`, one row per `sid`: `path, n_residues, n_valid_residues,
valid_fraction, n_chains, selected_chain, has_missing_ca, has_missing_backbone, sha256,
parse_status`. Make chain selection explicit (record that "first chain" was used and how many
existed); note insertion codes / altlocs / hetero residues skipped.
**Accept:** every input structure appears with a `parse_status`; domains that yield no features
are explainable from this table.

### Task 1.6 — CIGAR-semantics validation
**In `tdi.data validate`:** assert parsed index pairs are within bounds, non-negative,
in-range, and that pair counts are consistent with the custom `P`=aligned-pair convention.
Fail loudly on violation.
**Accept:** a malformed/foreign-tool CIGAR is rejected with a clear error rather than silently
producing bad pairs.

### Task 1.7 — Pair metadata to Parquet
**Output:** `train_metadata.parquet` / `val_metadata.parquet` with columns:
`row_id, sid_source, sid_target, idx_source, idx_target, alignment_id, source_pairfile_row,
ca_dist_superposed, ca_dist_raw, source_is_forward, scop_source, scop_target, fold_source,
fold_target, superfamily_source, superfamily_target, family_source, family_target`.
Most already exist in the `align_features` meta dict; join SCOP/fold/superfamily/family from the
lookup. Adds a `pyarrow` dependency.
**Accept:** row count of metadata == row count of the pairs array; metadata is filterable by
fold and Cα-distance bin.

### Task 1.8 — Data card
**Output:** `DATACARD.md` per processed dataset: source files, filters, known exclusions,
counts (domains / alignments / residue pairs), feature definitions, intended and not-intended
use. Can be generated from `manifest.json` + `report.json`.

---

## Verification (final)
1. `make` lint/type/test targets green; add unit tests for: cache key disambiguates `virt_cb`;
   `move_CB` input not mutated; jitter identical across `num_workers ∈ {0,4}`; manifest output
   hashes stable across two runs; metadata row count matches array length; CIGAR validator
   rejects an out-of-range pair.
2. Run the full pipeline on the SCOPe baseline via `configs/data/scop.yaml`; confirm
   `report.md`, `structures.parquet`, and `manifest.json` are produced and internally consistent.
3. Re-run; confirm record-identical outputs (same hashes).

## Implementation order
0.1 → 0.2 → 0.3  (correctness) → 1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6 → 1.7 → 1.8.

## Deferred (sound, but gate on the diagnostics above — do NOT build now)
- **Balancing / sampling weights / per-domain & per-fold caps** and `WeightedRandomSampler` by
  fold/superfamily/seq-sep bin. The Parquet metadata (1.7) is what makes these testable later;
  implement only if the histograms in `report.md` show real imbalance.
- **Second validation view** (`val_balanced` alongside `val_raw`) — depends on balancing.
- **DVC / external artifact versioning** — the manifest+hashes (1.3) suffice for the current
  small artifacts; adopt DVC only once processed arrays / structure archives grow large.
- **Descriptor-space jitter** — kept behind `descriptor_jitter_std`, default off, experimental.
