# Preprocessing Pipeline (tdi.data) Remaining Items - Implementation Plan

This plan addresses the remaining six data-layer items that span both `tdi.data` and `tdi.v2`, ensuring unified logic, correct CIGAR validation, deduplicated parsing, and robust provenance tracking under Python 3.13 constraints.

---

## Proposed Changes

### 1. Canonical CIGAR Column Centralization
**Files to modify:**
- `src/tdi/v2/util.py` (Helper definition)
- `src/tdi/data/pipeline.py` (Pairfile reading)
- `src/tdi/v2/submat.py` (Accumulate counts parsing)
- `src/tdi/v2/cli.py` (Evaluate CLI parsing)

**Do:**
- Introduce a centralized helper `parse_pairfile_line(line: str) -> tuple[str, str, str] | None` in [util.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/src/tdi/v2/util.py):
  ```python
  def parse_pairfile_line(line: str) -> tuple[str, str, str] | None:
      """Parse a pairfile line into (sid1, sid2, cigar_string).

      Assumes the last column is the CIGAR string, supporting both 3-column
      and multi-column alignments (e.g., tmaln-06.out).
      """
      parts = line.strip().split()
      if len(parts) >= 3:
          return parts[0], parts[1], parts[-1]
      return None
  ```
- Refactor all pairfile readers to use `parse_pairfile_line`.

**Acceptance:**
- Unified parsing path for both 3-column files and 10-column files.
- Unit test verification with a dummy alignment containing extra columns.

---

### 2. Unify PDB Path Resolution with Extension Fallback
**Files to modify:**
- `src/tdi/v2/util.py` (Helper definition)
- `src/tdi/data/structures.py` (QC lookup)
- `src/tdi/v2/training_data.py` (Feature extraction)
- `src/tdi/v2/cli.py` (Process PDB files)

**Do:**
- Add `resolve_pdb_path(pdb_dir: str | Path, sid: str) -> Path` in [util.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/src/tdi/v2/util.py):
  ```python
  def resolve_pdb_path(pdb_dir: str | Path, sid: str) -> Path:
      """Resolve the path of a PDB file, checking for sid first, then sid.pdb."""
      path1 = Path(pdb_dir) / sid
      if path1.exists():
          return path1
      path2 = Path(pdb_dir) / f"{sid}.pdb"
      if path2.exists():
          return path2
      return path1  # Fallback to the first path if neither exists
  ```
- Unify structures, training feature extraction, and CLI evaluation paths to use `resolve_pdb_path`.
- Pass `resolved_path.name` and `str(resolved_path.parent)` to `process_pdb` in `v2/cli.py` to maintain compatibility with `process_pdb` signature.

**Acceptance:**
- Files named `sid.pdb` resolve correctly during feature builds and QC, avoiding empty splits.

---

### 3. Wire CIGAR Validation & Deduplicate Structure Parsing
**Files to modify:**
- `src/tdi/data/config.py` (Config schema updates)
- `src/tdi/data/validate.py` (Support prebuilt structure table & format messages)
- `src/tdi/data/pipeline.py` (Optimize build/validation loop)
- `src/tdi/data/structures.py` (PDB QC caching)

**Do:**
- Add `validate_cigars: bool = True` to `PreprocessingConfig` in [config.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/src/tdi/data/config.py).
- Update `validate_dataset` in [validate.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/src/tdi/data/validate.py) to accept an optional `prebuilt_structures: pd.DataFrame | None = None`. Map residue counts from the prebuilt table if available.
- Prefix CIGAR validation errors with row/alignment identifier:
  `Row {source_row} ({sid1} aligned to {sid2}): {exc}`
- Update `build_features` in [pipeline.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/src/tdi/data/pipeline.py) to:
  1. Build the structures table once for all referenced SIDs.
  2. If `validate_cigars` is enabled, validate the entire dataset using the prebuilt structures table *before* feature extraction.
  3. Write the prebuilt `structures` table directly at the end, eliminating the redundant second `build_structures_table` call.
- Add an `mtime`/`size` cached lookup dict inside `structure_qc` in [structures.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/src/tdi/data/structures.py) to cache parsed structure records and accelerate duplicate parsing in unit tests.

**Acceptance:**
- Out-of-range CIGARs fail the build early with a `CigarValidationError` naming the exact row and alignment.
- Structures are parsed exactly once during a combined run.

---

### 4. Meaningful Completeness Flags (Standard Residues only)
**Files to modify:**
- `src/tdi/data/structures.py` (`structure_qc` completeness computation)

**Do:**
- Filter to standard residues (where `len(res.id[0].strip()) == 0`) before evaluating missing coordinates:
  ```python
  is_standard = np.array([len(r.id[0].strip()) == 0 for r in residues], dtype=bool)
  if len(is_standard) > 0 and is_standard.any():
      record["has_missing_ca"] = bool(np.isnan(coords[is_standard, 0:3]).any())
      record["has_missing_backbone"] = bool(np.isnan(coords[is_standard, 6:12]).any())
  else:
      record["has_missing_ca"] = True
      record["has_missing_backbone"] = True
  ```
- Keeps the boolean return type to avoid schema breakages in Parquet consumers.

**Acceptance:**
- Structures with HETATM/water rows but complete standard backbones do not get falsy flags.

---

### 5. CLI Polish, Quantiles, and Provenance Trackers
**Files to modify:**
- `src/tdi/data/hashing.py` (Add git working tree dirty status check)
- `src/tdi/data/pipeline.py` (Include git_dirty in manifest)
- `src/tdi/data/cli.py` (Add new CLI overrides)
- `src/tdi/data/report.py` (Report quantiles for alignments)
- `tests/test_v2_data_layer.py` (Fix test regex assertion)

**Do:**
- Implement `git_dirty() -> bool` via `git status --porcelain` and output it as `git_dirty` in `manifest.json`.
- Add `--virtual_center` (3 floats) and `--fail_on_skipped` (action="store_true", default=None) overrides to the common data parser.
- Change `examples_per_alignment` in [report.py](file:///Users/skn506/Documents/Claude/Projects/3di-new/src/tdi/data/report.py) to a compiled quantile dictionary (`min`, `p25`, `median`, `p75`, `p90`, `p95`, `p99`, `max`, `mean`, `std`, `count`) rather than storing all alignment IDs.
- Fix the regex expectation in `test_build_features_fail_on_skipped_alignments` to match the actual exception message.

**Acceptance:**
- Smaller, cleaner `report.json` output files.
- Working CLI overrides matching the YAML schema.
- Green test suite.

---

## Verification Plan

### Automated Tests
- Run `uv run pytest` to check correctness.
- Run `uv run pyright src/tdi` to verify types.
- Run `uv run ruff check` to ensure syntax formatting.

### Manual Verification
- Validate config override commands:
  `uv run python -m tdi.data build-features --config configs/data/scop.yaml --fail_on_skipped --virtual_center 270.0 0.0 2.0`
- Confirm `manifest.json` writes `git_dirty` and `report.json` uses quantiles.
