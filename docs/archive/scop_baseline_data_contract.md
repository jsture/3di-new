# SCOPe Baseline Data Contract and Action Plan

This document defines the concrete dataset-handling recommendations for the current SCOPe/Foldseek-style baseline in this repository. It is intentionally narrow: it assumes the existing data files are available and should be preserved as the initial reproducible baseline.

The goal is not to regenerate the corpus. The goal is to make the existing corpus auditable, split-safe, and reliable before changing the model or adding new datasets.

## Current observed state

From the checked files:

| Artifact | Count | Interpretation |
|---|---:|---|
| `scop_lookup.tsv` | 11,211 SIDs | Full SCOPe lookup/classification universe in the repo |
| `pdbs_train.txt` | 8,953 logical SIDs | Training domain split |
| `pdbs_val.txt` | 2,207 logical SIDs | Validation domain split |
| train/val overlap | 0 | Train and validation domain lists are disjoint |
| train SIDs missing from lookup | 0 | Every train SID has a SCOP label |
| validation SIDs missing from lookup | 0 | Every validation SID has a SCOP label |
| lookup SIDs not assigned | 51 | Small number of special/unassigned lookup entries |
| `tmaln-06.out` | 24,525 rows | Original filtered structural-alignment pairfile |
| train/train alignments | 20,657 rows | Alignment rows usable for training |
| val/val alignments | 3,814 rows | Alignment rows usable for validation diagnostics |
| train/val cross-split alignments | 47 rows | Should be ignored in the simple baseline |
| unassigned/stale alignment rows | 7 rows | Should be ignored |

The partition sums correctly:

```text
20,657 train/train
 3,814 val/val
    47 train/val cross-split
     7 unassigned/stale
------
24,525 total rows in tmaln-06.out
```

This is coherent enough to freeze as the first baseline. The next improvements should be controlled derived artifacts and stricter residue-pair expansion, not corpus replacement.

---

## Recommendation 1: Freeze the raw input files

### Action

Treat the following files as immutable baseline inputs:

```text
data/pdbs_train.txt
data/pdbs_val.txt
data/scop_lookup.tsv
data/tmaln-06.out
```

Do not edit them in place. If line endings or trailing-newline cleanup is needed, either:

1. make a clearly documented one-time normalization commit, or
2. create normalized derived copies under a separate name.

Recommended derived layout:

```text
data/
  pdbs_train.txt                 # raw baseline split
  pdbs_val.txt                   # raw baseline split
  scop_lookup.tsv                # raw SID -> SCOP classification table
  tmaln-06.out                   # raw alignment pairfile
  derived/
    pdbs_train.normalized.txt
    pdbs_val.normalized.txt
    tmaln-06.assigned.out
    tmaln-06.train.out
    tmaln-06.val.out
    tmaln-06.cross_split.out
    dataset_partition_report.json
```

If the repo convention strongly prefers keeping derived files directly under `data/`, that is acceptable, but the filenames should make derivation explicit.

### Reasoning

The current files are already consistent enough for a baseline. Regenerating them now would change multiple factors at once: domain universe, split definition, pairfile coverage, alignment paths, and training examples. That makes it harder to distinguish model effects from data effects.

Freezing the raw files gives you a stable reference point. All downstream changes can then be evaluated as deliberate transformations.

---

## Recommendation 2: Create explicit partitioned pairfiles

### Action

Generate separate pairfiles for train/train, val/val, cross-split, and assigned-only alignments.

From the repository root or from `data/`, first rebuild clean temporary SID lists:

```bash
cd data

LC_ALL=C awk 'NF {print $1}' scop_lookup.tsv | sort -u > /tmp/scop_sids.txt
LC_ALL=C awk 'NF {print $1}' pdbs_train.txt  | sort -u > /tmp/train_sids.txt
LC_ALL=C awk 'NF {print $1}' pdbs_val.txt    | sort -u > /tmp/val_sids.txt
LC_ALL=C cat /tmp/train_sids.txt /tmp/val_sids.txt | sort -u > /tmp/assigned_sids.txt
```

Create the derived pairfiles:

```bash
mkdir -p derived

# Rows where both domains belong to either train or validation.
awk 'NR==FNR {ok[$1]=1; next} ($1 in ok) && ($2 in ok)' \
  /tmp/assigned_sids.txt tmaln-06.out > derived/tmaln-06.assigned.out

# Rows where both domains are in the training split.
awk 'NR==FNR {train[$1]=1; next} ($1 in train) && ($2 in train)' \
  /tmp/train_sids.txt tmaln-06.out > derived/tmaln-06.train.out

# Rows where both domains are in the validation split.
awk 'NR==FNR {val[$1]=1; next} ($1 in val) && ($2 in val)' \
  /tmp/val_sids.txt tmaln-06.out > derived/tmaln-06.val.out

# Rows that connect one training domain and one validation domain.
awk '
  FILENAME==ARGV[1] {train[$1]=1; next}
  FILENAME==ARGV[2] {val[$1]=1; next}
  (($1 in train) && ($2 in val)) || (($1 in val) && ($2 in train))
' /tmp/train_sids.txt /tmp/val_sids.txt tmaln-06.out > derived/tmaln-06.cross_split.out
```

Check counts:

```bash
wc -l \
  tmaln-06.out \
  derived/tmaln-06.assigned.out \
  derived/tmaln-06.train.out \
  derived/tmaln-06.val.out \
  derived/tmaln-06.cross_split.out
```

Expected baseline counts:

```text
24525 tmaln-06.out
24518 derived/tmaln-06.assigned.out
20657 derived/tmaln-06.train.out
 3814 derived/tmaln-06.val.out
   47 derived/tmaln-06.cross_split.out
```

### Reasoning

The raw pairfile contains alignments for the full split universe plus a small number of stale/unassigned rows. If every downstream script re-filters `tmaln-06.out` independently, it is easy to introduce inconsistent train/validation behavior.

Explicit pairfiles define the data contract:

| Derived file | Intended use |
|---|---|
| `tmaln-06.train.out` | VQ-VAE training-pair generation and training substitution-matrix construction |
| `tmaln-06.val.out` | validation partner loss, validation aligned-state MI, and validation diagnostics |
| `tmaln-06.cross_split.out` | diagnostic only; ignore for the simple baseline |
| `tmaln-06.assigned.out` | optional all-assigned summary/debug file |

This prevents accidental training on validation domains and prevents stale SIDs from appearing during example generation.

---

## Recommendation 3: Ignore cross-split alignments in the simple baseline

### Action

Do not use `tmaln-06.cross_split.out` for model training, model selection, substitution-matrix construction, or validation metrics in the first baseline.

Document the policy explicitly:

```text
Cross-split alignment rows are excluded from baseline training and validation.
Training examples come only from train/train rows.
Validation examples come only from val/val rows.
```

### Reasoning

The 47 cross-split alignments are a small fraction of the pairfile. They are not worth the ambiguity they introduce.

Using train/val alignments during training would leak validation-domain geometry into the training objective. Using them during validation would make validation depend partly on training-domain counterparts. The simplest policy is to ignore them until there is a deliberate reason to use cross-split alignments.

Later, cross-split rows could be useful for a specific generalization diagnostic, but that should be a separate experiment with a separate name.

---

## Recommendation 4: Remove unassigned or stale pairfile rows before example generation

### Action

Use `derived/tmaln-06.assigned.out`, `derived/tmaln-06.train.out`, or `derived/tmaln-06.val.out` instead of raw `tmaln-06.out` for any residue-pair expansion.

Do not try to repair the two pairfile SIDs that were missing from `scop_lookup.tsv` and train/val:

```text
d1dy9.1
d1o7d.1
```

They account for only 7 alignment rows after filtering by assigned SIDs.

### Reasoning

These rows are not needed for the baseline because the missing SIDs are not in the train or validation splits. Repairing them would require resolving historical SCOPe/ASTRAL identifier conventions and could introduce silent mapping assumptions.

Filtering them out is cleaner than renaming or remapping them.

---

## Recommendation 5: Add a dataset metadata check script

### Action

Add a script such as:

```text
scripts/check_scop_baseline_data.sh
```

Suggested implementation:

```bash
#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data}"
cd "$DATA_DIR"

LC_ALL=C awk 'NF {print $1}' scop_lookup.tsv | sort -u > /tmp/scop_sids.txt
LC_ALL=C awk 'NF {print $1}' pdbs_train.txt  | sort -u > /tmp/train_sids.txt
LC_ALL=C awk 'NF {print $1}' pdbs_val.txt    | sort -u > /tmp/val_sids.txt
LC_ALL=C cat /tmp/train_sids.txt /tmp/val_sids.txt | sort -u > /tmp/assigned_sids.txt
LC_ALL=C awk '{print $1; print $2}' tmaln-06.out | awk 'NF {print $1}' | sort -u > /tmp/pair_sids.txt

echo "Counts:"
wc -l /tmp/scop_sids.txt /tmp/train_sids.txt /tmp/val_sids.txt /tmp/assigned_sids.txt /tmp/pair_sids.txt

echo
echo "Train/val overlap:"
comm -12 /tmp/train_sids.txt /tmp/val_sids.txt | tee /tmp/train_val_overlap.txt | wc -l

echo
echo "Train SIDs missing from lookup:"
comm -23 /tmp/train_sids.txt /tmp/scop_sids.txt | tee /tmp/train_missing_lookup.txt | wc -l

echo
echo "Val SIDs missing from lookup:"
comm -23 /tmp/val_sids.txt /tmp/scop_sids.txt | tee /tmp/val_missing_lookup.txt | wc -l

echo
echo "Lookup SIDs not assigned to train or val:"
comm -23 /tmp/scop_sids.txt /tmp/assigned_sids.txt | tee /tmp/unassigned_lookup_sids.txt | wc -l

echo
echo "Pairfile SIDs missing from lookup:"
comm -23 /tmp/pair_sids.txt /tmp/scop_sids.txt | tee /tmp/pair_missing_lookup.txt | wc -l

echo
echo "Pairfile partition counts:"
printf "train/train\t"
awk 'NR==FNR {train[$1]=1; next} ($1 in train) && ($2 in train)' \
  /tmp/train_sids.txt tmaln-06.out | wc -l

printf "val/val\t"
awk 'NR==FNR {val[$1]=1; next} ($1 in val) && ($2 in val)' \
  /tmp/val_sids.txt tmaln-06.out | wc -l

printf "train/val\t"
awk '
  FILENAME==ARGV[1] {train[$1]=1; next}
  FILENAME==ARGV[2] {val[$1]=1; next}
  (($1 in train) && ($2 in val)) || (($1 in val) && ($2 in train))
' /tmp/train_sids.txt /tmp/val_sids.txt tmaln-06.out | wc -l
```

Run it before any training-data generation:

```bash
chmod +x scripts/check_scop_baseline_data.sh
./scripts/check_scop_baseline_data.sh data
```

### Reasoning

A small shell check catches several common failure modes:

- train/validation overlap;
- split SIDs absent from `scop_lookup.tsv`;
- pairfile rows referring to missing SIDs;
- accidental use of cross-split alignments;
- unexpected changes to the raw files.

This is cheap insurance. Dataset errors at this stage propagate into model training, substitution matrices, and benchmarks.

---

## Recommendation 6: Add a pairfile partitioning script

### Action

Add a script such as:

```text
scripts/create_scop_pairfile_partitions.sh
```

Suggested implementation:

```bash
#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data}"
OUT_DIR="${2:-$DATA_DIR/derived}"

mkdir -p "$OUT_DIR"
cd "$DATA_DIR"

LC_ALL=C awk 'NF {print $1}' pdbs_train.txt | sort -u > /tmp/train_sids.txt
LC_ALL=C awk 'NF {print $1}' pdbs_val.txt   | sort -u > /tmp/val_sids.txt
LC_ALL=C cat /tmp/train_sids.txt /tmp/val_sids.txt | sort -u > /tmp/assigned_sids.txt

awk 'NR==FNR {ok[$1]=1; next} ($1 in ok) && ($2 in ok)' \
  /tmp/assigned_sids.txt tmaln-06.out > "$OUT_DIR/tmaln-06.assigned.out"

awk 'NR==FNR {train[$1]=1; next} ($1 in train) && ($2 in train)' \
  /tmp/train_sids.txt tmaln-06.out > "$OUT_DIR/tmaln-06.train.out"

awk 'NR==FNR {val[$1]=1; next} ($1 in val) && ($2 in val)' \
  /tmp/val_sids.txt tmaln-06.out > "$OUT_DIR/tmaln-06.val.out"

awk '
  FILENAME==ARGV[1] {train[$1]=1; next}
  FILENAME==ARGV[2] {val[$1]=1; next}
  (($1 in train) && ($2 in val)) || (($1 in val) && ($2 in train))
' /tmp/train_sids.txt /tmp/val_sids.txt tmaln-06.out > "$OUT_DIR/tmaln-06.cross_split.out"

wc -l \
  tmaln-06.out \
  "$OUT_DIR/tmaln-06.assigned.out" \
  "$OUT_DIR/tmaln-06.train.out" \
  "$OUT_DIR/tmaln-06.val.out" \
  "$OUT_DIR/tmaln-06.cross_split.out"
```

Run:

```bash
chmod +x scripts/create_scop_pairfile_partitions.sh
./scripts/create_scop_pairfile_partitions.sh data data/derived
```

### Reasoning

A script makes the derived files reproducible. It also prevents manually created pairfiles from drifting across machines, branches, or experiments.

The output of this script should be deterministic for a given set of raw files.

---

## Recommendation 7: Apply C-alpha distance filtering during CIGAR expansion

### Action

When expanding CIGAR alignments into residue-pair training examples, add a residue-level geometric filter:

```text
keep aligned residue pair only if C-alpha distance <= 5.0 angstroms
```

Implementation-level logic:

```text
for each pairfile row sid1, sid2, cigar:
    parse CIGAR into aligned residue index pairs
    load CA coordinates for sid1 and sid2
    load descriptor validity masks for sid1 and sid2

    for each aligned residue pair idx1, idx2:
        require descriptor_valid_1[idx1]
        require descriptor_valid_2[idx2]
        compute ca_distance = ||CA1[idx1] - CA2[idx2]||_2
        require ca_distance <= 5.0
        emit descriptor pair and metadata
```

The filter should be applied separately for training and validation example generation:

```text
derived/tmaln-06.train.out -> training examples
derived/tmaln-06.val.out   -> validation examples
```

### Reasoning

Pair-level TM-score filtering says the two structures align globally or semi-globally. It does not guarantee that every aligned residue pair is locally trustworthy.

A good global structural alignment can still include locally poor residue correspondences. Those noisy local pairs are harmful for the VQ-VAE objective, because the model is trained to predict the aligned partner descriptor.

The Foldseek paper describes excluding aligned residue pairs whose C-alpha atoms are more than 5 angstroms apart. This is therefore not an optional cleanup; it is part of making the training target match the intended 3Di learning setup.

---

## Recommendation 8: Save residue-pair metadata

### Action

For every emitted descriptor pair, save at least the following metadata:

```text
sid1
sid2
idx1
idx2
ca_distance
pairfile_row
split
```

Recommended outputs:

```text
artifacts/training_data/
  train_pairs.npy
  train_pair_metadata.tsv
  val_pairs.npy
  val_pair_metadata.tsv
  training_data_report.json
```

Example metadata rows:

```text
split  pairfile_row  sid1     sid2     idx1  idx2  ca_distance
train  128           d1abc__  d2xyz__  42    39    1.84
train  128           d1abc__  d2xyz__  43    40    2.11
val    52            d3foo__  d4bar__  7     9     3.02
```

### Reasoning

The final training array alone is not auditable. If a feature distribution looks strange, a loss spike occurs, or a learned state collapses, you need to trace examples back to their source domains and residue pairs.

Metadata also makes it possible to compute later diagnostics:

- per-domain-pair contribution counts;
- per-fold contribution counts;
- C-alpha distance distributions;
- sequence-separation distributions;
- high-loss residue-pair examples;
- state usage by fold or structure class.

---

## Recommendation 9: Write a training-data generation report

### Action

Every run that generates descriptor-pair arrays should write a JSON report.

Suggested report fields:

```json
{
  "input_pairfile": "data/derived/tmaln-06.train.out",
  "split": "train",
  "n_pairfile_rows": 20657,
  "n_rows_skipped_missing_structure": 0,
  "n_rows_skipped_parse_error": 0,
  "n_aligned_residue_pairs_before_filters": null,
  "n_pairs_removed_invalid_descriptor_sid1": null,
  "n_pairs_removed_invalid_descriptor_sid2": null,
  "n_pairs_removed_ca_distance_gt_5": null,
  "n_pairs_kept": null,
  "ca_distance_threshold_angstrom": 5.0,
  "feature_names": [
    "cos_phi12",
    "cos_phi34",
    "cos_phi15",
    "cos_phi35",
    "cos_phi14",
    "cos_phi23",
    "cos_phi13",
    "ca_distance_i_j",
    "seq_distance_clipped_signed",
    "seq_distance_log_signed"
  ]
}
```

Also produce a separate validation report:

```text
training_data_report.train.json
training_data_report.val.json
```

### Reasoning

Counts are the easiest way to detect silent data bugs. The report should make it clear whether training examples were lost because of missing structures, invalid local descriptors, CIGAR parsing problems, or the C-alpha distance filter.

Without such a report, two model runs may differ because of hidden data-generation changes rather than architecture or objective changes.

---

## Recommendation 10: Fit feature normalization on training examples only

### Action

After generating training examples, fit feature-wise normalization parameters from the training split only:

```text
mean_j = mean of feature j over train examples
std_j  = standard deviation of feature j over train examples
```

Save them:

```text
artifacts/training_data/feature_scaler.npz
```

The file should contain:

```text
mean
std
feature_names
source_train_pairs_path
```

Apply the same normalization parameters to:

- training pairs;
- validation pairs;
- model inference/export;
- any downstream diagnostic using model inputs.

Do not fit or refit on validation data.

### Reasoning

The 10 input features are not on the same scale. Seven are cosines, one is a Euclidean C-alpha distance, and two encode sequence separation. Without normalization, distance-like features can dominate the encoder and decoder simply because of scale.

Fitting on training only preserves the train/validation boundary and makes validation metrics meaningful.

---

## Recommendation 11: Add small data tests

### Action

Add tests for the data-contract layer. Suggested tests:

1. `pdbs_train.txt` and `pdbs_val.txt` have no overlap.
2. All train and validation SIDs exist in `scop_lookup.tsv`.
3. `tmaln-06.train.out` contains only train/train rows.
4. `tmaln-06.val.out` contains only val/val rows.
5. `tmaln-06.cross_split.out` contains only train/val or val/train rows.
6. Raw plus derived partition counts match expected values for the frozen baseline.
7. CIGAR expansion emits no pair with missing descriptor masks.
8. C-alpha distance filtering removes all residue pairs above 5 angstroms.
9. Feature scaler is fit only from training examples.

### Reasoning

The dataset is small enough that these tests should be fast. They protect against accidental changes to the raw files, script behavior, and split logic.

Data tests are more useful here than additional model tests until the corpus contract is stable.

---

## Recommendation 12: Do not add another corpus yet

### Action

Do not update to SCOPe 2.08, CATH, AlphaFoldDB, TED, or ESM Atlas until the following baseline is working:

```text
fixed SCOPe split
train/train pairfile
val/val pairfile
C-alpha filtered residue pairs
metadata sidecars
training-data reports
feature normalization
basic VQ/VQ-like model comparison
```

### Reasoning

Adding another corpus now would introduce several new sources of variation:

- different domain definitions;
- different structural-label hierarchy;
- different redundancy profile;
- different alignment coverage;
- different quality filters;
- predicted versus experimental structure differences.

Those changes are valuable later, but they would make it harder to debug the current model and preprocessing pipeline.

The current corpus is already sufficient for comparing model variants and validating whether the learned states are stable, non-collapsed, and conserved across aligned residue pairs.

---

## Recommended immediate implementation order

Use this order to keep changes reviewable:

1. Add `scripts/check_scop_baseline_data.sh`.
2. Add `scripts/create_scop_pairfile_partitions.sh`.
3. Generate `data/derived/tmaln-06.train.out`, `data/derived/tmaln-06.val.out`, and `data/derived/tmaln-06.cross_split.out`.
4. Update training-data generation to consume `data/derived/tmaln-06.train.out` explicitly.
5. Add validation-data generation from `data/derived/tmaln-06.val.out`.
6. Add C-alpha distance filtering during CIGAR expansion.
7. Save pair metadata sidecars.
8. Write train/validation data-generation reports.
9. Fit and save the training-only feature scaler.
10. Add tests for split integrity, pairfile partitioning, C-alpha filtering, and scaler provenance.

This is the minimal useful data contract before model modernization.
