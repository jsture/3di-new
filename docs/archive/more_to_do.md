Trimmed status after inspecting v2.tdi.repomix.xml

Many core model upgrades are already implemented. The remaining list should be much shorter and mostly focused on data provenance, leakage control, preprocessing reports, and a few model-evaluation metrics.

Keep on the todo list — genuine remaining improvements

These are the remaining items that still matter.

1. Add group-level split manifests

Status: Not implemented.

Add:

structure_id,domain_id,group_id,split

Use the strongest available group_id: fold, superfamily, sequence cluster, structure cluster, or source domain ID.

Reasoning: This is the main remaining leakage-control issue. Residue-pair-level splitting is invalid for this task.

Priority: High / genuine improvement

⸻

2. Build an expanded residue-pair provenance table

Status: Not implemented.

Add a row-level metadata table:

row_id
structure_id_1
structure_id_2
domain_id_1
domain_id_2
residue_index_1
residue_index_2
alignment_id
alignment_score
split
group_id_1
group_id_2
valid_descriptor_1
valid_descriptor_2
distance_filter_status
ca_pair_distance, if available

Store arrays separately:

pairs.parquet
x.npy
y.npy

Reasoning: The current pair arrays are not auditable enough. You need row-level provenance to debug bad alignments, leakage, group imbalance, and state sources.

Priority: High / genuine improvement

⸻

3. Implement superposition-aware Cα distance filtering

Status: Not implemented, but safely blocked.

Current behavior is good in the sense that it refuses to apply the filter without superposed coordinates. The next step is to implement it correctly.

Required logic:

if aligned/superposed coordinates are available:
    compute ca_pair_distance
    keep pairs with distance <= 5 Å
else:
    set distance_filter_status = "unavailable"
    do not filter

Reasoning: This filter is useful only after structural superposition. Raw coordinate-frame filtering is wrong.

Priority: High / genuine improvement

⸻

4. Add explicit finite-feature validation

Status: Not clearly implemented.

Add checks after pair generation and after standardization:

def assert_finite_features(x: np.ndarray, name: str) -> None:
    if not np.isfinite(x).all():
        bad = np.size(x) - np.isfinite(x).sum()
        raise ValueError(f"{name} contains {bad} non-finite values")

Run on:

raw x
raw y
scaled x
scaled y

Reasoning: Invalid descriptors should fail preprocessing, not appear as unstable training behavior.

Priority: High / genuine improvement

⸻

5. Enforce train-only scaling in the pipeline

Status: Primitives implemented; pipeline enforcement not shown.

PairDataset can fit scaling internally if mean/std are omitted. That is convenient, but dangerous for validation/test if used incorrectly.

Add explicit training flow:

mean, std = fit_standardizer(x_train)
train_ds = PairDataset(x_train, y_train, mean=mean, std=std, jitter_std=...)
val_ds = PairDataset(x_val, y_val, mean=mean, std=std, jitter_std=0.0)
test_ds = PairDataset(x_test, y_test, mean=mean, std=std, jitter_std=0.0)

For validation/test, disallow mean=None.

Reasoning: The function exists, but the pipeline should prevent accidental leakage.

Priority: High / genuine improvement

⸻

6. Add group/alignment weighting or capping

Status: Not implemented.

Add either:

max_pairs_per_alignment = 512 or 1024

or sample weights:

1 / examples_per_alignment
1 / examples_per_group
1 / cluster_size

Use WeightedRandomSampler only after row-level provenance exists.

Reasoning: Long structures and overrepresented groups can dominate the learned alphabet.

Priority: Medium-high / genuine improvement

⸻

7. Add raw and balanced validation views

Status: Not implemented.

Create two validation loaders:

val_raw
val_balanced

Use val_raw for final reporting and val_balanced for diagnostics.

Reasoning: Raw validation measures real held-out distribution. Balanced validation detects whether strong performance is driven by overrepresented groups.

Priority: Medium-high / genuine improvement

⸻

8. Add state-margin metric

Status: Not implemented.

For VQ:

d_sorted = distances.sort(dim=1).values
margin = d_sorted[:, 1] - d_sorted[:, 0]

Report:

val_margin_mean
val_margin_p05
val_fraction_low_margin

For FSQ, report distance to nearest quantization boundary instead.

Reasoning: Stability is already implemented through perturbation. Margin gives a complementary geometric diagnostic: how close points are to decision boundaries.

Priority: Medium / genuine improvement

⸻

9. Add transition-adjusted MI to validation

Status: Partially implemented elsewhere, not in model validation.

submat.py has transition-adjusted MI machinery, but TdiV2Model.on_validation_epoch_end() only computes aligned-state MI.

Add validation metrics:

val_aligned_mi
val_transition_mi
val_transition_adjusted_mi

Reasoning: A useful 3Di-like alphabet should avoid excessive local sequential dependence. Aligned MI alone is not enough.

Priority: Medium / genuine improvement

⸻

Keep as engineering work, not model-improvement work

These are still useful but should not be presented as core model improvements.

10. Preprocessing reports

Status: Not implemented.

Add:

training_data_report.json
validation_data_report.json
training_data_report.md
validation_data_report.md

Include:

counts before/after filters
NaN/Inf counts
feature statistics
contact-bin histograms
examples per group
examples per alignment
state usage after encoding

Priority: Useful engineering

⸻

11. Resolved run config

Status: Not implemented.

Add:

run_config.resolved.json
preprocessing_config.json
model_config.json
train_config.json

Priority: Useful engineering

⸻

12. CLI entry points

Status: Not implemented.

Add commands like:

tdi-v2 make-splits
tdi-v2 build-pairs
tdi-v2 fit-scaler
tdi-v2 train
tdi-v2 encode
tdi-v2 build-submat
tdi-v2 evaluate

Priority: Useful engineering

⸻

13. Schema validation

Status: Not implemented.

Validate pair tables before training.

Priority: Useful engineering

⸻

14. CI and smoke tests

Status: Not shown in repomix.

Keep:

ruff check .
pyright src/tdi
pytest
small train/export/load smoke test

Priority: Useful engineering

⸻

Keep as experiments only

Do not prioritize these yet.

Item	Status	Recommendation
State-transition auxiliary head	Not implemented	Experiment after current baseline is benchmarked.
K-means codebook initialization	Not implemented	Experiment; less urgent because EMA + dead-code replacement exists.
Context-window encoder	Not implemented	Experiment; may increase local dependency between letters.
Rotation-trick VQ	Not implemented	Experiment; not needed for a stable baseline.
Product quantization	Not implemented	Probably unnecessary for 20–32 states.
torch.compile	Not implemented	Nice-to-have for inference speed only.
Mixed precision	Not implemented	Nice-to-have after numerical stability is confirmed.
Large lazy datasets / Zarr / Arrow	Not implemented	Only needed once data no longer fits in RAM.

⸻

Final trimmed list

Genuine remaining improvements

1. Group-level split manifest.
2. Expanded residue-pair provenance table.
3. Correct superposition-aware Cα distance filtering.
4. Explicit finite-feature validation.
5. Enforced train-only scaler workflow.
6. Group/alignment weighting or per-alignment capping.
7. Raw and balanced validation views.
8. State-margin metric.
9. Transition-adjusted MI in validation.

Useful engineering work

1. Preprocessing reports.
2. Resolved run configs.
3. CLI entry points.
4. Pair-table schema validation.
5. CI and train/export/load smoke tests.

Experiments only

1. State-transition auxiliary head.
2. K-means centroid initialization.
3. Context-window encoder.
4. Rotation-trick VQ.
5. Product quantization.
6. Mixed precision.
7. torch.compile.
8. Lazy large-corpus storage.

Main correction to the previous list

The model side is now mostly modernized. The remaining important work is not another model rewrite. It is mainly:

leakage-proof data construction
auditable pair provenance
correct structural filtering
strong validation metrics
controlled sampling

That is where the next real gains in trustworthiness will come from.
