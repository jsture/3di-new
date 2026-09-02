# v2 Simplification Plan — single, auditable structural-alphabet learner

Goal: one reliable path from aligned descriptors to a trained structural alphabet. Keep the
quantizer performance machinery and the data-robustness/audit trail; cut the framework drift,
auxiliary objectives, training scaffolding, and sequence-level scoring from the training path.

Locked decisions:

- EMA-VQ and FSQ are **both first-class quantizers behind one shared interface**, selected by a
  `--quantizer {vq,fsq}` flag. A `train` run trains **exactly one** quantizer; **running both is
  never enforced**. EMA-VQ is the reference learner (and the likely shipped alphabet); FSQ `[5,4]`
  is the low-complexity comparator. (The quantizer is the alphabet-forming mechanism, not an
  auxiliary knob — so a lone FSQ result at ~20 states would be ambiguous; keeping both available
  guards the scientific conclusion at near-zero structural cost.)
- The VQ-vs-FSQ comparison is a deliverable, but it is produced by a **separate, optional driver**
  that invokes the normal `train`/`evaluate` path twice and tabulates — it is **not** baked into
  the core training loop or model, and the core never imports it.
- Dead-code replacement is a **mandatory** part of the VQ path (collapse prevention, not a flag).
- Keep simple one-shot k-means init on the VQ path (config flag, default on for VQ); the
  random-init "floor" is available by toggling the flag. Keep a cheap VQ margin as a validation
  diagnostic.
- **Single gradient path: straight-through estimator only. The rotation trick is removed from
  core** (no `gradient_mode`); it lives in git history / `experiments/`.
- Mutate `tdi.v2` in place; delete `tdi.v1` entirely (but only after the new path passes tests).
- No hardcoded 20 — `n_states` configurable, capped by the 50-letter alphabet; the alphabet is
  recorded in the export so encode/eval don't hardcode it.
- Excise PyTorch Lightning; drop bf16/AMP (run fp32 throughout).
- **Behavior-lock golden tests land before any deletion or refactor**, so intentional
  simplification is distinguishable from silent drift.
- Default to a fixed learning rate (scheduler optional), keeping grad-clip + early-stop.
- Update the README with a pipeline explainer.

Conventions: Python 3.13 (no `from __future__`; avoid `Any`/`cast`/`type: ignore`); verify with
`uv run ruff check`, `uv run pyright src/tdi`, `uv run pytest`; commit with `--no-verify`.

---

## Phase 0 — Behavior-lock golden tests (must land first)

Before touching the model, Lightning, or v1, pin the data-path invariants so the refactor can be
checked for drift. Add to `tests/`:

- `parse_cigar` rejects unsupported ops and parses `P` pairs to the expected `(N, 2)`.
- `extract_features` returns the expected shape and a finite-only valid mask.
- `filter_ca_distance` uses CA columns `0:3` only, and removes a known bad pair after
  superposition (construct a small synthetic case).
- `align_features` produces bidirectional `x`/`y` rows in the expected (forward-then-reverse)
  order, and `len(metadata) == len(x) == len(y)`.
- Train-only scaler: `fit_standardizer` is fit on train and reused for val (no val leakage).
- An exported model encodes states strictly in `[0, n_states)`.
- `evaluate` produces `sequences.txt`, `submat.txt`, `evaluation_report.json`.

**Acceptance:** all golden tests pass against the *current* code; they become the regression
guard for every later phase.

---

## Phase 1 — Carve `model.py` to one path

Target `src/tdi/v2/model.py` (~230 lines, plain `nn.Module`, no Lightning):

```
MLP(input_dim, hidden_dim, out_dim, depth)            # LayerNorm + SiLU, as today
EMAVectorQuantizer(n_states, z_dim, decay, commitment_cost, l2_normalize=True, min_count)
    .init_codebook(z, seed)                            # k-means (one-shot)
    .forward(z) -> (z_q, indices, q_loss, metrics)     # metrics = {perplexity, n_replaced, margin}
FSQQuantizer(levels)
    .forward(z) -> (z_q, indices, q_loss=0, metrics)   # metrics = {perplexity}
AlphabetModel(nn.Module)
    encoder=MLP(input_dim,hidden_dim,z_dim); decoder=MLP(z_dim,hidden_dim,input_dim)
    quantizer = vq | fsq
    .forward(x) -> dict(y_hat, indices, q_loss, metrics)
    .encode_states(x) -> indices                       # fp32, no autocast guard
    .init_codebook_from_loader(loader, n_batches, seed)  # VQ only, no-op for FSQ
    .save(dir, mean, std); classmethod .load(dir)
```

**Quantizer return shape:** `(z_q, indices, q_loss, metrics)` where `metrics` is a dict. Keeps
optional diagnostics (perplexity, n_replaced, margin) out of the stable signature so adding or
dropping a metric never churns call sites.

**Both quantizers are first-class** behind a `make_quantizer(cfg) -> EMAVectorQuantizer |
FSQQuantizer` factory; `AlphabetModel` is identical except for which quantizer it holds, and
`--quantizer {vq,fsq}` selects one per run (never both). Dead-code replacement is mandatory inside
the VQ path, not a toggle. (The two quantizer classes may live in a small `quantizers.py` to keep
them out of the model file — that's cosmetic; what matters is the one shared interface + factory.)

**Keep:** EMA updates, commitment loss, L2-normalized lookup, dead-code replacement, simple
one-shot k-means init, centroids export. **Gradient: STE only**, inline in each quantizer
(`z_q = z + (z_q - z).detach()`); delete `gradient_mode` and `rotation_trick` from core.

**Remove from the model:** GaussianNLL + `var_*` heads; `mu_self` / self-recon; contrastive
projectors + `logit_scale`; usage-entropy; `quantizer_warmup_epochs` / `forward(quantize=False)`
/ `_aux_ramp`; the `on_train_epoch_start` k-means timing; all `torch.autocast` / fp32-guard
blocks (`_quantizer_distances` becomes a plain fp32 helper); `save_hyperparameters`,
`training_step` / `validation_step` / `on_validation_epoch_end` / `configure_optimizers`,
`_can_log`. Keep `replacement_warmup_steps` as an internal default, not a surfaced config knob.

**No hardcoded 20:** `n_states` is a constructor arg (default 20); for FSQ
`n_states = prod(levels)` (default `[5, 4]`). At construction and in `encode`:

```
if n_states > len(LETTERS):  # 50
    raise ValueError(f"n_states={n_states} exceeds alphabet size {len(LETTERS)}")
```

Rewrite `tests/test_v2_*.py` against this API (VQ train-one-step + save/load round-trip; FSQ
forward; `n_states` cap raises; k-means populates a non-random codebook; dead-code replacement
reduces dead states on a degenerate batch).

**Acceptance:** Phase 0 golden tests still pass; `AlphabetModel(quantizer="vq", n_states=24)` and
`quantizer="fsq", levels=[5,5]` both train one step; `save`/`load` round-trips; fp32 forward
yields finite `y_hat`/`q_loss`.

---

## Phase 2 — Plain training loop (`train.py`) + slim config

Rewrite `src/tdi/v2/train.py` as a plain loop:

```
load train_x_raw/y, val_x_raw/y, scaler.npz   (existing contract from tdi.data)
standardize with train mean/std
PairDataset -> DataLoader(shuffle=True, drop_last=True) / val shuffle=False
model = AlphabetModel(...from cfg...)
if cfg.quantizer == "vq" and cfg.kmeans_init: model.init_codebook_from_loader(train_loader)
opt = AdamW(no-decay groups for bias/LayerNorm)        # keep — cheap, correct
sched = none by default (fixed LR); optional CosineAnnealingLR if cfg.scheduler == "cosine"
for epoch in range(max_epochs):
    train: loss = smooth_l1(y_hat, y) + q_loss; clip_grad_norm_(1.0); step (sched.step() if set)
    val:   val_loss + state diagnostics (Phase 4)
    if val_loss < best: save best to out_dir; else patience++ (early stop)
model.load(best); model.save(out_dir, mean, std); write run_config.resolved.json + train_log.csv
```

Default optimizer/loop: `lr=1e-3`, `weight_decay=1e-4`, `max_epochs=20`, `patience=5`,
`clip_grad_norm=1.0`, `scheduler="none"`. Cosine is opt-in via config; **no linear warmup**.

Slim `train_config.py` to the surviving fields: `quantizer`, `n_states`, `z_dim`, `levels`,
`hidden_dim`, `loss` (`smooth_l1|mse`), `commitment_cost`, `decay`, `min_count`, `kmeans_init`
(+ `kmeans_seed`, `kmeans_init_batches`), `lr`, `weight_decay`, `batch_size`, `max_epochs`,
`scheduler` (`none|cosine`), `seed`, `patience`. **Delete** `lambda_self/usage/contrast`,
`temperature`, `gradient_mode`, `quantizer_warmup_epochs`, `aux_ramp_epochs`, `warmup_ratio`,
`descriptor_jitter_std`, `precision`, `accumulate_grad_batches`, `sampler`/`alignments_per_batch`,
`replacement_warmup_steps`. Keep `config_hash()` and write the resolved snapshot.

**Acceptance:** `python -m tdi.v2 train --config ...` trains with no Lightning import; selection by
`val_loss`; the export contract (Phase 4) is produced.

---

## Phase 3 — Slim `training_data.py` / dataset

Keep `align_features`, `filter_valid_pairs`, `filter_ca_distance`, `make_bidirectional_pairs`,
`fit_standardizer`, `transform`, `extract_features`, `encoder_features`, `FEATURE_CACHE`.

**Remove:** `jitter_coords`, the `coordinate_jitter_std` / `descriptor_jitter_std` params,
`AlignmentBatchSampler`, `set_epoch`, and the per-item RNG in `PairDataset`.
`PairDataset.__getitem__` returns `(tensor(x_scaled[i]), tensor(y_scaled[i]))` only.

Keep bidirectional pairs in the **data** (`make_bidirectional_pairs`); **no symmetric loss** in
training (avoids double counting).

**Acceptance:** Phase 0 golden tests still pass; `PairDataset` has no RNG/epoch state;
`extract_features(path, virt)` has no jitter arg; `tdi.data` still imports and runs.

---

## Phase 4 — Validation metrics + self-describing export/eval

**In-training validation only:** `val_loss` (smooth_l1), `state_counts` (`bincount(indices)`),
`dead_state_count`, `perplexity`, and `vq_margin` (VQ only — cheap, from the metrics dict).
Select on `val_loss`. Log to `train_log.csv`.

**Export contract** (`AlphabetModel.save`): `encoder_state_dict.pt`, `decoder_state_dict.pt`
(cheap; enables "what does a state decode to" diagnostics), `config.json`, `scaler.json`, and
`centroids.npy` (vq) or `fsq_levels.json` (fsq). **`config.json` is self-describing** and carries
the alphabet so encode/eval never hardcode it:

```
{ "input_dim":10, "hidden_dim":64, "z_dim":4, "n_states":20, "quantizer":"vq",
  "levels":null, "loss":"smooth_l1", "feature_convention":"seq_delta_j_minus_i",
  "letters":"ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwyz",
  "invalid_state":"X" }
```

`encode.py` reads `letters`/`invalid_state`/`n_states` from the loaded config instead of a
module constant (removes the hidden export↔encode coupling).

**Evaluation stays one script** (`evaluate.py` / `cli.py evaluate`): encode val PDBs →
`sequences.txt`; counts → `submat.txt`; MI + transition-adjusted MI + MI_tot + state frequencies
→ `evaluation_report.json`. No sequence-level metrics in the train loop.

**CLI shape:**

- `python -m tdi.data build-features --config ...`
- `python -m tdi.v2 train --config ... --quantizer {vq,fsq} --out runs/<name>` — trains exactly
  one quantizer and writes a single run directory.
- `python -m tdi.v2 evaluate --model runs/<name> --pdb-dir ... --pairfile ... --out runs/<name>/eval`

**Acceptance:** `evaluate` produces the three files using the alphabet from `config.json`;
invalid residues render as the configured `invalid_state`; a `train` run produces exactly one
model (the comparison of two is the optional driver in Phase 7).

---

## Phase 5 — Lean-but-auditable metadata + report (`tdi.data`)

Slim the reports, but **do not over-slim provenance** (this is the "robust data" the project
values).

Keep per-row `metadata.parquet`: `alignment_id`, `sid_source`, `sid_target`, `idx_source`,
`idx_target`, `split_group_source`, `split_group_target`, `fold_source`, `fold_target`,
`superfamily_source`, `superfamily_target`, `ca_dist_raw`, `ca_dist_superposed`.

- **Enrich `alignment_id`** to `"{pairfile_stem}:{source_row}:{sid1}:{sid2}"`. Because it then
  encodes the source row (and the split via the stem), the standalone `source_pairfile_row`
  column is redundant and can be dropped without losing the row→input mapping.
- **Drop:** the SHA `row_id` (deterministic row order suffices), `source_is_forward` (order is
  forward-then-reverse), `family_*`.
- Add explicit `split_group_*` columns so leakage audits don't depend on inferring the group
  from the classification.

Report: keep one `report.json` (joint train/val: stage counts, feature mean/std/min/max, nan/inf,
`examples_per_alignment` **quantile summary**, `examples_per_fold` counts) + `report.md` +
`manifest.json` + `DATACARD.md`. **Drop** `train_report.json`/`val_report.json` duplicates; move
the seq-sep and Cα histograms behind a `--full-report` flag (off by default). Keep the corrected
`reconcile`, manifest hashes, structure QC, and resolved-config.

**Open item:** read `scripts/make_splits.py` / `scripts/split_folds.py` to confirm the grouping
level (fold vs superfamily). Keep both `fold_*` and `superfamily_*` until confirmed, then drop the
unused one and set `split_group_*` to the level actually used.

**Acceptance:** default build writes the auditable-lean `metadata.parquet` and a single
`report.json`; `--full-report` restores histograms; a metadata row still maps back to its source
alignment via `alignment_id`.

---

## Phase 6 — Delete v1 + dead entrypoints (after the new path passes)

Once Phases 1–5 are green under the golden tests:

- Delete `src/tdi/v1/` (whole directory).
- Delete `scripts/train.py` (v1), `scripts/train_v2.py` (old Lightning), `scripts/encode_pdbs.py`
  (v1), `scripts/create_submat.py` (v1), `scripts/create_training_data.py` (superseded by
  `tdi.data build-features`). Keep `fetch_scop40_structures.py`, `make_splits.py`,
  `split_folds.py`.
- Delete `tests/test_training.py` (v1).
- Remove v1 references in `CLAUDE.md` / README.

**Acceptance:** `grep -rn "tdi\.v1" src scripts tests` returns nothing; all tests green.

---

## Phase 7 — Optional quantizer-comparison driver (separate code, opt-in)

The VQ-vs-FSQ comparison is a deliverable but must not touch the core path. Add a **standalone
driver** `scripts/compare_quantizers.py` (not imported by `tdi.v2`):

- runs the normal `train` + `evaluate` twice — once `--quantizer vq`, once `--quantizer fsq` —
  into `runs/ema_vq/` and `runs/fsq_5x4/`;
- reads the two `evaluation_report.json` (+ `train_log.csv`) and writes `comparison_report.json`
  (and optional `comparison.md`) with side-by-side `val_loss`, `state_entropy`, `dead_states`,
  `aligned_mi`, `mi_tot`, and a substitution-matrix summary.

Exactly two runs and one table — no sweeps, no extra quantizer variants, no k-means / levels
exploration. The core `train` / `evaluate` / `model` code has no knowledge of it; you normally
train one quantizer at a time and only reach for this driver when you want the comparison.

**Acceptance:** the driver produces two `runs/*/` dirs and one `comparison_report.json`; deleting
the driver leaves `train` / `evaluate` fully functional; `grep` confirms core modules do not
import it.

---

## Phase 8 — README + experiments quarantine

**README** — add a "v2 pipeline" section covering:

- the three stages, commands, and the artifact tree;
- the model: two quantizers selected by `--quantizer {vq,fsq}` behind one interface — EMA-VQ
  (reference: EMA codebook, commitment, dead-code reset, k-means init) and FSQ `[5,4]`
  (comparator) — one smooth_l1 partner-prediction loss, STE gradient, fp32, plain PyTorch (no
  Lightning), fixed LR by default; a `train` run uses one quantizer;
- the optional `scripts/compare_quantizers.py` driver that runs both and writes a comparison
  table (standalone, not part of the core path);
- `n_states` configurable via `--n-states` (VQ) or `--levels` (FSQ), **capped at 50** (the
  structural alphabet has 50 letters; beyond that needs a longer alphabet), default 20 = FSQ
  `[5, 4]`;
- how to read outputs (`sequences.txt`, `submat.txt`, `evaluation_report.json`); invalid residues
  = the configured `invalid_state`;
- a one-line note that removed objectives (GaussianNLL, contrastive, self-recon, warmup
  curriculum, transition head, rotation trick, augmentation) live in git history / `experiments/`.

**Quarantine** — recommended: don't maintain dead `.py` files. Add `experiments/README.md`
pointing at the pre-refactor commit/tag. (Alternative: extract runnable snapshots.)

---

## Do not stage or commit any changes as you go. I handle this.

1. `test(v2): behavior-lock golden tests for the data path` (Phase 0)
2. `refactor(v2): carve model — VQ + FSQ behind one flagged interface, one loss, STE, fp32, metrics dict` (Phase 1)
3. `refactor(v2): plain training loop (fixed-LR default) + slim config` (Phase 2)
4. `refactor(v2): slim dataset (drop jitter/sampler)` (Phase 3)
5. `refactor(v2): lean validation + self-describing export/eval` (Phase 4)
6. `refactor(data): lean-but-auditable metadata and report` (Phase 5)
7. `refactor(v2): delete v1 and dead entrypoints` (Phase 6)
8. `feat(v2): optional standalone quantizer-comparison driver` (Phase 7)
9. `docs(v2): README pipeline explainer + experiments pointer` (Phase 8)

Phase 0 first (the regression guard). Phases 1–5 each must keep the golden tests green. Phase 6
(deletion) only after the new path is proven. Phase 5 is independent of 1–4 and can land anytime
after Phase 0. Phase 7 (comparison driver) needs Phases 1–4. Phase 8 last.

## Risks / things to watch

- **Export/eval coupling:** the loader rename (`TdiV2Model.load_from_export` →
  `AlphabetModel.load`) and the alphabet-in-config change must land with `encode.py`/`evaluate.py`
  in the same commit as Phase 1/4.
- **`tdi.data` ↔ model:** `tdi.data` imports `align_features`, `fit_standardizer` from
  `training_data` — keep those names stable through Phase 3.
- **Split grouping unknown:** resolve the Phase 5 open item before dropping `fold_*` or
  `superfamily_*`.
- **FSQ margin:** only emit `vq_margin` for VQ; FSQ validation omits it.
- **k-means dependency:** keeps `scikit-learn` (already a dependency).
- **Comparison driver isolation:** `scripts/compare_quantizers.py` must stay standalone — core
  `tdi.v2` modules must never import it (keep the comparison out of the single-run path).
