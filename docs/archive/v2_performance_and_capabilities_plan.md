# v2 Performance & Capability Plan

Improvements to `src/tdi/v2` identified during code review that are **optimizations and
capabilities**, not bug fixes. Everything here is additive — no item depends on another for
correctness — so the ordering below is by leverage, not requirement.

**Conventions / guardrails**

- Python 3.13; no `from __future__ import annotations`; avoid `Any` / `cast` / `type: ignore`;
  keep existing comments.
- For the performance items the non-negotiable gate is **numerical parity** with the current
  implementation on a committed test structure — add the equivalence test in the same commit so
  a speedup can never silently change features.
- Verify: `uv run ruff check src/tdi/v2`, `uv run pyright src/tdi/v2`, `uv run pytest`.
- Commit with `git commit --no-verify` (the pre-commit hook isn't installed). If a stale
  `.git/index.lock` appears (mounted filesystem disallows unlink), point the index off-mount:
  `export GIT_INDEX_FILE=/tmp/v2_index && git read-tree HEAD` before staging.
- Do **not** touch `src/tdi/v1`.

---

## Phase 1 — Feature-extraction performance (highest leverage)

This is where the real CPU time goes, so do it first.

### 1.1 Vectorize `features.calc_angles_forloop`

**Why:** it is a Python loop over every residue, each iteration calling `calc_angles`, which
itself runs ~a dozen tiny numpy ops (`unit_vec`, dot products, `norm`). Across thousands of
structures this dominates preprocessing wall-clock, and it is embarrassingly vectorizable
because every quantity is a per-residue array operation.

**Approach:**
- Precompute CA step vectors for all `i` at once: `u_fwd[i] = normalize(ca[i+1] - ca[i])`,
  `u_bwd[i] = normalize(ca[i] - ca[i-1])`.
- Gather the partner index `j = partner_idx`; build `u_5 = normalize(ca[j] - ca[i])`.
- Compute the seven cosines as row-wise dot products (`(A * B).sum(-1)`), the distance as
  `norm(ca[i] - ca[j])`, and the clipped `seq_dist`; stack to `(N, 9)`.
- Keep the scalar `calc_angles` as the reference implementation (used by tests).

**Preserve exactly:** the current masking semantics — rows are filled only where
`valid_mask[i-1..i+1]` and `valid_mask[j-1..j+1]` hold; reproduce that elementwise and leave the
rest `NaN` so downstream filtering is unchanged.

**Acceptance:** output equals the current loop within fp tolerance (`np.allclose`) on an existing
test PDB; the new-valid-mask is identical; log a measured speedup.

### 1.2 Drop the `(N,N,3)` materialization and the needless `sqrt`

Files: `features.distance_matrix`, `features.find_nearest_residues`.

**Why:** `distance_matrix` broadcasts to an `(N, N, 3)` array and square-roots it, but
`find_nearest_residues` only uses the result for an `argmin` plus one threshold compare —
`argmin` is invariant to the monotonic `sqrt`.

**Approach:** compute squared distances with `scipy.spatial.distance.cdist(..., "sqeuclidean")`
(or an einsum form) on the CB slice; do masking and `argmin` on squared distances; compare
against `fall_back_dist ** 2`.

**Acceptance:** partner indices and threshold behavior are identical on a test structure; lower
peak memory on a large domain. (Overlaps the leftover-fixes handoff item 3 — implement once,
here.)

---

## Phase 2 — Throughput

### 2.1 Batch / device-aware encoding in `encode.process_pdb` and the `evaluate` CLI

**Why:** `evaluate` calls `process_pdb` once per structure in a Python loop, and
`predict`/`discretize` build CPU tensors regardless of where the model lives, so a large
pairfile encodes serially on CPU even when a GPU is present.

**Approach:** move the input tensor to `model.device` (or accept a `device` arg); keep each
structure's residues batched (already the case); optionally add a higher-level batched path that
concatenates residues across several structures with an offset index to amortize Python
overhead.

**Acceptance:** produced sequences are byte-identical to the current path; measured speedup on
GPU and on a multi-structure pairfile. **Risk:** keep CPU the default so the CLI still runs
without a GPU.

### 2.2 Opt-in `torch.compile` for encoder/decoder

**Why:** the bf16 path already exists but the modules run eagerly; `torch.compile` is a low-risk
throughput win on the hot training/inference path.

**Approach:** add a config flag (default off); compile `self.encoder`/`self.decoder` after
construction; fall back gracefully if compilation fails.

**Acceptance:** a short training run shows no NaNs and parity in val metrics vs uncompiled;
document the first-step compile latency. **Risk:** complicates debugging and export — hence
default-off and flagged.

### 2.3 (Optional, low value) Trim the triple encoder pass in `validation_step`

**Why:** validation runs the encoder on `x`, `y`, and `x_noisy` per batch. The `x_noisy` pass is
needed for the stability metric and `self(y)` for aligned MI, so the savings are limited.
**Recommendation:** skip unless validation cost shows up in profiling.

---

## Phase 3 — Capabilities

### 3.1 Export the decoder (or the full module)

**Why:** `export_model` saves only the encoder state dict and centroids, so reconstruction-based
diagnostics and round-trip debugging are impossible from an exported model — you can encode but
never inspect what a state decodes back to.

**Approach:** also save `decoder_state_dict.pt` and load it in `load_from_export`; or rely on the
Lightning checkpoint for full-state restore and document which artifact is canonical for which
use.

**Acceptance:** a loaded export can run a forward pass and reconstruct `mu_partner`/`mu_self`.
**Risk:** slightly larger export; gate behind a flag if size matters.

### 3.2 Resolve the two training entrypoints

**Why:** `AlignmentBatchSampler` and the epoch-seeding/jitter machinery are wired only into
`tdi/v2/train.py`, while `scripts/train_v2.py` still uses a plain shuffled loader — two divergent
training paths invite drift and confusion about which is canonical.

**Approach (preferred):** make `scripts/train_v2.py` a thin shim that calls `tdi.v2.train.main`,
or delete it and document the config-driven entrypoint.

**Acceptance:** one training path; the README/usage points at it. **Risk:** if external scripts
call the old one, keep the shim rather than deleting.

### 3.3 Finish or remove `encode.process_pdb`'s `exclude_feat`

**Why:** it slices out a feature column, silently changing dimensionality, and would break a
`TdiV2Model` encoder's fixed `input_dim` if ever exercised — neither a working ablation nor
inert.

**Recommendation:** remove it (confirm no caller passes it). If feature-ablation is actually
wanted, build it properly — adjust `input_dim` at construction and assert consistency at encode
time — rather than leaving the half-version.

**Acceptance:** either the param is gone and callers updated, or it's a validated, tested
ablation switch.

### 3.4 Enrich the `evaluate` report with diagnostics the model already computes

**Why:** the CLI emits MI, transition-adjusted MI, and the substitution matrix, but the model's
`validation_step`/epoch-end already compute per-state usage, dead-state fraction, stability, and
margin — surfacing them at evaluation time turns the CLI into a real diagnostic at near-zero
cost.

**Approach:** over the encoded sequences, compute the state-usage histogram, dead-state fraction,
and normalized entropy; add them to `evaluation_report.json`.

**Acceptance:** the report includes usage/dead-state fields and totals reconcile with the encoded
sequence counts.

---

## Sequencing and what to skip

- Do **Phase 1 first** — 1.1 is the biggest single win; 1.2 is cheap and overlaps an
  already-queued fix.
- Phase 2 and Phase 3 are independent of each other and of Phase 1; pick by need (2.1 if
  `evaluate` is slow on real pairfiles, 3.2 if the dual entrypoints cause confusion).
- **Safe to skip** with no downside: 2.3 (marginal) and 2.2 if the `torch.compile` debugging
  overhead isn't wanted.
- For both perf items, gate the merge on numerical parity against the current implementation on a
  committed test structure.
