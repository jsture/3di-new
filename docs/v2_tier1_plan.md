# v2 Model — Tier 1 Implementation Plan (Claude Code)

Scope: the "always do" / low-risk upgrades only. Everything conditional, ablation, or
experimental (rotation trick, LFQ/soft-assignment entropy, FSQ retuning, Sinkhorn,
EMA/Polyak weights, residual/product VQ) is **out of scope** for this plan.

Targets: `src/tdi/v2/model.py` and `scripts/train_v2.py`.
Backends: changes that touch the codebook apply to the **EMA-VQ** backend only and must be
no-ops for FSQ.

Selection metric for any before/after check: validation **aligned-state MI** and
substitution-matrix quality / retrieval — not reconstruction loss alone.

---

## Task 1 — k-means codebook initialization (EMA-VQ only)

**File:** `src/tdi/v2/model.py`

**Why:** `EMAVectorQuantizer.__init__` seeds the codebook with `torch.randn(n_states, z_dim)`.
For a 20-state alphabet, random init wastes early steps and over-relies on dead-code
replacement. Seeding from k-means of real encoder outputs makes early usage non-arbitrary.

**Changes:**
1. In `EMAVectorQuantizer.__init__`, add `self.register_buffer("initialized", torch.tensor(False))`.
2. Add `EMAVectorQuantizer.init_codebook(self, z: torch.Tensor)`:
   - If `self.l2_normalize`, run k-means on `F.normalize(z, dim=-1)` (match the lookup space);
     otherwise on raw `z`.
   - `n_clusters = self.n_states`. Use a fixed seed. (sklearn `KMeans` is fine; if you want
     zero new deps, a short torch k-means / `torch.pca_lowrank`-free Lloyd loop is acceptable.)
   - Set `self.embedding.copy_(centers)`, `self.ema_sum.copy_(centers)`,
     `self.ema_count.fill_(1.0)`, `self.initialized.fill_(True)`.
3. Add `TdiV2Model.init_codebook_from_data(self, loader, n_batches=8)`:
   - No-op unless `isinstance(self.quantizer, EMAVectorQuantizer)`.
   - `self.eval()`; with `torch.no_grad()`, run the **encoder** over the first `n_batches`
     batches (scaled `x`), concatenate `z`, call `self.quantizer.init_codebook(z)`. Use
     **more than one batch** — a single batch is biased. Restore `self.train()`.
4. **Wire in** `scripts/train_v2.py`: after the model is constructed (~line 95) and before
   `trainer.fit(...)` (~line 117), call `model.init_codebook_from_data(train_loader)`.

**Accept when:** for VQ runs, codebook is non-random at step 0; early `val_perplexity` is
higher and `val_dead_states` lower than a random-init run. FSQ runs are unaffected.

---

## Task 2 — LR schedule tied to real training length + warmup

**File:** `src/tdi/v2/model.py`, method `TdiV2Model.configure_optimizers`

**Why:** current `CosineAnnealingLR(T_max=100)` is hardcoded and decoupled from the actual
run length, so the cosine floor lands in the wrong place unless you happen to train exactly
100 units.

**Changes:**
1. Total steps: `total_steps = int(self.trainer.estimated_stepping_batches)`.
2. `warmup_steps = max(1, int(warmup_ratio * total_steps))` — add `warmup_ratio: float = 0.03`
   as an `__init__` hyperparameter (saved via `save_hyperparameters`).
3. Build warmup→cosine with
   `SequentialLR(opt, [LinearLR(opt, start_factor=1e-2, total_iters=warmup_steps),
   CosineAnnealingLR(opt, T_max=total_steps - warmup_steps)], milestones=[warmup_steps])`.
4. Return scheduler with `{"scheduler": sched, "interval": "step"}` so it steps per optimizer
   step (not per epoch).

**Accept when:** logged LR warms up then follows a full cosine to ~0 by the final step, for any
`--max_epochs` / dataset size.

---

## Task 3 — AdamW parameter groups: no weight decay on bias/norm

**File:** `src/tdi/v2/model.py`, method `TdiV2Model.configure_optimizers`

**Why:** `AdamW(self.parameters(), weight_decay=...)` currently decays LayerNorm weights and
biases. Standard hygiene: decay only ≥2-D matmul weights.

**Changes:**
1. Split parameters: `decay = [p for n, p in self.named_parameters() if p.requires_grad and
   p.ndim >= 2]`, `no_decay = [... p.ndim < 2 ...]` (catches biases + LayerNorm γ/β).
2. `AdamW([{ "params": decay, "weight_decay": self.weight_decay },
   { "params": no_decay, "weight_decay": 0.0 }], lr=self.lr)`.

**Accept when:** optimizer has two param groups; the no-decay group contains exactly the bias
and LayerNorm parameters and has `weight_decay == 0.0`.

---

## Task 4 — Gradient clipping  ✅ ALREADY DONE — verify only

**File:** `scripts/train_v2.py` (lines 113–114)

`Trainer(... gradient_clip_val=1.0, gradient_clip_algorithm="norm")` is already present.
**No code change.** Just confirm it stays set. Do not add a second clipping path.

---

## Task 5 — Modernize the contrastive head (symmetric + learnable clamped temperature)

**File:** `src/tdi/v2/model.py`, `TdiV2Model.__init__` and `TdiV2Model.training_step`

**Why:** the contrastive auxiliary (enabled by default, `lambda_contrast=0.05`) uses a fixed
`temperature=0.1` and a single-direction cross-entropy. CLIP-style hygiene: learnable clamped
temperature and a symmetric loss. Keep it strictly inside the existing `lambda_contrast > 0`
guard so it remains auxiliary.

**Changes:**
1. In `__init__`, inside the `if lambda_contrast > 0.0:` block, add a learnable log-scale:
   `self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1.0 / temperature))))`.
2. In `training_step`, replace the contrastive block:
   - `scale = self.logit_scale.clamp(max=np.log(100.0)).exp()`
   - `logits = scale * (zq_proj @ h_proj.t())`  (drop the `/ self.temperature` divide)
   - `labels = torch.arange(logits.shape[0], device=logits.device)`
   - `loss_contrast = self.lambda_contrast * 0.5 *
     (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))`

**Accept when:** with `lambda_contrast=0` the model is unchanged; with it enabled, training runs
and `logit_scale` updates and stays ≤ log(100). Contrastive term must not dominate
`loss_partner` (spot-check the logged components).

---

## Task 6 — bf16 mixed precision with fp32-safe quantizer internals

**Files:** `scripts/train_v2.py` (Trainer) and `src/tdi/v2/model.py` (`EMAVectorQuantizer.forward`)

**Why:** bf16 is a free speed/stability win on supported hardware, but the squared-distance
lookup, EMA count/sum updates, and any variance/NLL term must stay fp32.

**Changes:**
1. `scripts/train_v2.py`: add CLI arg `--precision` (default `"bf16-mixed"`) and pass
   `precision=args.precision` into `L.Trainer(...)`. Keeping it a flag lets non-bf16 hardware
   fall back to `"32-true"`.
2. `src/tdi/v2/model.py`, `EMAVectorQuantizer.forward`: wrap the distance computation, the
   `argmin`/one-hot lookup, and the EMA update block in
   `with torch.autocast(device_type=z.device.type, enabled=False):` and cast the working tensor
   to fp32 (`z32 = z.float()`) for those ops. Cast `z_q` back to `z.dtype` before the
   straight-through line.
3. If `loss_type == "gaussian_nll"`, compute the NLL term in fp32 (cast `mu`/`var`/target).

**Accept when:** a short `--precision bf16-mixed` run trains without NaNs and produces VQ
metrics (perplexity, dead-states, MI) consistent with a `32-true` run; codebook math verified
fp32.

---

## Final verification

1. `make` lint/type targets pass (ruff, mypy, pyright) and `pytest` is green; add/adjust unit
   tests for: k-means init populates a non-random codebook and is a no-op for FSQ; optimizer
   has two param groups with correct decay; scheduler warms up then cosines over
   `estimated_stepping_batches`; symmetric contrastive loss with `lambda_contrast=0` is a no-op.
2. Smoke train (a few epochs) under both `--precision 32-true` and `--precision bf16-mixed`;
   confirm no NaNs and comparable `val_aligned_mi` / `val_dead_states`.
3. Judge any before/after by validation **aligned-state MI** and substitution-matrix /
   retrieval quality — not reconstruction loss alone.

## Implementation order
1 (k-means) → 2 (scheduler) → 3 (param groups) → 5 (contrastive) → 6 (bf16). Task 4 is verify-only.
