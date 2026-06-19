# Training the v2 3Di Model

This document describes the minimal end-to-end workflow for training the v2 structural alphabet model from the SCOPe baseline data.

## 1. Expected data layout

The repository should contain:

```text
data/
  raw/
    pdbs_train.txt
    pdbs_val.txt
    scop_lookup.tsv
    tmaln-06.out

  derived/
    pairfiles/
      tmaln-06.train.out
      tmaln-06.val.out

  external/
    foldseek_scop40/
      pdb_by_sid/
        d1xxxx_
        d2yyyy_
        ...
```

The raw files define the fixed SCOPe baseline. The derived pairfiles define which structural alignments are used for training and validation.

Training uses:

```text
data/derived/pairfiles/tmaln-06.train.out
```

Validation uses:

```text
data/derived/pairfiles/tmaln-06.val.out
```

Do not train directly from `data/raw/tmaln-06.out`.

---

## 2. Recreate derived pairfiles

Run from the repository root:

```bash
mkdir -p data/derived/pairfiles

LC_ALL=C awk 'NF {print $1}' data/raw/pdbs_train.txt | sort -u > /tmp/train_sids.txt
LC_ALL=C awk 'NF {print $1}' data/raw/pdbs_val.txt   | sort -u > /tmp/val_sids.txt
LC_ALL=C cat /tmp/train_sids.txt /tmp/val_sids.txt | sort -u > /tmp/assigned_sids.txt

awk 'NR==FNR {train[$1]=1; next} ($1 in train) && ($2 in train)' \
  /tmp/train_sids.txt data/raw/tmaln-06.out \
  > data/derived/pairfiles/tmaln-06.train.out

awk 'NR==FNR {val[$1]=1; next} ($1 in val) && ($2 in val)' \
  /tmp/val_sids.txt data/raw/tmaln-06.out \
  > data/derived/pairfiles/tmaln-06.val.out
```

Check expected counts:

```bash
wc -l \
  data/raw/tmaln-06.out \
  data/derived/pairfiles/tmaln-06.train.out \
  data/derived/pairfiles/tmaln-06.val.out
```

Expected approximately:

```text
24525 data/raw/tmaln-06.out
20657 data/derived/pairfiles/tmaln-06.train.out
3814  data/derived/pairfiles/tmaln-06.val.out
```

---

## 3. Build training arrays

Create processed arrays from the pairfiles. This step expands CIGAR alignments into aligned residue-descriptor pairs, applies descriptor-validity filtering, applies Cα-distance filtering, and standardizes features.

Example script:

```bash
python scripts/build_training_arrays.py \
  --pdb-dir data/external/foldseek_scop40/pdb_by_sid \
  --train-pairfile data/derived/pairfiles/tmaln-06.train.out \
  --val-pairfile data/derived/pairfiles/tmaln-06.val.out \
  --out-dir data/processed/scop_ca5_v1 \
  --virt 270 0 2 \
  --max-ca-dist 5.0 \
  --max-pairs-per-alignment 1024 \
  --seed 123
```

Expected outputs:

```text
data/processed/scop_ca5_v1/
  train_x.npy
  train_y.npy
  val_x.npy
  val_y.npy
  scaler.npz
  report.json
```

The scaler must be fit on training features only and reused for validation and model export.

---

## 4. Minimal `build_training_arrays.py`

```python
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tdi.training_data import align_features, fit_standardizer, transform


def read_pairfile(path: Path):
    with path.open() as handle:
        for row_id, line in enumerate(handle):
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            yield row_id, parts[0], parts[1], parts[2]


def build_split(pairfile: Path, pdb_dir: str, virt, max_ca_dist, max_pairs, seed):
    xs = []
    ys = []
    report = {
        "pairfile": str(pairfile),
        "alignment_rows": 0,
        "final_examples": 0,
        "pairs_before_filters": 0,
        "pairs_after_descriptor_validity": 0,
        "pairs_after_ca_filter": 0,
        "pairs_after_max_pairs": 0,
        "skipped_rows": 0,
    }

    for row_id, sid1, sid2, cigar in read_pairfile(pairfile):
        report["alignment_rows"] += 1

        try:
            x, y, meta = align_features(
                pdb_dir=pdb_dir,
                virtual_center=virt,
                sid1=sid1,
                sid2=sid2,
                cigar_string=cigar,
                max_ca_dist=max_ca_dist,
                max_pairs=max_pairs,
                seed=seed + row_id,
            )
        except Exception as exc:
            report["skipped_rows"] += 1
            continue

        report["pairs_before_filters"] += int(meta.get("n_pairs_before_filters", 0))
        report["pairs_after_descriptor_validity"] += int(
            meta.get("n_pairs_after_descriptor_validity", 0)
        )
        report["pairs_after_ca_filter"] += int(meta.get("n_pairs_after_ca_filter", 0))
        report["pairs_after_max_pairs"] += int(meta.get("n_pairs_after_max_pairs", 0))

        if len(x) > 0:
            xs.append(x)
            ys.append(y)

    if xs:
        x_all = np.vstack(xs).astype(np.float32)
        y_all = np.vstack(ys).astype(np.float32)
    else:
        x_all = np.zeros((0, 10), dtype=np.float32)
        y_all = np.zeros((0, 10), dtype=np.float32)

    report["final_examples"] = int(len(x_all))
    return x_all, y_all, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--train-pairfile", type=Path, required=True)
    parser.add_argument("--val-pairfile", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--virt", type=float, nargs=3, default=[270.0, 0.0, 2.0])
    parser.add_argument("--max-ca-dist", type=float, default=5.0)
    parser.add_argument("--max-pairs-per-alignment", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    virt = tuple(args.virt)

    train_x, train_y, train_report = build_split(
        args.train_pairfile,
        args.pdb_dir,
        virt,
        args.max_ca_dist,
        args.max_pairs_per_alignment,
        args.seed,
    )

    val_x, val_y, val_report = build_split(
        args.val_pairfile,
        args.pdb_dir,
        virt,
        args.max_ca_dist,
        args.max_pairs_per_alignment,
        args.seed,
    )

    mean, std = fit_standardizer(train_x)

    np.save(args.out_dir / "train_x.npy", transform(train_x, mean, std))
    np.save(args.out_dir / "train_y.npy", transform(train_y, mean, std))
    np.save(args.out_dir / "val_x.npy", transform(val_x, mean, std))
    np.save(args.out_dir / "val_y.npy", transform(val_y, mean, std))
    np.savez(args.out_dir / "scaler.npz", mean=mean, std=std)

    report = {
        "train": train_report,
        "val": val_report,
        "scaler": {
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "preprocessing": {
            "virt": args.virt,
            "max_ca_dist": args.max_ca_dist,
            "max_pairs_per_alignment": args.max_pairs_per_alignment,
            "seed": args.seed,
        },
    }

    with (args.out_dir / "report.json").open("w") as handle:
        json.dump(report, handle, indent=2)

    print(f"Wrote processed dataset to {args.out_dir}")
    print(f"Train examples: {len(train_x)}")
    print(f"Val examples:   {len(val_x)}")


if __name__ == "__main__":
    main()
```

---

## 5. Train the model

Example command:

```bash
python scripts/train_v2.py \
  --data-dir data/processed/scop_ca5_v1 \
  --out-dir outputs/models/scop_vq_z4_seed1 \
  --seed 1 \
  --quantizer-type vq \
  --n-states 20 \
  --hidden-dim 64 \
  --z-dim 4 \
  --loss-type smooth_l1 \
  --batch-size 512 \
  --max-epochs 20 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --precision bf16-mixed
```

Use this as the default baseline:

```text
quantizer_type = vq
n_states = 20
z_dim = 4
hidden_dim = 64
loss_type = smooth_l1
batch_size = 512
```

---

## 6. Minimal `train_v2.py`

```python
#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tdi.model import TdiV2Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--quantizer-type", choices=["vq", "fsq"], default="vq")
    parser.add_argument("--n-states", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--z-dim", type=int, default=4)
    parser.add_argument("--loss-type", choices=["smooth_l1", "gaussian_nll"], default="smooth_l1")

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_x = np.load(args.data_dir / "train_x.npy")
    train_y = np.load(args.data_dir / "train_y.npy")
    val_x = np.load(args.data_dir / "val_x.npy")
    val_y = np.load(args.data_dir / "val_y.npy")
    scaler = np.load(args.data_dir / "scaler.npz")

    train_ds = TensorDataset(
        torch.from_numpy(train_x).float(),
        torch.from_numpy(train_y).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_x).float(),
        torch.from_numpy(val_y).float(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    model = TdiV2Model(
        input_dim=train_x.shape[1],
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        n_states=args.n_states,
        quantizer_type=args.quantizer_type,
        loss_type=args.loss_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Optional but useful for EMA-VQ: initialize centroids from real encoder outputs.
    if hasattr(model.quantizer, "init_codebook"):
        model.eval()
        with torch.no_grad():
            init_batch = torch.from_numpy(train_x[: min(len(train_x), 8192)]).float()
            z = model.encoder(init_batch)
            model.quantizer.init_codebook(z)

    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=args.out_dir / "checkpoints",
            filename="{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        default_root_dir=args.out_dir,
        log_every_n_steps=50,
    )

    trainer.fit(model, train_loader, val_loader)

    export_dir = args.out_dir / "exported"
    model.export_model(
        export_dir,
        mean=scaler["mean"],
        std=scaler["std"],
    )

    print(f"Exported model to {export_dir}")


if __name__ == "__main__":
    main()
```

---

## 7. Train multiple seeds

```bash
for seed in 1 2 3 4 5; do
  python scripts/train_v2.py \
    --data-dir data/processed/scop_ca5_v1 \
    --out-dir outputs/models/scop_vq_z4_seed${seed} \
    --seed ${seed} \
    --quantizer-type vq \
    --n-states 20 \
    --hidden-dim 64 \
    --z-dim 4 \
    --loss-type smooth_l1 \
    --batch-size 512 \
    --max-epochs 20 \
    --precision bf16-mixed
done
```

Use `32-true` precision if bf16 is unsupported:

```bash
--precision 32-true
```

---

## 8. Exported model contents

After training, each run should contain:

```text
outputs/models/scop_vq_z4_seed1/
  checkpoints/
  exported/
    encoder_state_dict.pt
    model_config.json
    feature_scaler.json
    centroids.npy
```

The exported folder is what downstream encoding/evaluation code should consume.

---

## 9. Common checks

Check that processed data exists:

```bash
python - <<'PY'
import numpy as np
from pathlib import Path

p = Path("data/processed/scop_ca5_v1")
for name in ["train_x.npy", "train_y.npy", "val_x.npy", "val_y.npy"]:
    arr = np.load(p / name)
    print(name, arr.shape, arr.dtype, np.isfinite(arr).all())
PY
```

Check the scaler:

```bash
python - <<'PY'
import numpy as np
s = np.load("data/processed/scop_ca5_v1/scaler.npz")
print("mean:", s["mean"])
print("std:", s["std"])
PY
```

Check exported model files:

```bash
ls -lh outputs/models/scop_vq_z4_seed1/exported
```
