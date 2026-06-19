# Training

Scripts and data for learning the 3Di structural alphabet via VQ-VAE.

## Quick start

```bash
cd training
./learnAlphabet.sh 20 100 ../data/pdbs_train.txt ../data/pdbs_val.txt tmp/output
```

Outputs `encoder.pt`, `states.txt`, and `sub_score.mat` into `tmp/output/`.
Downloads ~500MB of SCOPe PDBs and the SSW aligner on first run.

## Scripts

| Script | Purpose |
|---|---|
| `learnAlphabet.sh` | Core pipeline: train N seeds, pick best by benchmark AUC |
| `crossval.sh` | 4-fold cross-validation against the `data/v1` reference alphabet |
| `koptimization.sh` | Grid search over alphabet size K=4..40 to find optimal K |
| `run-benchmark.sh` | Encode PDBs → Smith-Waterman → ROC AUC at fam/sfam/fold level |
| `run-smithwaterman.sh` | Parallel Smith-Waterman search (64 jobs via semaphore) |
| `roc1.awk` | Sensitivity-at-1FP/query metric broken down by SCOP hierarchy |
| `ssw.patch` | C patch to SSW library: expands buffer/matrix for alphabets >20 states |

All Python steps are invoked via `uv run ../scripts/<script>.py`.

## Python entry points

| Script | Purpose |
|---|---|
| `scripts/train.py` | Train VQ-VAE, export encoder + states |
| `scripts/create_training_data.py` | Extract aligned features from PDB pairs |
| `scripts/encode_pdbs.py` | Encode PDB structures to 3Di sequences |
| `scripts/create_submat.py` | Build substitution matrix from alignments |
| `scripts/split_folds.py` | Partition SCOP domains into cross-validation folds |

## Data (`../data/`)

| File | Description |
|---|---|
| `pdbs_train.txt` | SCOPe 2.07 domain SIDs used for training (8952 domains) |
| `pdbs_val.txt` | SCOPe 2.07 domain SIDs used for validation (2206 domains) |
| `scop_lookup.tsv` | SCOP classification per domain (fam/sfam/fold) |
| `tmaln-06.out` | TM-align all-vs-all on SCOPe 2.07, TMscore ≥ 0.6, with CIGAR strings |
| `v1/` | Trained Foldseek v1 alphabet (reference model) |
| `v1/encoder.pt` | Trained encoder network weights |
| `v1/decoder.pt` | Trained decoder network weights |
| `v1/states.txt` | VQ centroid coordinates (20 × 2) |
| `v1/sub_score.mat` | 20-state substitution matrix (half-bit log-odds) |

## Requirements

Dependencies are managed via `uv` — run `uv sync` from the repo root.
External tools fetched automatically by `learnAlphabet.sh`:
- SSW (Smith-Waterman aligner, compiled from source)
- SCOPe PDB structures (~500MB, downloaded once to `tmp/pdb/`)
