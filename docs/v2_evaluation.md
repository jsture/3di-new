# Downstream Alphabet Evaluation

This document describes how to evaluate the quality of the trained discrete structural alphabet using alignment metrics and substitution scoring.

Evaluation is performed via the `evaluate` command inside the `tdi.v2` CLI (implemented in [`src/tdi/v2/cli.py`](../src/tdi/v2/cli.py)).

---

## Running Evaluation

To evaluate a trained model checkpoint against validation alignments, run the following subcommand from the repository root:

```bash
uv run python -m tdi.v2 evaluate \
  --model_dir outputs/models/scop_v2_default_seed1 \
  --pdb_dir data/external/foldseek_scop40/pdb_by_sid \
  --pairfile data/derived/pairfiles/tmaln-06.val.out \
  --out_dir outputs/evaluations/scop_v2_default_seed1 \
  --virt 270.0 0.0 2.0
```

### CLI Arguments

- **`--model_dir`**: Path to the exported model folder containing model weights, scaling parameters, and VQ centroids.
- **`--pdb_dir`**: Path to the directory containing candidate PDB files.
- **`--pairfile`**: Path to the structural alignment pairfile to evaluate (e.g., validation split).
- **`--out_dir`**: Output directory to write encoded sequences and evaluation reports.
- **`--virt`**: Parameters defining the $C_\beta$ virtual center configuration (alpha, beta, d). Default SCOPe baseline setting is `270.0 0.0 2.0`.
- **`--invalid_state`**: State code used to denote missing or invalid coordinate positions. Defaults to `"X"`.

---

## Evaluation Outputs

The evaluation run writes the following artifacts to the specified `--out_dir`:

1. **`sequences.txt`**:
   Contains the structural alphabet encodings for all processed domains in the dataset (one domain per line, e.g., `<domain_id> <sequence>`).

2. **`submat.txt`**:
   A computed log-odds substitution scoring matrix (similar to BLOSUM/BLAST) derived from aligned residue transitions in the pairfile. Higher diagonal values reflect consistent state mappings within structurally aligned regions.

3. **`evaluation_report.json`**:
   Reports summary statistics and information-theoretic metrics for the alphabet:
   - **Mutual Information (MI)**: Measures the dependency between aligned residues.
   - **Transition-Adjusted MI ($MI_{tot}$)**: Adjusts the baseline MI to account for state transitions, evaluating the vocabulary's representation strength.

---

## Diagnostics

To print validation score progress or inspect model metrics directly, refer to the training summary reports generated in the preprocessing output directory.
