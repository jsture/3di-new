"""CLI entrypoint and evaluation actions for v2."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .encode import LETTERS, process_pdb
from .model import TdiV2Model
from .submat import accumulate_counts, calc_alphabet_mi, write_mat


def run_evaluate(args: argparse.Namespace) -> None:
    """Run model evaluation pipeline on sequence alignments."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load exported model and feature standardizer scaler
    print(f"Loading exported model from {args.model_dir}...")
    model, mean, std = TdiV2Model.load_from_export(args.model_dir)

    # 2. Extract unique structure identifiers from pairfile
    unique_sids = set()
    alignments = []
    with open(args.pairfile) as f:
        for line in f:
            parts = line.rstrip("\n").split()
            if len(parts) >= 3:
                sid1, sid2, cigar = parts[0], parts[1], parts[2]
                unique_sids.add(sid1)
                unique_sids.add(sid2)
                alignments.append((sid1, sid2, cigar))

    # 3. Encode PDB files into 3Di sequences
    print(f"Encoding {len(unique_sids)} PDB files using trained encoder...")
    sid2seq = {}
    virt_cb = (args.virt[0], args.virt[1], args.virt[2])
    for sid in sorted(unique_sids):
        # We try both sid, sid + ".pdb", or look directly under pdb_dir
        pdb_candidates = [sid, f"{sid}.pdb"]
        success = False
        for cand in pdb_candidates:
            pdb_path = Path(args.pdb_dir) / cand
            if pdb_path.exists():
                try:
                    _, seq = process_pdb(
                        cand,
                        model,
                        None,
                        args.pdb_dir,
                        virt_cb,
                        args.invalid_state,
                        mean=mean,
                        std=std,
                    )
                    sid2seq[sid] = seq
                    success = True
                    break
                except Exception as e:
                    print(f"Error encoding {sid}: {e}", file=sys.stderr)
                    break
        if not success:
            print(f"Warning: Could not find/process PDB file for {sid}", file=sys.stderr)

    # 4. Save sequences to sequences.txt
    seq_path = out_dir / "sequences.txt"
    with open(seq_path, "w") as f_seq:
        for sid, seq in sorted(sid2seq.items()):
            f_seq.write(f"{sid} {seq}\n")
    print(f"Saved encoded sequences to {seq_path}")

    # 5. Accumulate transition count matrices
    n_states = model.n_states
    alphabet = LETTERS[:n_states]
    letter2idx = {char: i for i, char in enumerate(alphabet)}

    print("Accumulating transition counts from alignments...")
    counts, counts_prev = accumulate_counts(args.pairfile, sid2seq, letter2idx, n_states)

    # 6. Calculate Mutual Information metrics
    mi, mi_tot = calc_alphabet_mi(counts, counts_prev)
    print(f"Mutual Information (MI): {mi:.4f}")
    print(f"Transition-Adjusted MI (MI_tot): {mi_tot:.4f}")

    # 7. Compute log-odds substitution scoring matrix
    p_ab = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=np.float32)
    p_a = p_ab.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        scores = 2 * np.log2(p_ab / (p_a * p_a[:, np.newaxis]))
    scores[~np.isfinite(scores)] = 0
    scores = np.rint(scores).astype(int)

    # 8. Save substitution matrix to submat.txt
    mat_path = out_dir / "submat.txt"
    with open(mat_path, "w") as f_mat:
        write_mat(f_mat, list(alphabet), scores)
    print(f"Saved substitution matrix to {mat_path}")

    # 9. Compile evaluation summary report
    report = {
        "n_sequences": len(sid2seq),
        "total_counts": int(counts.sum()),
        "mi": float(mi),
        "mi_tot": float(mi_tot),
        "n_letters": n_states,
        "letters": alphabet,
    }
    report_path = out_dir / "evaluation_report.json"
    with open(report_path, "w") as f_rep:
        json.dump(report, f_rep, indent=2)
    print(f"Saved evaluation report to {report_path}")


def main() -> None:
    """Main CLI driver."""
    parser = argparse.ArgumentParser(description="Tdi-v2 CLI tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model on alignments.")
    eval_parser.add_argument(
        "--model_dir", type=str, required=True, help="Path to exported model folder."
    )
    eval_parser.add_argument(
        "--pdb_dir", type=str, required=True, help="Directory containing PDB files."
    )
    eval_parser.add_argument(
        "--pairfile", type=str, required=True, help="Path to structural alignment pairfile."
    )
    eval_parser.add_argument(
        "--out_dir", type=str, required=True, help="Output directory to save evaluation results."
    )
    eval_parser.add_argument(
        "--virt",
        type=float,
        nargs=3,
        required=True,
        help="Virtual center parameters (alpha, beta, d).",
    )
    eval_parser.add_argument(
        "--invalid_state",
        type=str,
        default="X",
        help="State used to represent invalid coordinates.",
    )

    args = parser.parse_args()
    if args.command == "evaluate":
        run_evaluate(args)


if __name__ == "__main__":
    main()
