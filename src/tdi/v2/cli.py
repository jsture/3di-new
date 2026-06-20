"""CLI entrypoint and evaluation actions for v2."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .encode import process_pdb
from .model import AlphabetModel
from .submat import accumulate_counts, calc_alphabet_mi, write_mat
from .util import parse_pairfile_line, resolve_pdb_path


def run_evaluate(args: argparse.Namespace) -> None:
    """Run model evaluation pipeline on sequence alignments."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load exported model and feature standardizer scaler
    print(f"Loading exported model from {args.model_dir}...")
    model, mean, std = AlphabetModel.load(args.model_dir)

    if model.n_states > len(model.letters):
        raise ValueError(
            f"Model has {model.n_states} states, but only {len(model.letters)} letters are "
            f"available in the alphabet definition."
        )

    # Resolve virtual center from args or config
    if args.virt is not None:
        virt_cb: tuple[float, float, float] = (
            float(args.virt[0]),
            float(args.virt[1]),
            float(args.virt[2]),
        )
    elif getattr(model, "virtual_center", None) is not None:
        vc = model.virtual_center
        assert vc is not None
        virt_cb = (float(vc[0]), float(vc[1]), float(vc[2]))
    else:
        raise ValueError("Virtual center not found in model config, and --virt was not provided.")

    # 2. Extract unique structure identifiers from pairfile.
    unique_sids = set()
    with open(args.pairfile) as f:
        for line in f:
            res = parse_pairfile_line(line)
            if res is not None:
                unique_sids.add(res[0])
                unique_sids.add(res[1])

    # 3. Encode PDB files into 3Di sequences
    print(f"Encoding {len(unique_sids)} PDB files using trained encoder...")
    sid2seq = {}
    failed_sids = []
    for sid in sorted(unique_sids):
        try:
            resolved_path = resolve_pdb_path(args.pdb_dir, sid)
            _, seq = process_pdb(
                resolved_path.name,
                model,
                None,
                str(resolved_path.parent),
                virt_cb,
                args.invalid_state,
                mean=mean,
                std=std,
            )
            sid2seq[sid] = seq
        except Exception as e:
            print(f"Error encoding {sid}: {e}", file=sys.stderr)
            failed_sids.append(sid)

    n_requested = len(unique_sids)
    n_encoded = len(sid2seq)
    n_failed = len(failed_sids)
    failure_rate = n_failed / n_requested if n_requested > 0 else 0.0
    failed_sids_sample = failed_sids[:10]

    # Report failure diagnostics in stdout
    print(
        f"Encoding diagnostics: requested {n_requested}, encoded {n_encoded}, "
        f"failed {n_failed} ({failure_rate:.1%})."
    )
    if n_failed > 0:
        print(f"Failed sample IDs: {failed_sids_sample}", file=sys.stderr)

    max_failure_rate = getattr(args, "max_failure_rate", 1.0)
    if failure_rate > max_failure_rate:
        raise RuntimeError(
            f"Encoding failure rate {failure_rate:.1%} exceeds maximum allowed "
            f"threshold {max_failure_rate:.1%} (failed: {n_failed}/{n_requested}). "
            f"Failed sample: {failed_sids_sample}"
        )

    # 4. Save sequences to sequences.txt
    seq_path = out_dir / "sequences.txt"
    with open(seq_path, "w") as f_seq:
        for sid, seq in sorted(sid2seq.items()):
            f_seq.write(f"{sid} {seq}\n")
    print(f"Saved encoded sequences to {seq_path}")

    # 5. Accumulate transition count matrices
    n_states = model.n_states
    alphabet = model.letters[:n_states]
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

    # 9. State-usage diagnostics over the encoded sequences. Mirrors the per-state usage,
    # dead-state fraction, and normalized entropy the model computes in validation, surfaced
    # at evaluation time. Invalid-state characters are not in letter2idx and are excluded.
    usage_counts = np.zeros(n_states, dtype=np.int64)
    for seq in sid2seq.values():
        for ch in seq:
            state_idx = letter2idx.get(ch)
            if state_idx is not None:
                usage_counts[state_idx] += 1
    total_states = int(usage_counts.sum())
    if total_states > 0:
        p = usage_counts / total_states
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = float(-(p * np.log(p + 1e-10)).sum())
        normalized_entropy = entropy / np.log(n_states) if n_states > 1 else 0.0
        dead_state_fraction = float(np.sum(p < 1e-5) / n_states)
    else:
        normalized_entropy = 0.0
        dead_state_fraction = 1.0

    # 10. Compile evaluation summary report
    report = {
        "n_sequences": len(sid2seq),
        "total_counts": int(counts.sum()),
        "mi": float(mi),
        "mi_tot": float(mi_tot),
        "n_letters": n_states,
        "letters": alphabet,
        "state_usage": usage_counts.tolist(),
        "dead_state_fraction": dead_state_fraction,
        "normalized_entropy": normalized_entropy,
    }
    report_path = out_dir / "evaluation_report.json"
    with open(report_path, "w") as f_rep:
        json.dump(report, f_rep, indent=2)
    print(f"Saved evaluation report to {report_path}")


def main() -> None:
    """Main CLI driver."""
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        from .train import main as train_main

        sys.argv = [sys.argv[0], *sys.argv[2:]]
        train_main()
        return

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
        default=None,
        help="State for invalid coordinates (defaults to the model config's invalid_state).",
    )

    args = parser.parse_args()
    if args.command == "evaluate":
        run_evaluate(args)


if __name__ == "__main__":
    main()
