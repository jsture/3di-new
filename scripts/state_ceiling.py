#!/usr/bin/env python3
"""Measure what a 20-state alphabet can actually achieve on these descriptors.

The saturation curve bounds ``I(x; y)`` -- what *unlimited* states could carry. Under a fixed
20-symbol budget that is the wrong target: quantizing to 20 cells destroys information no
objective can recover, so the gap between a model's 1.50 bits and the descriptors' >=2.13 is
not all headroom. This probe estimates the reachable part.

**Method.** Cluster the descriptors finely (K=300 by default), then greedily merge state pairs
-- at each step the merge costing the least mutual information -- down to 20. The MI of the
resulting 20-state partition is *constructive*: an alphabet achieving it demonstrably exists,
because this one does. That makes it a target a training objective can be aimed at, unlike an
information-theoretic bound.

**It is a lower bound on the best 20-state partition**, for two reasons that both point the
same way: greedy merging is not optimal, and merges can only coarsen the K=300 boundaries,
never redraw them. A better 20-state partition may exist; a worse conclusion cannot be drawn
from this number being high.

**Held-out scoring.** The merge schedule is chosen on one half of the aligned pairs and scored
on the other. Choosing merges to maximize MI on the same pairs that report it would inflate the
answer, and the answer is the whole point of running this.

A random-merge control runs the same schedule with arbitrary merges, so the greedy result can
be read against what any coarsening of the same clustering would give.

Usage::

    uv run python scripts/state_ceiling.py --out runs/probes/state_ceiling
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.descriptor_mi_curve import (  # noqa: E402
    evaluation_equivalent_pairs,
    joint_counts,
    load_fit_sample,
)
from tdi.v2.submat import (  # noqa: E402
    merge_columns,
    mutual_information_from_counts,
)
from tdi.v2.train import _read_provenance  # noqa: E402

# Published reference points on this same validation pairfile, all raw pair MI in bits.
LEARNED_MI = 1.5010  # best trained 20-state arm (vq_kmeans)
FOLDSEEK_V1_MI = 1.4171  # published Foldseek 3Di release
KMEANS_K20_MI = 1.2098  # raw k-means at K=20, no learning


@dataclass
class MergeStep:
    """One point on the merge trajectory, scored both in-sample and held out."""

    n_states: int
    mi_merge_half_bits: float
    mi_score_half_bits: float
    strategy: str


def _f(counts: np.ndarray) -> np.ndarray:
    """``c * log2(c)`` with the ``0 * log2(0) = 0`` convention."""
    counts = np.asarray(counts, dtype=np.float64)
    return np.where(counts > 0, counts * np.log2(np.where(counts > 0, counts, 1.0)), 0.0)


def mi_from_sufficient_stats(counts: np.ndarray) -> float:
    """Mutual information of a symmetric joint, via the entropy decomposition.

    ``MI = H(X) + H(Y) - H(X,Y)``, and for a symmetric table ``H(X) = H(Y)``. Writing each
    entropy as ``log2(N) - (1/N) * sum(c * log2 c)`` gives

        ``MI = log2(N) + (F_joint - 2 * F_marginal) / N``

    which is the form the incremental merge search differentiates. Kept separate from
    ``mutual_information_from_counts`` so the two can be cross-checked against each other.
    """
    total = counts.sum()
    if total <= 0:
        return 0.0
    f_joint = float(_f(counts).sum())
    f_marginal = float(_f(counts.sum(axis=1)).sum())
    return math.log2(total) + (f_joint - 2.0 * f_marginal) / total


def merge_deltas_for_state(counts: np.ndarray, i: int) -> np.ndarray:
    """Change in MI from merging state ``i`` into each other state, vectorized over ``j``.

    Recomputing MI for every candidate merge would cost O(S^2) per candidate and O(S^4) per
    step, which is hopeless at K=300 (~45,000 candidates). Merging only touches rows and
    columns ``i`` and ``j``, so the delta has a closed form costing O(S) per candidate, and
    this evaluates all ``j`` at once.

    Args:
        counts: Symmetric joint count matrix of shape (S, S).
        i: State whose merges are being evaluated.

    Returns:
        Array of shape (S,) giving the MI change for merging ``i`` with each ``j``. The entry
        at ``j == i`` is ``-inf``, since a state cannot merge into itself.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n_states = len(counts)
    total = counts.sum()

    row_i = counts[i]
    # Off-block joint term: rows i and j collapse into one, and by symmetry the columns do
    # too, hence the factor of 2 applied below.
    pairwise = _f(row_i[None, :] + counts) - _f(row_i)[None, :] - _f(counts)
    off_block = pairwise.sum(axis=1) - pairwise[:, i] - np.diagonal(pairwise)

    # The 2x2 block (i,i), (i,j), (j,i), (j,j) collapses into a single cell.
    diagonal = np.diagonal(counts)
    block = (
        _f(diagonal[i] + 2.0 * row_i + diagonal)
        - _f(np.full(n_states, diagonal[i]))
        - 2.0 * _f(row_i)
        - _f(diagonal)
    )
    delta_joint = 2.0 * off_block + block

    marginals = counts.sum(axis=1)
    delta_marginal = _f(marginals[i] + marginals) - _f(marginals[i]) - _f(marginals)

    deltas = (delta_joint - 2.0 * delta_marginal) / total
    deltas[i] = -np.inf
    return deltas


def best_merge(counts: np.ndarray) -> tuple[int, int, float]:
    """Find the state pair whose merge costs the least mutual information.

    Returns:
        A tuple of (i, j, delta_mi) with ``i < j``. ``delta_mi`` is normally negative:
        coarsening a partition cannot increase MI in population, though it can rise slightly
        on a finite sample when a merge removes a sparsely-populated state.
    """
    n_states = len(counts)
    best = (0, 1, -np.inf)
    for i in range(n_states - 1):
        deltas = merge_deltas_for_state(counts, i)
        # Only consider j > i, so each unordered pair is evaluated once.
        deltas[: i + 1] = -np.inf
        j = int(np.argmax(deltas))
        if deltas[j] > best[2]:
            best = (i, j, float(deltas[j]))
    return best


def merge_to_target(
    merge_counts: np.ndarray,
    score_counts: np.ndarray,
    target_states: int,
    strategy: str,
    rng: np.random.Generator,
    verify_every: int = 25,
) -> list[MergeStep]:
    """Merge states down to ``target_states``, scoring each step on held-out pairs.

    Both matrices receive the *same* merge at every step: the schedule is chosen using
    ``merge_counts`` alone, and ``score_counts`` only ever reports what that schedule achieves
    on pairs the choice never saw.

    Args:
        merge_counts: Symmetric joint over the half used to choose merges.
        score_counts: Symmetric joint over the held-out half.
        target_states: State count to stop at.
        strategy: ``"greedy"`` (least-cost merge) or ``"random"`` (control).
        rng: Source of randomness for the random control.
        verify_every: Cross-check the incremental MI against a full recomputation this often;
            0 disables. Guards the closed-form delta against silent drift.

    Returns:
        The trajectory, one entry per state count from the start down to ``target_states``.
    """
    trajectory = [
        MergeStep(
            n_states=len(merge_counts),
            mi_merge_half_bits=mi_from_sufficient_stats(merge_counts),
            mi_score_half_bits=mi_from_sufficient_stats(score_counts),
            strategy=strategy,
        )
    ]

    while len(merge_counts) > target_states:
        if strategy == "greedy":
            i, j, _ = best_merge(merge_counts)
        else:
            i, j = sorted(rng.choice(len(merge_counts), size=2, replace=False))

        # merge_columns is the tested primitive; the fast delta above only chooses the pair.
        merge_counts = merge_columns(merge_counts, i, j)
        score_counts = merge_columns(score_counts, i, j)

        n_states = len(merge_counts)
        mi_merge = mi_from_sufficient_stats(merge_counts)
        if verify_every and n_states % verify_every == 0:
            direct = mutual_information_from_counts(merge_counts)
            if not math.isclose(mi_merge, direct, rel_tol=1e-9, abs_tol=1e-9):
                raise AssertionError(
                    f"Incremental MI {mi_merge} disagrees with recomputed {direct} at "
                    f"{n_states} states."
                )

        trajectory.append(
            MergeStep(
                n_states=n_states,
                mi_merge_half_bits=mi_merge,
                mi_score_half_bits=mi_from_sufficient_stats(score_counts),
                strategy=strategy,
            )
        )
    return trajectory


def split_pairs(n_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split aligned pairs into a merge half and a held-out scoring half."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_pairs)
    midpoint = n_pairs // 2
    return order[:midpoint], order[midpoint:]


def _print_report(
    greedy: list[MergeStep], random_control: list[MergeStep], target_states: int
) -> None:
    """Print the trajectory milestones and the decision this probe exists to settle."""
    by_states = {step.n_states: step for step in greedy}
    milestones = sorted(
        (s for s in (300, 200, 150, 100, 75, 50, 40, 30, 25, target_states) if s in by_states),
        reverse=True,
    )

    print("\nGreedy merge trajectory (MI in bits, raw pair MI)")
    header = f"{'states':>7} {'merge-half':>11} {'held-out':>10} {'overfit':>9}"
    print(header)
    print("-" * len(header))
    for states in milestones:
        step = by_states[states]
        gap = step.mi_merge_half_bits - step.mi_score_half_bits
        print(
            f"{states:7d} {step.mi_merge_half_bits:11.4f} {step.mi_score_half_bits:10.4f} "
            f"{gap:9.4f}"
        )

    final = by_states[target_states]
    control = next(s for s in random_control if s.n_states == target_states)
    ceiling = final.mi_score_half_bits

    print(f"\n{target_states}-state achievable MI (held out):  {ceiling:.4f} bits")
    print(f"  random-merge control:                {control.mi_score_half_bits:.4f} bits")
    print(f"  learned model (vq_kmeans):           {LEARNED_MI:.4f} bits")
    print(f"  published Foldseek v1:               {FOLDSEEK_V1_MI:.4f} bits")
    print(f"  raw k-means at K={target_states}:                 {KMEANS_K20_MI:.4f} bits")

    headroom = ceiling - LEARNED_MI
    print(
        f"\nHeadroom over the learned model: {headroom:+.4f} bits "
        f"({100 * headroom / LEARNED_MI:+.1f}%)"
    )
    if headroom <= 0.05:
        print(
            "  READ: the learned model is at or above this constructive floor. A better\n"
            "  partition of THESE descriptors into this many states is not clearly available,\n"
            "  so objective work -- anything that only rearranges the same inputs into the\n"
            "  same number of cells -- has little to win.\n"
            "  This says nothing against richer descriptors. The bottleneck is not full in an\n"
            "  entropy sense (log2(K) is far above this value); what is exhausted is the\n"
            "  aligned information THESE inputs expose to a partition of this size. Different\n"
            "  inputs move the ceiling itself, so re-run this probe on new descriptors to see\n"
            "  whether they raise it before building anything on them."
        )
    elif headroom < 0.20:
        print(
            "  READ: a real but modest gap, close to the seed-noise floor of this project.\n"
            "  Worth confirming over matched seeds before committing to objective work."
        )
    else:
        print(
            "  READ: substantial headroom at a fixed state budget. A partition achieving this\n"
            "  exists, so the gap is reachable in principle and an objective aimed at pair MI\n"
            "  has a concrete target."
        )
    print(
        "\nThis is a LOWER bound on the best partition at this state count: greedy merging is\n"
        "not optimal, and merges can only coarsen the initial clustering, never redraw it."
    )


def main() -> None:
    """Estimate the achievable MI of a fixed-size alphabet built on these descriptors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/probes/state_ceiling", help="Output directory.")
    parser.add_argument(
        "--processed-dir",
        default="data/processed/scop_ca5_v1",
        help="Processed dataset supplying the scaler and clustering sample.",
    )
    parser.add_argument(
        "--pdb-dir",
        default="data/external/foldseek_scop40/pdb_by_sid",
        help="Directory of structures to encode.",
    )
    parser.add_argument(
        "--pairfile",
        default="data/derived/pairfiles/tmaln-06.val.out",
        help="Validation alignment pairfile.",
    )
    parser.add_argument(
        "--k-start", type=int, default=300, help="Initial fine clustering to merge down from."
    )
    parser.add_argument(
        "--target-states", type=int, default=20, help="State budget to measure the ceiling at."
    )
    parser.add_argument(
        "--fit-samples", type=int, default=500_000, help="Training descriptors to cluster on."
    )
    parser.add_argument(
        "--max-alignments", type=int, default=None, help="Cap parsed alignments (debugging)."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling and clustering.")
    parser.add_argument(
        "--virt",
        type=float,
        nargs=3,
        default=None,
        help="Virtual center override; otherwise read from the dataset manifest.",
    )
    args = parser.parse_args()

    if args.target_states >= args.k_start:
        parser.error("--k-start must exceed --target-states; there would be nothing to merge.")

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_center, _ = _read_provenance(processed_dir)
    center = args.virt if args.virt is not None else manifest_center
    if center is None:
        parser.error("No virtual center in the dataset manifest; pass --virt explicitly.")
    virtual_center = (float(center[0]), float(center[1]), float(center[2]))

    print(f"Building evaluation-equivalent pairs from {args.pairfile}...")
    x_raw, y_raw, pair_stats = evaluation_equivalent_pairs(
        args.pdb_dir, args.pairfile, virtual_center, args.max_alignments
    )
    print(f"  {len(x_raw):,} aligned pairs from {pair_stats['alignments_used']:,} alignments.")

    fit_sample, mean, std = load_fit_sample(processed_dir, args.fit_samples, args.seed)
    x_std = (x_raw - mean) / std
    y_std = (y_raw - mean) / std

    print(f"\nClustering to K={args.k_start} on {len(fit_sample):,} training descriptors...")
    model = MiniBatchKMeans(
        n_clusters=args.k_start, random_state=args.seed, n_init="auto", batch_size=4096
    )
    model.fit(fit_sample)
    labels_x = np.asarray(model.predict(x_std))
    labels_y = np.asarray(model.predict(y_std))

    merge_rows, score_rows = split_pairs(len(labels_x), args.seed)
    merge_counts = joint_counts(labels_x[merge_rows], labels_y[merge_rows], args.k_start)
    score_counts = joint_counts(labels_x[score_rows], labels_y[score_rows], args.k_start)
    print(
        f"  merge half {len(merge_rows):,} pairs, held-out half {len(score_rows):,} pairs; "
        f"starting MI {mi_from_sufficient_stats(score_counts):.4f} bits (held out)."
    )

    print(f"\nGreedy merging {args.k_start} -> {args.target_states}...")
    greedy = merge_to_target(
        merge_counts,
        score_counts,
        args.target_states,
        "greedy",
        np.random.default_rng(args.seed),
    )

    print(f"Random-merge control {args.k_start} -> {args.target_states}...")
    random_control = merge_to_target(
        merge_counts,
        score_counts,
        args.target_states,
        "random",
        np.random.default_rng(args.seed),
    )

    _print_report(greedy, random_control, args.target_states)

    final = next(s for s in greedy if s.n_states == args.target_states)
    control = next(s for s in random_control if s.n_states == args.target_states)
    payload = {
        "interpretation": (
            "mi_score_half_bits at the target state count is a CONSTRUCTIVE LOWER BOUND on "
            "what an alphabet of that size can achieve on these descriptors: a partition "
            "reaching it exists. Greedy merging is not optimal and can only coarsen the "
            "initial clustering, so the best partition of this size may score higher."
        ),
        "inputs": {
            "processed_dir": str(processed_dir),
            "pdb_dir": args.pdb_dir,
            "pairfile": args.pairfile,
            "virtual_center": list(virtual_center),
            "k_start": args.k_start,
            "target_states": args.target_states,
            "fit_samples": len(fit_sample),
            "seed": args.seed,
        },
        "pair_stats": pair_stats,
        "n_merge_pairs": len(merge_rows),
        "n_score_pairs": len(score_rows),
        "achievable_mi_bits": final.mi_score_half_bits,
        "random_control_mi_bits": control.mi_score_half_bits,
        "reference_learned_mi_bits": LEARNED_MI,
        "reference_foldseek_v1_mi_bits": FOLDSEEK_V1_MI,
        "reference_kmeans_k20_mi_bits": KMEANS_K20_MI,
        "headroom_over_learned_bits": final.mi_score_half_bits - LEARNED_MI,
        "trajectory": [asdict(step) for step in greedy],
        "random_trajectory": [asdict(step) for step in random_control],
    }
    with open(out_dir / "state_ceiling.json", "w") as f:
        json.dump(payload, f, indent=2)

    with open(out_dir / "state_ceiling.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n_states", "strategy", "mi_merge", "mi_score"])
        writer.writeheader()
        for step in [*greedy, *random_control]:
            writer.writerow(
                {
                    "n_states": step.n_states,
                    "strategy": step.strategy,
                    "mi_merge": step.mi_merge_half_bits,
                    "mi_score": step.mi_score_half_bits,
                }
            )
    print(f"\nWrote state_ceiling.json and state_ceiling.csv to {out_dir}")


if __name__ == "__main__":
    main()
