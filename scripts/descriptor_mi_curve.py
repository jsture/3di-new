#!/usr/bin/env python3
"""Measure how much aligned information the 10-D residue descriptors themselves carry.

States are a function of descriptors, so ``I(state_x; state_y) <= I(x; y)``: whatever the
descriptors share across a structural alignment bounds every alphabet built from them, at any
state count, under any training objective. This probe estimates that shared information by
clustering the descriptors at growing K and reading the resulting saturation curve.

**What the numbers are, and are not.** Every binned figure here is a *discretized lower
bound*, not a ceiling. Two biases run in opposite directions and neither is dominant by
construction:

* Binning is a function of the data, so by the same data-processing inequality it can only
  destroy information: ``I(q(x); q(y)) <= I(x; y)`` in population.
* The finite-sample plug-in estimator is positively biased, which inflates the estimate of
  that already-reduced quantity.

The net sign against continuous ``I(x; y)`` is therefore undetermined, and no single number
here should be quoted as a hard ceiling. The curve is read by *shape* and by agreement across
estimators. The evidence is asymmetric: a high curve is strong evidence the descriptors carry
information the current objective is not extracting, whereas a low curve is suggestive but not
conclusive, since a finer partition or a different estimator could still find structure.

**Evaluation-equivalent pairs.** The K=20 anchor only means something against the published
raw-k-means baseline if both see the same residue pairs. The processed validation arrays have
already been Ca-distance filtered (``max_ca_dist: 5.0``) and per-alignment capped
(``max_pairs_per_alignment: 768``); ``accumulate_counts`` applies neither. This script
therefore rebuilds pairs from the PDBs using the CIGAR and descriptor-validity semantics of
``run_evaluate`` alone. It deliberately does not route through ``align_features``, whose
``filter_ca_distance`` runs Kabsch superposition and drops ``too_few_pairs`` and
rank-deficient alignments even when ``max_ca_dist`` is ``None``.

**Orientation, and which bias correction applies where.** ``accumulate_counts`` records each
aligned position once and then mirrors it (``counts[a1, a2] += 1; counts[a2, a1] += 1``). This
script matches that, so ``mi_plugin_bits`` is directly comparable to a reported ``mi``.

That symmetrization writes every observation into two dependent cells, which breaks the
multinomial assumption behind Miller-Madow. Passing the one-orientation ``N`` does not repair
it -- neither the independence of the counts nor the degrees of freedom. So the two bias
estimates are reported against different tables and are not interchangeable:

* ``mi_miller_madow_bits`` corrects the *unsymmetrized* joint, where each observation
  contributes exactly one count and the analytic formula is valid.
* ``mi_bootstrap_corrected_bits`` corrects the symmetrized, evaluator-matching statistic by
  resampling whole aligned pairs -- the independent unit -- so the symmetrization is rebuilt
  inside every replicate.

Note also that the Miller-Madow correction is not signed: it lowers an inflated estimate only
when the joint occupies more bins than the marginals combined. On a concentrated joint it is
positive.

Usage::

    uv run python scripts/descriptor_mi_curve.py --out runs/probes/descriptor_mi
    uv run python scripts/descriptor_mi_curve.py --out runs/probes/descriptor_mi --ksg
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree
from scipy.special import digamma
from sklearn.cluster import MiniBatchKMeans

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.kmeans_baseline import _fit_centroids  # noqa: E402
from tdi.v2.submat import (  # noqa: E402
    miller_madow_corrected_mi,
    mutual_information_from_counts,
)
from tdi.v2.train import _read_provenance  # noqa: E402
from tdi.v2.training_data import extract_features, filter_valid_pairs  # noqa: E402
from tdi.v2.util import (  # noqa: E402
    parse_cigar,
    parse_pairfile_line,
    resolve_pdb_path,
)

# Curve points. K=20 matches the deployed alphabet size; the rest probe whether more states
# would find more shared information.
DEFAULT_K_VALUES = (20, 50, 100, 300, 1000)


@dataclass
class CurvePoint:
    """One K on the saturation curve, with the diagnostics needed to trust or reject it."""

    k: int
    estimator: str
    # Symmetrized (evaluator-matching) plug-in MI, uncorrected.
    mi_plugin_bits: float
    # Unsymmetrized plug-in MI, and the analytic correction that is valid only on it.
    mi_plugin_directed_bits: float
    mi_miller_madow_bits: float
    # Pair-level resampling bias for the symmetrized statistic, and the corrected value.
    mi_bootstrap_bias_bits: float
    mi_bootstrap_corrected_bits: float
    n_bootstrap: int
    mi_shuffled_bits: float
    occupied_x: int
    occupied_y: int
    occupied_joint: int
    n_pairs: int
    n_fit_samples: int


@dataclass
class DimensionPoint:
    """Per-dimension aligned MI under shared training-derived quantile bins."""

    dimension: int
    n_bins: int
    mi_plugin_bits: float
    mi_plugin_directed_bits: float
    mi_miller_madow_bits: float
    mi_shuffled_bits: float
    n_pairs: int


def evaluation_equivalent_pairs(
    pdb_dir: str,
    pairfile: str,
    virtual_center: tuple[float, float, float],
    max_alignments: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build aligned descriptor pairs using ``run_evaluate``'s semantics exactly.

    Applies CIGAR parsing and descriptor-validity masking and nothing else -- no Ca-distance
    filter, no per-alignment cap, no Kabsch superposition. Returns one orientation per
    aligned position; callers symmetrize the joint afterwards.

    Args:
        pdb_dir: Directory of structure files.
        pairfile: Alignment pairfile.
        virtual_center: The (alpha, beta, d) used to build descriptors.
        max_alignments: Stop after this many parsed alignments (None = all).

    Returns:
        A tuple of (x, y, stats) where x and y are (N, 10) aligned descriptors and stats
        records alignment and pair attrition.
    """
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    stats = {
        "alignments_parsed": 0,
        "alignments_used": 0,
        "alignments_failed": 0,
        "pairs_before_validity": 0,
        "pairs_out_of_range": 0,
        "pairs_after_validity": 0,
    }

    with open(pairfile) as f:
        for line in f:
            parsed = parse_pairfile_line(line)
            if parsed is None:
                continue
            if max_alignments is not None and stats["alignments_parsed"] >= max_alignments:
                break
            stats["alignments_parsed"] += 1
            sid1, sid2, cigar = parsed

            try:
                # extract_features is cached per (path, mtime, virt), so the ~1k structures
                # behind a pairfile are each parsed once however many alignments cite them.
                feat1, mask1, _ = extract_features(
                    str(resolve_pdb_path(pdb_dir, sid1)), virtual_center
                )
                feat2, mask2, _ = extract_features(
                    str(resolve_pdb_path(pdb_dir, sid2)), virtual_center
                )
                idx_pairs = parse_cigar(cigar)
            except (OSError, ValueError, KeyError):
                stats["alignments_failed"] += 1
                continue

            if idx_pairs.size == 0:
                continue
            idx_1, idx_2 = idx_pairs.T
            stats["pairs_before_validity"] += len(idx_1)

            # accumulate_counts skips CIGAR positions past the end of either sequence rather
            # than failing. Drop them first: filter_valid_pairs indexes the masks directly, so
            # an out-of-range position would raise IndexError and abort the whole probe on one
            # malformed alignment.
            in_range = (idx_1 < len(mask1)) & (idx_2 < len(mask2))
            stats["pairs_out_of_range"] += int((~in_range).sum())
            idx_1, idx_2 = idx_1[in_range], idx_2[in_range]
            if len(idx_1) == 0:
                continue

            # Equivalent to run_evaluate: positions invalid in either structure encode as the
            # invalid letter, which accumulate_counts then skips.
            idx_1, idx_2 = filter_valid_pairs(idx_1, idx_2, mask1, mask2)
            if len(idx_1) == 0:
                continue
            stats["pairs_after_validity"] += len(idx_1)
            stats["alignments_used"] += 1

            x_chunks.append(feat1[idx_1])
            y_chunks.append(feat2[idx_2])

    if not x_chunks:
        raise RuntimeError(f"No aligned descriptor pairs recovered from {pairfile}.")
    return (
        np.concatenate(x_chunks).astype(np.float32),
        np.concatenate(y_chunks).astype(np.float32),
        stats,
    )


def load_fit_sample(
    processed_dir: Path, n_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw a deterministic standardized sample of training descriptors.

    Codebooks are fit on training descriptors and only applied to the validation pairs, so
    the curve never fits on the data it scores. Memory-mapped so the 5.6M-row array is not
    materialized.

    Row order is deliberately left as ``rng.choice`` returned it rather than sorted. k-means
    seeding consumes points in array order, so sorting silently changes which points are
    chosen at initialization and lands on a different local optimum -- on a 5,000-row check,
    inertia 10767.64 sorted against 10941.25 unsorted. Matching ``kmeans_baseline`` here is
    what lets the calibration point reproduce the published baseline.

    Returns:
        A tuple of (standardized sample, scaler mean, scaler std).
    """
    descriptors = np.load(processed_dir / "train_x_raw.npy", mmap_mode="r")
    scaler = np.load(processed_dir / "scaler.npz")
    mean, std = scaler["mean"], scaler["std"]

    rng = np.random.default_rng(seed)
    total = descriptors.shape[0]
    if n_samples >= total:
        rows = np.arange(total)
    else:
        rows = rng.choice(total, size=n_samples, replace=False)
    sample = np.asarray(descriptors[rows], dtype=np.float32)
    return (sample - mean) / std, mean, std


def directed_counts(labels_x: np.ndarray, labels_y: np.ndarray, k: int) -> np.ndarray:
    """Unsymmetrized joint count matrix: one count per aligned pair.

    Each observation contributes exactly one cell, so this table *is* a multinomial sample of
    ``len(labels_x)`` independent draws and the ordinary Miller-Madow correction applies to it.
    """
    flat = np.bincount(labels_x.astype(np.int64) * k + labels_y.astype(np.int64), minlength=k * k)
    return flat.reshape(k, k)


def joint_counts(labels_x: np.ndarray, labels_y: np.ndarray, k: int) -> np.ndarray:
    """Symmetrized joint count matrix over one orientation of aligned assignments.

    Mirrors ``accumulate_counts``: each pair is recorded once, then the matrix is made
    symmetric. This is the evaluator-matching table, and the one whose plug-in MI is directly
    comparable to a reported ``mi``. Note that every observation now occupies two dependent
    cells, which is why analytic multinomial bias corrections do not apply to it.
    """
    counts = directed_counts(labels_x, labels_y, k)
    return counts + counts.T


def bootstrap_bias_bits(
    labels_x: np.ndarray,
    labels_y: np.ndarray,
    k: int,
    n_resamples: int,
    rng: np.random.Generator,
) -> float:
    """Estimate the plug-in bias of the *symmetrized* MI by resampling whole pairs.

    The symmetrized table writes each observation into two dependent cells, so the multinomial
    Miller-Madow formula is not valid on it -- and passing the one-orientation ``N`` does not
    repair either the independence assumption or the degrees of freedom. Resampling at the
    level of the independent unit (the aligned pair) does respect that structure: each
    replicate rebuilds the symmetrized joint exactly as the estimator does, so the spread of
    replicate MIs around the observed value estimates the bias of this specific statistic.

    Returns:
        The estimated upward bias in bits (mean resampled MI minus the observed MI), or 0.0
        when resampling is disabled.
    """
    if n_resamples <= 0:
        return 0.0
    observed = mutual_information_from_counts(joint_counts(labels_x, labels_y, k))
    n = len(labels_x)
    resampled = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        rows = rng.integers(0, n, size=n)
        resampled[i] = mutual_information_from_counts(
            joint_counts(labels_x[rows], labels_y[rows], k)
        )
    return float(resampled.mean() - observed)


def curve_point(
    labels_x: np.ndarray,
    labels_y: np.ndarray,
    k: int,
    estimator: str,
    n_fit_samples: int,
    rng: np.random.Generator,
    n_bootstrap: int = 0,
) -> CurvePoint:
    """Score one K: plug-in MI, two bias estimates, and the shuffled-y control.

    Three numbers, each valid for a different target, and they are not interchangeable:

    * ``mi_plugin_bits`` -- symmetrized joint. Evaluator-matching, directly comparable to a
      reported ``mi``, and uncorrected.
    * ``mi_miller_madow_bits`` -- the *unsymmetrized* joint. The analytic multinomial
      correction is only valid where each observation contributes one count, so it is applied
      there and nowhere else.
    * ``mi_bootstrap_corrected_bits`` -- symmetrized joint, minus a pair-level resampling
      estimate of its bias. This is the corrected form of the evaluator-matching number.

    The shuffle control is the empirical floor. It permutes ``y`` against ``x`` under the same
    fitted codebook, so dependence is destroyed while the marginals and bin occupancy are
    preserved; whatever it returns is this configuration's noise floor, which is more
    trustworthy than any analytic bias term.
    """
    counts = joint_counts(labels_x, labels_y, k)
    directed = directed_counts(labels_x, labels_y, k)
    n_pairs = len(labels_x)

    shuffled = joint_counts(labels_x, rng.permutation(labels_y), k)
    plugin = mutual_information_from_counts(counts)
    bias = bootstrap_bias_bits(labels_x, labels_y, k, n_bootstrap, rng)

    return CurvePoint(
        k=k,
        estimator=estimator,
        mi_plugin_bits=plugin,
        mi_plugin_directed_bits=mutual_information_from_counts(directed),
        # Valid here and only here: the directed table is a genuine multinomial sample.
        mi_miller_madow_bits=miller_madow_corrected_mi(directed, n_observations=n_pairs),
        mi_bootstrap_bias_bits=bias,
        mi_bootstrap_corrected_bits=plugin - bias,
        n_bootstrap=n_bootstrap,
        mi_shuffled_bits=mutual_information_from_counts(shuffled),
        occupied_x=int(np.count_nonzero(counts.sum(axis=1))),
        occupied_y=int(np.count_nonzero(counts.sum(axis=0))),
        occupied_joint=int(np.count_nonzero(counts)),
        n_pairs=n_pairs,
        n_fit_samples=n_fit_samples,
    )


def saturation_curve(
    fit_sample: np.ndarray,
    x_std: np.ndarray,
    y_std: np.ndarray,
    k_values: tuple[int, ...],
    seed: int,
    n_bootstrap: int = 0,
) -> list[CurvePoint]:
    """Fit one shared codebook per K and score the aligned pairs under it.

    A single codebook serves both sides. Two separately fitted codebooks would make the
    joint non-square and the marginals incomparable, and the alphabet being modelled is one
    shared symbol set.
    """
    points: list[CurvePoint] = []
    rng = np.random.default_rng(seed)
    for k in k_values:
        print(f"  K={k:<5d} fitting MiniBatchKMeans on {len(fit_sample):,} descriptors...")
        model = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init="auto", batch_size=4096)
        model.fit(fit_sample)
        labels_x = np.asarray(model.predict(x_std))
        labels_y = np.asarray(model.predict(y_std))
        point = curve_point(
            labels_x, labels_y, k, "minibatch_kmeans", len(fit_sample), rng, n_bootstrap
        )
        points.append(point)
        print(
            f"          plug-in {point.mi_plugin_bits:.4f}  "
            f"boot-corrected {point.mi_bootstrap_corrected_bits:.4f}  "
            f"shuffled {point.mi_shuffled_bits:.4f}  bins {point.occupied_joint:,}/{k * k:,}"
        )
    return points


def exact_kmeans_calibration(
    processed_dir: Path,
    x_std: np.ndarray,
    y_std: np.ndarray,
    n_samples: int,
    seed: int,
    n_bootstrap: int = 0,
    centroids_path: Path | None = None,
) -> CurvePoint:
    """Score K=20 with the *same* centroids the published raw-k-means baseline used.

    This point exists solely to confirm that the pair construction above reproduces the
    evaluator, so it has to be an exact reproduction or it proves nothing: a mismatch would
    otherwise be unattributable between the pairs and the clustering.

    Two things make it exact. It calls ``kmeans_baseline._fit_centroids`` rather than
    re-implementing the fit, and that function samples with ``rng.choice`` in the order
    returned. k-means seeding consumes points in array order, so any reordering -- sorting
    the indices for memory-mapped reads, most naturally -- selects different initial points
    and converges elsewhere (inertia 10767.64 sorted against 10941.25 unsorted on a 5,000-row
    check). Passing ``centroids_path`` skips fitting altogether and loads the baseline run's
    saved ``centroids.npy``, which is exact by construction.

    The MiniBatchKMeans K=20 curve point is a different estimator and is neither expected nor
    required to match this.
    """
    if centroids_path is not None and centroids_path.exists():
        print(f"  K=20    calibration: loading baseline centroids from {centroids_path}")
        centroids = np.load(centroids_path)
        n_fit = 0
    else:
        print(f"  K=20    calibration: exact KMeans via kmeans_baseline on {n_samples:,} rows...")
        centroids, _, _ = _fit_centroids(processed_dir, 20, n_samples, seed)
        n_fit = n_samples

    # Nearest-centroid assignment, matching RawDescriptorAlphabet.encode_states.
    labels_x = np.argmin(((x_std[:, None, :] - centroids[None, :, :]) ** 2).sum(-1), axis=1)
    labels_y = np.argmin(((y_std[:, None, :] - centroids[None, :, :]) ** 2).sum(-1), axis=1)
    rng = np.random.default_rng(seed)
    return curve_point(labels_x, labels_y, 20, "exact_kmeans", n_fit, rng, n_bootstrap)


def dimension_mi(
    fit_sample: np.ndarray, x_std: np.ndarray, y_std: np.ndarray, n_bins: int, seed: int
) -> list[DimensionPoint]:
    """Per-dimension aligned MI under shared training-derived quantile bins.

    Quantile edges come from the training sample, so both sides of every pair are binned
    identically and the joint stays square. If the aligned signal concentrates in one or two
    of the ten descriptors, that says concretely what a richer descriptor would need to add.
    """
    points: list[DimensionPoint] = []
    rng = np.random.default_rng(seed)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    for dim in range(fit_sample.shape[1]):
        edges = np.quantile(fit_sample[:, dim], quantiles)
        labels_x = np.digitize(x_std[:, dim], edges)
        labels_y = np.digitize(y_std[:, dim], edges)
        counts = joint_counts(labels_x, labels_y, n_bins)
        directed = directed_counts(labels_x, labels_y, n_bins)
        shuffled = joint_counts(labels_x, rng.permutation(labels_y), n_bins)
        points.append(
            DimensionPoint(
                dimension=dim,
                n_bins=n_bins,
                mi_plugin_bits=mutual_information_from_counts(counts),
                mi_plugin_directed_bits=mutual_information_from_counts(directed),
                # Analytic correction on the directed table only, where it is valid.
                mi_miller_madow_bits=miller_madow_corrected_mi(
                    directed, n_observations=len(labels_x)
                ),
                mi_shuffled_bits=mutual_information_from_counts(shuffled),
                n_pairs=len(labels_x),
            )
        )
    return points


def ksg_mutual_information(
    x: np.ndarray, y: np.ndarray, k: int, jitter: float = 1e-10, seed: int = 0
) -> float:
    """Kraskov-Stoegbauer-Grassberger MI estimator (variant 1), in bits.

    Continuous and binning-free, so its bias structure differs from the plug-in estimators
    above and it serves as an independent cross-check.

    Caveat that governs how much weight it carries: ``x`` and ``y`` are 10-D each, so the
    joint search space is 20-D. Nearest-neighbour distances concentrate badly above roughly
    six dimensions, and this estimator degrades there in a direction that is not predictable.
    Treat a KSG number here as weak corroboration at best, never as the tie-breaker.
    """
    n = len(x)
    if k >= n:
        raise ValueError(f"KSG needs k < N; got k={k} with N={n}.")

    # Ties break the estimator. Duplicated observations give a zero k-th-neighbour radius, the
    # strictly-inside neighbour counts then come back empty, and digamma(0) sends the result to
    # -inf (observed as an `inf` MI on a duplicate-heavy input). Kraskov's own remedy is a
    # deterministic jitter far below the data scale, which breaks exact ties without moving the
    # geometry the estimator depends on.
    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        scale = jitter * max(float(np.std(x)), float(np.std(y)), 1e-12)
        x = x + rng.normal(scale=scale, size=x.shape)
        y = y + rng.normal(scale=scale, size=y.shape)

    joint = np.hstack([x, y])
    # Chebyshev radius to the k-th neighbour in the joint space, excluding the point itself.
    joint_distances, _ = KDTree(joint).query(joint, k=k + 1, p=np.inf)
    radius = joint_distances[:, k]

    tree_x, tree_y = KDTree(x), KDTree(y)
    # Strictly-inside counts: shrink the radius so the k-th neighbour itself is excluded.
    n_x = np.array(
        [len(tree_x.query_ball_point(x[i], radius[i] - 1e-12, p=np.inf)) for i in range(n)]
    )
    n_y = np.array(
        [len(tree_y.query_ball_point(y[i], radius[i] - 1e-12, p=np.inf)) for i in range(n)]
    )

    # Any empty neighbourhood means ties survived the jitter; digamma(0) is -inf, so refuse to
    # return a number rather than reporting an infinite mutual information.
    if np.any(n_x < 1) or np.any(n_y < 1):
        raise ValueError(
            "KSG hit an empty neighbourhood, which means tied points survived jittering. "
            "Increase --ksg-jitter or use fewer duplicate observations."
        )

    # query_ball_point counts the point itself, matching the estimator's n_x + 1 convention.
    mi_nats = digamma(k) + digamma(n) - float(np.mean(digamma(n_x) + digamma(n_y)))
    result = mi_nats / math.log(2)
    if not np.isfinite(result):
        raise ValueError(f"KSG produced a non-finite estimate ({result}).")
    return max(0.0, result)


def run_ksg(
    x_std: np.ndarray,
    y_std: np.ndarray,
    sample_sizes: tuple[int, ...],
    neighbours: tuple[int, ...],
    seed: int,
    jitter: float = 1e-10,
) -> list[dict[str, float | int]]:
    """Run KSG across sample sizes and k, so drift with N is visible.

    A KSG estimate that keeps moving as N grows has not converged and should not be used to
    corroborate anything.
    """
    results: list[dict[str, float | int]] = []
    rng = np.random.default_rng(seed)
    for n in sample_sizes:
        if n > len(x_std):
            print(f"  KSG N={n:,} skipped: only {len(x_std):,} pairs available.")
            continue
        rows = rng.choice(len(x_std), size=n, replace=False)
        xs, ys = x_std[rows], y_std[rows]
        for k in neighbours:
            # A tie-induced failure on one (N, k) must not lose the rest of the grid.
            try:
                value = ksg_mutual_information(xs, ys, k, jitter, seed)
                shuffled = ksg_mutual_information(xs, rng.permutation(ys), k, jitter, seed)
            except ValueError as exc:
                print(f"  KSG N={n:,} k={k}: skipped ({exc})")
                continue
            results.append({"n_samples": n, "k": k, "mi_bits": value, "mi_shuffled_bits": shuffled})
            print(f"  KSG N={n:,} k={k}: {value:.4f} bits   shuffled {shuffled:.4f}")
    return results


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a list of uniform dict rows to CSV."""
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_summary(points: list[CurvePoint], calibration: CurvePoint) -> None:
    """Print the curve and the reading guide for it."""
    print("\nDescriptor-information saturation curve (discretized LOWER BOUNDS, not a ceiling)")
    header = (
        f"{'K':>6s} {'estimator':>17s} {'plug-in':>9s} {'boot-corr':>10s} {'directed':>9s} "
        f"{'MM(dir)':>9s} {'shuffled':>9s} {'occ_joint':>10s} {'pairs':>10s}"
    )
    print(header)
    print("-" * len(header))
    for point in [*points, calibration]:
        print(
            f"{point.k:6d} {point.estimator:>17s} {point.mi_plugin_bits:9.4f} "
            f"{point.mi_bootstrap_corrected_bits:10.4f} {point.mi_plugin_directed_bits:9.4f} "
            f"{point.mi_miller_madow_bits:9.4f} {point.mi_shuffled_bits:9.4f} "
            f"{point.occupied_joint:10,d} {point.n_pairs:10,d}"
        )
    print(
        "\nplug-in = symmetrized, evaluator-matching, uncorrected. boot-corr = the same "
        "statistic\nminus a pair-level resampling bias estimate (0 bias when --bootstrap is 0). "
        "directed and\nMM(dir) are the unsymmetrized joint and its analytic Miller-Madow "
        "correction, which is only\nvalid on that table. MM is not signed: on a concentrated "
        "joint it raises the estimate."
    )

    print(
        "\nRead by shape, not by any single value:\n"
        "  Plateaus near the learned model by K~100  -> descriptors bind; prioritize richer\n"
        "                                               descriptors over objective work.\n"
        "  Still growing substantially through K~300 -> information exists that the current\n"
        "                                               objective is not extracting.\n"
        "  Still climbing steeply at K=1000          -> information exists, but the 20-state\n"
        "                                               budget is likely the binding\n"
        "                                               compression constraint.\n"
        "  Large shuffled MI                         -> fix the estimator before reading\n"
        "                                               anything else.\n"
        "A low curve is suggestive, not conclusive. A high curve is the stronger evidence."
    )


def main() -> None:
    """Build the saturation curve, the per-dimension breakdown, and optional KSG checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/probes/descriptor_mi", help="Output directory.")
    parser.add_argument(
        "--processed-dir",
        default="data/processed/scop_ca5_v1",
        help="Processed dataset supplying the training scaler and codebook-fitting sample.",
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
        "--k-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help="Cluster counts along the saturation curve.",
    )
    parser.add_argument(
        "--fit-samples", type=int, default=500_000, help="Training descriptors to fit codebooks on."
    )
    parser.add_argument(
        "--dimension-bins", type=int, default=32, help="Quantile bins for per-dimension MI."
    )
    parser.add_argument(
        "--max-alignments", type=int, default=None, help="Cap parsed alignments (debugging)."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling and clustering.")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=25,
        help="Pair-level resamples used to estimate the bias of the symmetrized MI. 0 disables.",
    )
    parser.add_argument(
        "--baseline-centroids",
        default="runs/ablation/kmeans_raw/centroids.npy",
        help="Saved raw-k-means centroids. Used for the calibration point when present, which "
        "makes it an exact reproduction; otherwise the baseline's own fit is re-run.",
    )
    parser.add_argument(
        "--ksg",
        action="store_true",
        help="Run the KSG continuous cross-check. Opt-in: 20-D neighbour search is expensive "
        "and unreliable at this dimensionality.",
    )
    parser.add_argument(
        "--ksg-sample-sizes",
        type=int,
        nargs="+",
        default=[10_000, 50_000],
        help="KSG sample sizes. Add 200000 explicitly; it is slow.",
    )
    parser.add_argument(
        "--ksg-neighbours", type=int, nargs="+", default=[3, 5, 10], help="KSG k values."
    )
    parser.add_argument(
        "--ksg-jitter",
        type=float,
        default=1e-10,
        help="Relative deterministic jitter breaking exact ties, which would otherwise send "
        "the KSG estimate to infinity.",
    )
    parser.add_argument(
        "--virt",
        type=float,
        nargs=3,
        default=None,
        help="Virtual center override; otherwise read from the dataset manifest.",
    )
    args = parser.parse_args()

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
    print(
        f"  {pair_stats['alignments_used']:,} alignments -> {len(x_raw):,} aligned pairs "
        f"(one orientation); {pair_stats['alignments_failed']:,} failed."
    )

    fit_sample, mean, std = load_fit_sample(processed_dir, args.fit_samples, args.seed)
    x_std = (x_raw - mean) / std
    y_std = (y_raw - mean) / std

    print("\nFitting shared codebooks...")
    points = saturation_curve(
        fit_sample, x_std, y_std, tuple(args.k_values), args.seed, args.bootstrap
    )
    calibration = exact_kmeans_calibration(
        processed_dir,
        x_std,
        y_std,
        args.fit_samples,
        args.seed,
        args.bootstrap,
        Path(args.baseline_centroids) if args.baseline_centroids else None,
    )

    print(f"\nPer-dimension MI ({args.dimension_bins} shared quantile bins)...")
    dimension_points = dimension_mi(fit_sample, x_std, y_std, args.dimension_bins, args.seed)
    for point in dimension_points:
        print(
            f"  dim {point.dimension:2d}: plug-in {point.mi_plugin_bits:.4f}  "
            f"MM {point.mi_miller_madow_bits:.4f}  shuffled {point.mi_shuffled_bits:.4f}"
        )

    ksg_results: list[dict[str, float | int]] = []
    if args.ksg:
        print("\nKSG cross-check (weak at 20-D; corroboration only)...")
        ksg_results = run_ksg(
            x_std,
            y_std,
            tuple(args.ksg_sample_sizes),
            tuple(args.ksg_neighbours),
            args.seed,
            args.ksg_jitter,
        )

    _print_summary(points, calibration)

    payload = {
        "interpretation": (
            "All binned values are discretized LOWER BOUNDS on the continuous aligned mutual "
            "information of the descriptors, not estimates of a hard ceiling. Binning can only "
            "destroy information (data-processing inequality), while finite-sample plug-in bias "
            "inflates the estimate of that reduced quantity; the net sign is undetermined. Read "
            "the curve by shape and by agreement across estimators."
        ),
        "inputs": {
            "processed_dir": str(processed_dir),
            "pdb_dir": args.pdb_dir,
            "pairfile": args.pairfile,
            "virtual_center": list(virtual_center),
            "seed": args.seed,
            "fit_samples": len(fit_sample),
        },
        "pair_stats": pair_stats,
        "curve": [asdict(point) for point in points],
        "calibration_exact_kmeans_k20": asdict(calibration),
        "dimension_mi": [asdict(point) for point in dimension_points],
        "ksg": ksg_results,
        "ksg_caveat": (
            "x and y are 10-D each, so KSG searches a 20-D joint space where neighbour "
            "distances concentrate and the estimator degrades unpredictably. Weak corroboration "
            "only; never the tie-breaker."
        ),
    }
    with open(out_dir / "descriptor_mi_curve.json", "w") as f:
        json.dump(payload, f, indent=2)
    _write_csv(
        out_dir / "descriptor_mi_curve.csv",
        [asdict(point) for point in [*points, calibration]],
    )
    _write_csv(
        out_dir / "descriptor_dimension_mi.csv", [asdict(point) for point in dimension_points]
    )
    print(f"\nWrote descriptor_mi_curve.json/.csv and descriptor_dimension_mi.csv to {out_dir}")
    print(
        "Compare the exact-KMeans K=20 calibration point against the raw-k-means baseline "
        "report\n(runs/ablation/kmeans_raw/eval/evaluation_report.json, mi 1.2098) to confirm "
        "the pair\nconstruction matches evaluation."
    )


if __name__ == "__main__":
    main()
