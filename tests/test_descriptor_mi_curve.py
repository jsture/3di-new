# ruff: noqa: E402
"""Tests for the descriptor-information probe and its shared MI estimators.

Deliberately small and synthetic: the probe's real inputs are millions of rows and K=1000
codebooks, none of which belong in a test suite. What is worth pinning is the estimator
behaviour the interpretation rests on -- that independence reads as zero, that a known
coupling reads as its known value, that the bias correction moves the right way, and that
the shuffle control actually destroys dependence.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.descriptor_mi_curve import (
    bootstrap_bias_bits,
    curve_point,
    dimension_mi,
    directed_counts,
    joint_counts,
    ksg_mutual_information,
    load_fit_sample,
    saturation_curve,
)
from tdi.v2.submat import miller_madow_corrected_mi, mutual_information_from_counts


def test_independent_variables_give_near_zero_mi() -> None:
    """Independent assignments carry no shared information."""
    rng = np.random.default_rng(0)
    labels_x = rng.integers(0, 4, size=20_000)
    labels_y = rng.integers(0, 4, size=20_000)
    mi = mutual_information_from_counts(joint_counts(labels_x, labels_y, 4))
    assert mi == pytest.approx(0.0, abs=0.01)


def test_perfectly_coupled_variables_give_log2_k_bits() -> None:
    """A deterministic 1:1 coupling over K balanced states carries exactly log2(K) bits."""
    labels = np.repeat(np.arange(4), 500)
    mi = mutual_information_from_counts(joint_counts(labels, labels, 4))
    assert mi == pytest.approx(np.log2(4), abs=1e-9)


def test_constant_assignment_gives_zero_mi() -> None:
    """A collapsed single-state alphabet carries no information."""
    labels = np.zeros(1000, dtype=np.int64)
    assert mutual_information_from_counts(joint_counts(labels, labels, 4)) == pytest.approx(0.0)


def test_miller_madow_lowers_the_positively_biased_plugin() -> None:
    """With many bins and few samples the correction must reduce the plug-in estimate.

    Computed on the directed table: each observation contributes one count there, which is
    what the multinomial correction assumes.
    """
    rng = np.random.default_rng(1)
    # Independent data over 30 bins from 400 pairs: the plug-in estimate is badly inflated.
    labels_x = rng.integers(0, 30, size=400)
    labels_y = rng.integers(0, 30, size=400)
    directed = directed_counts(labels_x, labels_y, 30)

    plugin = mutual_information_from_counts(directed)
    corrected = miller_madow_corrected_mi(directed, n_observations=400)
    assert corrected < plugin
    # The correction should move the inflated estimate toward the truth, which is zero.
    assert abs(corrected) < abs(plugin)


def test_miller_madow_scales_the_correction_with_observation_count() -> None:
    """A smaller N means a larger correction, since the bias term divides by N."""
    rng = np.random.default_rng(2)
    labels_x = rng.integers(0, 20, size=500)
    labels_y = rng.integers(0, 20, size=500)
    directed = directed_counts(labels_x, labels_y, 20)
    plugin = mutual_information_from_counts(directed)

    small_n = miller_madow_corrected_mi(directed, n_observations=500)
    large_n = miller_madow_corrected_mi(directed, n_observations=5000)
    assert abs(plugin - small_n) > abs(plugin - large_n)
    # Default with no explicit count uses the matrix total, which equals N for a directed table.
    assert miller_madow_corrected_mi(directed) == pytest.approx(
        miller_madow_corrected_mi(directed, n_observations=int(directed.sum()))
    )


def test_empty_counts_are_zero_not_nan() -> None:
    """An empty joint must not divide by zero."""
    empty = np.zeros((4, 4), dtype=np.int64)
    assert mutual_information_from_counts(empty) == 0.0
    assert miller_madow_corrected_mi(empty) == 0.0


def test_joint_counts_are_square_and_symmetric() -> None:
    """A shared codebook must produce a square, symmetrized joint."""
    labels_x = np.array([0, 1, 2, 1])
    labels_y = np.array([1, 1, 0, 2])
    counts = joint_counts(labels_x, labels_y, 3)
    assert counts.shape == (3, 3)
    assert np.array_equal(counts, counts.T)
    # Symmetrization records each pair twice, matching accumulate_counts.
    assert counts.sum() == 2 * len(labels_x)


def test_shuffling_destroys_dependence() -> None:
    """The shuffle control must collapse a strong coupling to near zero."""
    rng = np.random.default_rng(3)
    labels = np.repeat(np.arange(5), 2000)
    point = curve_point(labels, labels, 5, "test", n_fit_samples=0, rng=rng)
    assert point.mi_plugin_bits == pytest.approx(np.log2(5), abs=1e-9)
    assert point.mi_shuffled_bits == pytest.approx(0.0, abs=0.01)


def test_curve_point_records_occupancy_and_pair_count() -> None:
    """Occupancy diagnostics must reflect the states actually used, not the codebook size."""
    rng = np.random.default_rng(4)
    labels = np.repeat(np.arange(3), 100)
    point = curve_point(labels, labels, 10, "test", n_fit_samples=42, rng=rng)
    assert point.occupied_x == 3
    assert point.occupied_y == 3
    assert point.occupied_joint == 3
    assert point.n_pairs == 300
    assert point.n_fit_samples == 42


def test_saturation_curve_is_deterministic_under_a_fixed_seed() -> None:
    """Two runs at the same seed must agree exactly."""
    rng = np.random.default_rng(5)
    fit_sample = rng.normal(size=(600, 3))
    shared = rng.normal(size=(400, 3))
    x_std = shared + 0.05 * rng.normal(size=(400, 3))
    y_std = shared + 0.05 * rng.normal(size=(400, 3))

    first = saturation_curve(fit_sample, x_std, y_std, (4, 8), seed=7)
    second = saturation_curve(fit_sample, x_std, y_std, (4, 8), seed=7)
    assert [p.mi_plugin_bits for p in first] == [p.mi_plugin_bits for p in second]
    assert [p.mi_shuffled_bits for p in first] == [p.mi_shuffled_bits for p in second]


def test_saturation_curve_finds_coupling_and_shuffle_finds_none() -> None:
    """Strongly coupled latent structure must register above its own shuffle control."""
    rng = np.random.default_rng(6)
    fit_sample = rng.normal(size=(800, 2)) * 3
    shared = rng.normal(size=(600, 2)) * 3
    x_std = shared + 0.01 * rng.normal(size=(600, 2))
    y_std = shared + 0.01 * rng.normal(size=(600, 2))

    (point,) = saturation_curve(fit_sample, x_std, y_std, (6,), seed=1)
    assert point.mi_plugin_bits > 1.0
    assert point.mi_shuffled_bits < 0.2


def test_dimension_mi_reports_one_row_per_descriptor() -> None:
    """Per-dimension output must cover every input column with its own control."""
    rng = np.random.default_rng(8)
    fit_sample = rng.normal(size=(500, 3))
    shared = rng.normal(size=(400, 3))
    # Only dimension 0 is coupled across the pair; the others are independent noise.
    x_std = np.column_stack([shared[:, 0], rng.normal(size=400), rng.normal(size=400)])
    y_std = np.column_stack([shared[:, 0], rng.normal(size=400), rng.normal(size=400)])
    x_std[:, 0] = shared[:, 0]
    y_std[:, 0] = shared[:, 0]

    points = dimension_mi(fit_sample, x_std, y_std, n_bins=4, seed=2)
    assert [p.dimension for p in points] == [0, 1, 2]
    assert all(p.n_bins == 4 for p in points)
    # The coupled dimension must stand out against the independent ones.
    assert points[0].mi_plugin_bits > points[1].mi_plugin_bits
    assert points[0].mi_plugin_bits > points[2].mi_plugin_bits


# ---------------------------------------------------------------------------
# Regressions for review findings
# ---------------------------------------------------------------------------


def test_directed_counts_hold_one_count_per_observation() -> None:
    """The unsymmetrized table is the one the multinomial bias correction is valid on."""
    labels_x = np.array([0, 1, 2, 1])
    labels_y = np.array([1, 1, 0, 2])
    directed = directed_counts(labels_x, labels_y, 3)

    assert directed.sum() == len(labels_x)
    assert directed[0, 1] == 1
    assert directed[1, 0] == 0
    # The symmetrized table is its own transpose sum, so it double-counts each observation.
    assert np.array_equal(joint_counts(labels_x, labels_y, 3), directed + directed.T)


def test_miller_madow_is_applied_to_the_directed_table_not_the_symmetrized_one() -> None:
    """Regression: the analytic correction must not be computed on symmetrized counts.

    Symmetrization writes each observation into two dependent cells, breaking the multinomial
    assumption; passing the one-orientation N does not repair it. The reported value must
    therefore match the correction of the directed table.
    """
    rng = np.random.default_rng(0)
    labels_x = rng.integers(0, 6, size=300)
    labels_y = rng.integers(0, 6, size=300)
    point = curve_point(labels_x, labels_y, 6, "test", n_fit_samples=0, rng=rng)

    directed = directed_counts(labels_x, labels_y, 6)
    assert point.mi_miller_madow_bits == pytest.approx(
        miller_madow_corrected_mi(directed, n_observations=300)
    )
    assert point.mi_plugin_directed_bits == pytest.approx(mutual_information_from_counts(directed))
    # The evaluator-matching number stays on the symmetrized table.
    assert point.mi_plugin_bits == pytest.approx(
        mutual_information_from_counts(joint_counts(labels_x, labels_y, 6))
    )


def test_bootstrap_bias_is_positive_for_an_independent_joint() -> None:
    """Resampling must detect the upward plug-in bias where the truth is known to be zero."""
    rng = np.random.default_rng(1)
    labels_x = rng.integers(0, 10, size=300)
    labels_y = rng.integers(0, 10, size=300)

    bias = bootstrap_bias_bits(labels_x, labels_y, 10, n_resamples=40, rng=rng)
    assert bias > 0.0
    # Disabling resampling must leave the plug-in value untouched.
    assert bootstrap_bias_bits(labels_x, labels_y, 10, n_resamples=0, rng=rng) == 0.0


def test_curve_point_bootstrap_correction_subtracts_the_bias() -> None:
    """The corrected column must be exactly plug-in minus the reported bias."""
    rng = np.random.default_rng(2)
    labels_x = rng.integers(0, 5, size=200)
    labels_y = rng.integers(0, 5, size=200)
    point = curve_point(labels_x, labels_y, 5, "test", 0, rng, n_bootstrap=20)

    assert point.n_bootstrap == 20
    assert point.mi_bootstrap_corrected_bits == pytest.approx(
        point.mi_plugin_bits - point.mi_bootstrap_bias_bits
    )


def test_load_fit_sample_preserves_choice_order(tmp_path: Path) -> None:
    """Regression: sorting the sampled rows changes which points k-means seeds from.

    kmeans_baseline samples with rng.choice and does not sort, so sorting here silently fits a
    different codebook and the calibration point could no longer reproduce the baseline.
    """
    rows = np.arange(200, dtype=np.float32).reshape(100, 2)
    np.save(tmp_path / "train_x_raw.npy", rows)
    np.savez(
        tmp_path / "scaler.npz",
        mean=np.zeros(2, dtype=np.float32),
        std=np.ones(2, dtype=np.float32),
    )

    sample, _, _ = load_fit_sample(tmp_path, n_samples=20, seed=0)
    expected_rows = np.random.default_rng(0).choice(100, size=20, replace=False)
    assert np.array_equal(sample, rows[expected_rows])
    # The unsorted order is the point: a sorted sample would be a different fit.
    assert not np.array_equal(sample, rows[np.sort(expected_rows)])


def test_ksg_rejects_ties_instead_of_returning_infinity() -> None:
    """Regression: duplicated points previously produced digamma(0) and an infinite MI."""
    rng = np.random.default_rng(3)
    x = np.repeat(rng.normal(size=(20, 2)), 5, axis=0)
    y = np.repeat(rng.normal(size=(20, 2)), 5, axis=0)

    # Jitter breaks the exact ties and keeps the estimate finite.
    assert np.isfinite(ksg_mutual_information(x, y, k=3))
    # With jitter disabled the tie is fatal, and it must raise rather than report inf.
    with pytest.raises(ValueError, match="empty neighbourhood"):
        ksg_mutual_information(x, y, k=3, jitter=0.0)


def test_ksg_rejects_k_at_or_above_sample_size() -> None:
    """k >= N has no k-th neighbour and must be refused, not silently computed."""
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="k < N"):
        ksg_mutual_information(rng.normal(size=(4, 2)), rng.normal(size=(4, 2)), k=10)


def test_ksg_recovers_a_known_coupling() -> None:
    """A strongly coupled continuous pair must read well above its shuffled control."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=(600, 1))
    y = x + 0.1 * rng.normal(size=(600, 1))

    coupled = ksg_mutual_information(x, y, k=3)
    independent = ksg_mutual_information(x, rng.permutation(y), k=3)
    assert coupled > 1.0
    assert independent < 0.2


def test_out_of_range_cigar_positions_are_dropped_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a CIGAR citing a position past the end of a structure must not abort.

    accumulate_counts skips such positions, so the probe has to as well. Before the fix,
    filter_valid_pairs indexed the mask directly and one malformed alignment raised
    IndexError, killing the whole run.
    """
    import scripts.descriptor_mi_curve as probe

    # Three valid residues, but the alignment below cites five positions.
    features = np.arange(30, dtype=np.float32).reshape(3, 10)
    mask = np.ones(3, dtype=bool)

    monkeypatch.setattr(probe, "resolve_pdb_path", lambda pdb_dir, sid: Path(sid))
    monkeypatch.setattr(
        probe, "extract_features", lambda path, virt: (features, mask, np.zeros((3, 3)))
    )

    pairfile = tmp_path / "pairs.out"
    pairfile.write_text("sid1 sid2 5P\n")

    x, y, stats = probe.evaluation_equivalent_pairs(str(tmp_path), str(pairfile), (270.0, 0.0, 2.0))
    assert len(x) == 3
    assert len(y) == 3
    assert stats["pairs_before_validity"] == 5
    assert stats["pairs_out_of_range"] == 2
    assert stats["pairs_after_validity"] == 3
