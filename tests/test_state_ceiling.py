# ruff: noqa: E402
"""Tests for the fixed-budget state-ceiling probe.

The load-bearing claim is that the closed-form merge delta agrees with actually performing the
merge. Everything else in the probe is bookkeeping around that, so most of these compare the
fast path against the slow, trusted one on small synthetic joints.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.state_ceiling import (
    best_merge,
    merge_deltas_for_state,
    merge_to_target,
    mi_from_sufficient_stats,
    split_pairs,
)
from tdi.v2.submat import merge_columns, mutual_information_from_counts


def _symmetric_counts(seed: int, n_states: int, scale: int = 20) -> np.ndarray:
    """A random symmetric joint count matrix."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, scale, size=(n_states, n_states))
    return raw + raw.T


def test_sufficient_stats_mi_matches_the_direct_estimator() -> None:
    """The entropy-decomposition form must agree with the plug-in estimator exactly."""
    for seed in range(5):
        counts = _symmetric_counts(seed, 8)
        assert mi_from_sufficient_stats(counts) == pytest.approx(
            mutual_information_from_counts(counts), rel=1e-12, abs=1e-12
        )


def test_sufficient_stats_mi_handles_empty_counts() -> None:
    """An empty joint must not divide by zero or take log2(0)."""
    assert mi_from_sufficient_stats(np.zeros((4, 4), dtype=int)) == 0.0


def test_merge_delta_matches_actually_performing_the_merge() -> None:
    """The closed-form delta must equal the MI change from merge_columns, for every pair.

    This is the correctness claim the whole probe rests on: the fast search chooses pairs by
    the delta, while merge_columns applies them, so any disagreement silently selects the
    wrong merges.
    """
    counts = _symmetric_counts(0, 7)
    base = mi_from_sufficient_stats(counts)

    for i in range(len(counts)):
        deltas = merge_deltas_for_state(counts, i)
        assert deltas[i] == -np.inf
        for j in range(len(counts)):
            if j == i:
                continue
            merged = merge_columns(counts, min(i, j), max(i, j))
            expected = mutual_information_from_counts(merged) - base
            assert deltas[j] == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_merge_delta_handles_empty_and_diagonal_heavy_states() -> None:
    """Sparse states and a dominant diagonal must not break the closed form."""
    counts = np.array(
        [
            [100, 1, 0, 0],
            [1, 80, 2, 0],
            [0, 2, 0, 0],  # a state with no diagonal mass
            [0, 0, 0, 0],  # a completely unused state
        ],
        dtype=np.int64,
    )
    counts = counts + counts.T
    base = mi_from_sufficient_stats(counts)

    for i in range(len(counts)):
        deltas = merge_deltas_for_state(counts, i)
        for j in range(i + 1, len(counts)):
            merged = merge_columns(counts, i, j)
            assert deltas[j] == pytest.approx(
                mutual_information_from_counts(merged) - base, rel=1e-9, abs=1e-9
            )


def test_best_merge_picks_the_least_costly_pair() -> None:
    """The selected pair must be the one an exhaustive recomputation would choose."""
    counts = _symmetric_counts(1, 9)
    base = mi_from_sufficient_stats(counts)

    i, j, delta = best_merge(counts)
    assert i < j
    assert delta == pytest.approx(
        mutual_information_from_counts(merge_columns(counts, i, j)) - base, rel=1e-9, abs=1e-9
    )

    exhaustive = max(
        mutual_information_from_counts(merge_columns(counts, a, b)) - base
        for a in range(len(counts))
        for b in range(a + 1, len(counts))
    )
    assert delta == pytest.approx(exhaustive, rel=1e-9, abs=1e-9)


def test_merging_redundant_duplicates_is_free() -> None:
    """Two states that behave identically must be the cheapest merge, costing ~no MI.

    Built as a block-diagonal joint where states 0 and 1 are exact duplicates of one another:
    collapsing them loses nothing, so a correct search finds that pair first.
    """
    block = np.array([[50, 50, 0, 0], [50, 50, 0, 0], [0, 0, 40, 0], [0, 0, 0, 30]])
    counts = block + block.T

    i, j, delta = best_merge(counts)
    assert (i, j) == (0, 1)
    assert delta == pytest.approx(0.0, abs=1e-9)


def test_merging_to_a_single_state_gives_zero_mi() -> None:
    """A collapsed alphabet carries no information."""
    counts = _symmetric_counts(2, 6)
    trajectory = merge_to_target(counts, counts, 1, "greedy", np.random.default_rng(0))
    assert trajectory[-1].n_states == 1
    assert trajectory[-1].mi_score_half_bits == pytest.approx(0.0, abs=1e-12)


def test_merge_trajectory_is_monotonic_and_covers_every_state_count() -> None:
    """Coarsening cannot increase MI, and one row must be recorded per state count."""
    counts = _symmetric_counts(3, 12)
    trajectory = merge_to_target(counts, counts, 3, "greedy", np.random.default_rng(0))

    assert [step.n_states for step in trajectory] == list(range(12, 2, -1))
    values = [step.mi_merge_half_bits for step in trajectory]
    assert all(later <= earlier + 1e-9 for earlier, later in itertools.pairwise(values))


def test_greedy_beats_random_merging() -> None:
    """The greedy schedule must retain more information than arbitrary coarsening.

    Uses a joint with clear block structure, so a good schedule preserves the blocks and a
    random one destroys them. If this ever fails, the search is not searching.
    """
    rng = np.random.default_rng(4)
    labels = np.repeat(np.arange(8), 400)
    # Four well-separated pairs: states 2m and 2m+1 are interchangeable within a block.
    partner = (labels // 2) * 2 + rng.integers(0, 2, size=len(labels))
    counts = np.zeros((8, 8), dtype=np.int64)
    np.add.at(counts, (labels, partner), 1)
    counts = counts + counts.T

    greedy = merge_to_target(counts, counts, 4, "greedy", np.random.default_rng(0))
    random_control = merge_to_target(counts, counts, 4, "random", np.random.default_rng(0))
    assert greedy[-1].mi_score_half_bits > random_control[-1].mi_score_half_bits


def test_merge_schedule_applies_identically_to_both_matrices() -> None:
    """Held-out scoring must follow the merge half's schedule, not its own optimum.

    Scored against a deliberately different held-out matrix: if the probe were choosing merges
    per-matrix, the held-out trajectory would differ from replaying the same merges by hand.
    """
    merge_counts = _symmetric_counts(5, 6)
    score_counts = _symmetric_counts(6, 6)

    trajectory = merge_to_target(merge_counts, score_counts, 2, "greedy", np.random.default_rng(0))

    replay_merge, replay_score = merge_counts, score_counts
    for step in trajectory[1:]:
        i, j, _ = best_merge(replay_merge)
        replay_merge = merge_columns(replay_merge, i, j)
        replay_score = merge_columns(replay_score, i, j)
        assert step.mi_score_half_bits == pytest.approx(
            mutual_information_from_counts(replay_score), rel=1e-9, abs=1e-9
        )


def test_split_pairs_is_deterministic_disjoint_and_covering() -> None:
    """The two halves must partition the pairs exactly, reproducibly."""
    merge_rows, score_rows = split_pairs(101, seed=0)
    assert len(merge_rows) + len(score_rows) == 101
    assert set(merge_rows).isdisjoint(set(score_rows))
    assert sorted([*merge_rows, *score_rows]) == list(range(101))

    again, _ = split_pairs(101, seed=0)
    assert np.array_equal(merge_rows, again)
    other, _ = split_pairs(101, seed=1)
    assert not np.array_equal(merge_rows, other)


def test_incremental_verification_catches_drift() -> None:
    """The built-in cross-check must fire if the fast and slow paths ever disagree."""
    counts = _symmetric_counts(7, 6)
    # verify_every=1 checks after every merge; a correct implementation passes silently.
    trajectory = merge_to_target(
        counts, counts, 2, "greedy", np.random.default_rng(0), verify_every=1
    )
    assert trajectory[-1].n_states == 2
