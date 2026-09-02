# ruff: noqa: E402
"""Fast unit tests for the dataset split script helpers."""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.make_splits import select_validation_groups
from scripts.split_folds import partition_folds


def test_select_validation_groups_preserves_floor_count_and_seed() -> None:
    groups = [f"group-{index}" for index in range(10)]

    selected = select_validation_groups(groups, val_split=0.25, seed=7)

    assert len(selected) == 2
    assert selected <= set(groups)
    assert selected == select_validation_groups(groups, val_split=0.25, seed=7)
    assert select_validation_groups(groups, val_split=0.0, seed=7) == set()
    assert select_validation_groups(groups, val_split=1.0, seed=7) == set(groups)


def test_select_validation_groups_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        select_validation_groups(["group"], val_split=1.1, seed=7)


def test_partition_folds_is_complete_balanced_and_deterministic() -> None:
    folds = [f"fold-{index}" for index in range(7)]

    splits = partition_folds(folds, k=3, seed=11)

    assert sorted(map(len, splits)) == [2, 2, 3]
    assert sorted(fold for split in splits for fold in split) == folds
    assert splits == partition_folds(folds, k=3, seed=11)


@pytest.mark.parametrize(("k", "message"), [(1, ">= 2"), (4, "exceeds")])
def test_partition_folds_rejects_invalid_k(k: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        partition_folds(["a", "b", "c"], k=k, seed=0)
