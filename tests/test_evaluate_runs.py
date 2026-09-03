# ruff: noqa: E402
"""Tests for how the run-comparison table surfaces chain-contribution metrics.

The point of these columns is to expose an arm that raises raw ``mi`` only by emitting longer
correlated state runs. That only works if the table reads the evaluator's own numbers when
they are present, and still tabulates runs scored before those numbers were emitted -- the
existing helios reports are all in the second category.
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.evaluate_runs import (
    RunResult,
    _chain_autocorrelation,
    _chain_fraction,
    _read_eval_report,
)

_MI_PREV_WEIGHT = 1.0 - 0.057


def _result(**overrides: object) -> RunResult:
    """A scored run, with alphabet metrics overridable per test."""
    base = RunResult(
        name="arm",
        quantizer="vq",
        n_states=20,
        z_dim=4,
        best_val_loss=0.3,
        best_epoch=1,
        epochs=2,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_chain_metrics_prefer_the_reported_values() -> None:
    """When the evaluator emitted the numbers, the table must use them verbatim."""
    result = _result(mi=1.5, mi_tot=0.9, mi_prev=0.6363, chain_fraction=0.4)
    assert _chain_autocorrelation(result) == pytest.approx(0.6363)
    assert _chain_fraction(result) == pytest.approx(0.4)


def test_chain_metrics_fall_back_for_reports_without_them() -> None:
    """Runs evaluated before mi_prev was emitted must still tabulate, not blank out."""
    result = _result(mi=1.5, mi_tot=0.9)
    assert _chain_autocorrelation(result) == pytest.approx((1.5 - 0.9) / _MI_PREV_WEIGHT)
    assert _chain_fraction(result) == pytest.approx((1.5 - 0.9) / 1.5)


def test_chain_metrics_are_none_without_mi() -> None:
    """A skipped run has nothing to derive from and must not fabricate a value."""
    result = _result()
    assert _chain_autocorrelation(result) is None
    assert _chain_fraction(result) is None


def test_derived_chain_fraction_guards_zero_mi() -> None:
    """A zero-MI alphabet must not divide by zero in the fallback path."""
    assert _chain_fraction(_result(mi=0.0, mi_tot=0.0)) is None


def test_read_eval_report_picks_up_chain_metrics(tmp_path: Path) -> None:
    """The reader must carry mi_prev and chain_fraction off the report onto the row."""
    import json

    run_dir = tmp_path / "arm"
    (run_dir / "eval").mkdir(parents=True)
    with open(run_dir / "eval" / "evaluation_report.json", "w") as f:
        json.dump(
            {
                "mi": 1.5,
                "mi_tot": 0.9,
                "mi_prev": 0.6363,
                "chain_fraction": 0.4,
                "normalized_entropy": 0.9,
                "dead_state_fraction": 0.0,
                "n_sequences": 10,
                "failure_rate": 0.0,
                "state_usage": [5, 5],
            },
            f,
        )

    result = _read_eval_report(run_dir / "eval" / "evaluation_report.json", _result())
    assert result.mi_prev == pytest.approx(0.6363)
    assert result.chain_fraction == pytest.approx(0.4)
