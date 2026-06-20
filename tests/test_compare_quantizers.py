# ruff: noqa: E402
"""Tests for the standalone quantizer-comparison driver and its isolation from core."""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.compare_quantizers import build_comparison


def _make_run(run_dir: Path, val_loss: float, entropy: float, dead: float) -> None:
    """Write a minimal finished-run layout (train_log.csv + eval report)."""
    (run_dir / "eval").mkdir(parents=True)
    (run_dir / "train_log.csv").write_text(
        "epoch,train_loss,val_loss,perplexity,dead_states\n0,1.0,2.0,5.0,1\n"
        f"1,0.5,{val_loss},6.0,0\n"
    )
    report = {
        "mi": 1.23,
        "mi_tot": 2.34,
        "total_counts": 100,
        "n_letters": 20,
        "dead_state_fraction": dead,
        "normalized_entropy": entropy,
    }
    (run_dir / "eval" / "evaluation_report.json").write_text(json.dumps(report))


def test_build_comparison_tabulates_two_runs(tmp_path: Path) -> None:
    """The driver reads both runs' logs/reports into a side-by-side table."""
    vq = tmp_path / "ema_vq"
    fsq = tmp_path / "fsq_5x4"
    _make_run(vq, val_loss=0.4, entropy=0.9, dead=0.0)
    _make_run(fsq, val_loss=0.6, entropy=0.7, dead=0.1)

    comparison = build_comparison(vq, fsq)
    assert set(comparison) == {"ema_vq", "fsq_5x4"}
    assert comparison["ema_vq"]["val_loss"] == 0.4
    assert comparison["ema_vq"]["state_entropy"] == 0.9
    assert comparison["fsq_5x4"]["dead_states"] == 0.1
    assert comparison["ema_vq"]["aligned_mi"] == 1.23
    assert comparison["fsq_5x4"]["mi_tot"] == 2.34


def test_core_does_not_import_comparison_driver() -> None:
    """The core tdi.v2 modules must never import the standalone comparison driver."""
    v2_dir = project_root / "src" / "tdi" / "v2"
    for py in v2_dir.glob("*.py"):
        assert "compare_quantizers" not in py.read_text(), py.name
