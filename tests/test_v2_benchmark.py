"""Fast contract tests for historical Foldseek SCOP benchmark wiring."""

import os
from argparse import Namespace
from pathlib import Path

import pytest

from tdi.v2.benchmark import (
    HISTORICAL_ALPHABET,
    Roc1Row,
    ScopClass,
    _parse_ssw_output,
    _validate_submat,
    load_scop_lookup,
    read_sid_list,
    run_benchmark,
    score_roc1,
    summarize_roc1,
    write_roc1,
)


def _lookup() -> dict[str, ScopClass]:
    return {
        "q": ScopClass("a.1.1.1", "a.1.1", "a.1"),
        "fam": ScopClass("a.1.1.1", "a.1.1", "a.1"),
        "sfam": ScopClass("a.1.1.2", "a.1.1", "a.1"),
        "fold": ScopClass("a.1.2.1", "a.1.2", "a.1"),
        "fp": ScopClass("b.1.1.1", "b.1.1", "b.1"),
    }


def test_roc1_matches_historical_self_hit_and_first_fp_semantics(tmp_path: Path) -> None:
    hits = [
        ("q", "q", 100),
        ("q", "fam", 90),
        ("q", "sfam", 80),
        ("q", "fold", 70),
        ("q", "fp", 60),
    ]

    rows = score_roc1(_lookup(), hits)
    # Golden rows from original roc1.awk. All nine fields lock self-hit handling,
    # first-different-fold stopping, denominators, eligibility, and output values.
    assert rows == [
        Roc1Row("fam", "a.1.1.1", 0.0, 0.0, 0.0, 0, 2, 3, 4),
        Roc1Row("q", "a.1.1.1", 1.0, 1.0, 1.0, 1, 2, 3, 4),
        Roc1Row("sfam", "a.1.1.2", 0.0, 0.0, 0.0, 0, 1, 3, 4),
    ]

    output = tmp_path / "roc1.tsv"
    write_roc1(output, rows)
    assert output.read_text().splitlines()[0] == (
        "NAME\tSCOP\tFAM\tSFAM\tFOLD\tFP\tFAMCNT\tSFAMCNT\tFOLDCNT"
    )
    summary = summarize_roc1(rows)
    assert summary["n_eligible_queries"] == 3
    assert summary["family_roc1"] == pytest.approx(1 / 3)
    assert summary["legacy_header_biased_family_roc1"] == pytest.approx(1 / 4)


def test_roc1_rejects_unsorted_duplicate_and_unknown_hits() -> None:
    lookup = _lookup()
    with pytest.raises(ValueError, match="descending score"):
        score_roc1(lookup, [("q", "q", 1), ("q", "fam", 2)])
    with pytest.raises(ValueError, match="Duplicate hit pair"):
        score_roc1(lookup, [("q", "q", 2), ("q", "q", 1)])
    with pytest.raises(ValueError, match="absent from filtered SCOP lookup"):
        score_roc1(lookup, [("q", "unknown", 1)])


def test_manifest_and_scop_lookup_are_strict(tmp_path: Path) -> None:
    sid_path = tmp_path / "sids.txt"
    sid_path.write_text("q\nfam\n")
    lookup_path = tmp_path / "lookup.tsv"
    lookup_path.write_text("q\ta.1.1.1\nfam\ta.1.1.1\nextra\tb.1.1.1\n")

    sids = read_sid_list(sid_path)
    lookup = load_scop_lookup(lookup_path, set(sids))

    assert sids == ["q", "fam"]
    assert set(lookup) == {"q", "fam"}
    assert lookup["q"] == ScopClass("a.1.1.1", "a.1.1", "a.1")

    sid_path.write_text("q\nq\n")
    with pytest.raises(ValueError, match="duplicate SID"):
        read_sid_list(sid_path)


def test_ssw_parser_emits_rankable_hit_table(tmp_path: Path) -> None:
    output = tmp_path / "hits.raw"
    lines = [
        "target_name: target_a\n",
        "query_name: query_a\n",
        "optimal_alignment_score: 42\n",
        "target target_b\n",
        "query query_a\n",
        "optimal_alignment_score 7\n",
    ]

    count = _parse_ssw_output(lines, output)

    assert count == 2
    assert output.read_text() == "query_a\ttarget_a\t42\nquery_a\ttarget_b\t7\n"

    with pytest.raises(ValueError, match="no alignment-score records"):
        _parse_ssw_output(["unexpected output\n"], output)


def test_submat_validation_rejects_invalid_sequence_state(tmp_path: Path) -> None:
    matrix = tmp_path / "submat.txt"
    matrix.write_text(
        "  "
        + "   ".join(HISTORICAL_ALPHABET)
        + "\n"
        + "\n".join(
            state + "".join(f"{int(i == j):4d}" for j in range(20))
            for i, state in enumerate(HISTORICAL_ALPHABET)
        )
        + "\n"
    )

    _validate_submat(matrix, {"ok": HISTORICAL_ALPHABET})
    with pytest.raises(ValueError, match="absent from substitution matrix"):
        _validate_submat(matrix, {"bad": "AX"})


def test_artifact_benchmark_runs_fake_ssw_end_to_end(tmp_path: Path) -> None:
    sid_list = tmp_path / "sids.txt"
    sid_list.write_text("q\nfam\nsfam\nfold\nfp\n")
    lookup = tmp_path / "lookup.tsv"
    lookup.write_text("q\ta.1.1.1\nfam\ta.1.1.1\nsfam\ta.1.1.2\nfold\ta.1.2.1\nfp\tb.1.1.1\n")
    sequences = tmp_path / "sequences.txt"
    sequences.write_text("".join(f"{sid} ABCD\n" for sid in read_sid_list(sid_list)))
    matrix = tmp_path / "submat.txt"
    matrix.write_text(
        "  "
        + "   ".join(HISTORICAL_ALPHABET)
        + "\n"
        + "\n".join(
            state + "".join(f"{int(i == j):4d}" for j in range(20))
            for i, state in enumerate(HISTORICAL_ALPHABET)
        )
        + "\n"
    )
    fake_ssw = tmp_path / "ssw_test"
    fake_ssw.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        "def ids(path):\n"
        "    return [line[1:] for line in Path(path).read_text().splitlines() "
        "if line.startswith('>')]\n"
        "targets = ids(sys.argv[-2])\n"
        "for query in ids(sys.argv[-1]):\n"
        "    for target in targets:\n"
        "        print('target_name:', target)\n"
        "        print('query_name:', query)\n"
        "        print('optimal_alignment_score:', 100 if query == target else 10)\n"
    )
    os.chmod(fake_ssw, 0o755)

    out_dir = tmp_path / "benchmark"
    report = run_benchmark(
        Namespace(
            out_dir=str(out_dir),
            ssw=str(fake_ssw),
            val_sids=str(sid_list),
            scop_lookup=str(lookup),
            model_dir=None,
            sequences=str(sequences),
            submat=str(matrix),
            gap_open=8,
            gap_extend=2,
            min_score=50,
            jobs=2,
        )
    )

    assert report["n_validation_domains"] == 5
    assert report["n_eligible_queries"] == 3
    assert (out_dir / "roc1.tsv").exists()
    assert (out_dir / "benchmark_report.json").exists()
