"""Portable Foldseek 3Di SCOP search benchmark.

The historical benchmark ranks an all-versus-all Smith-Waterman search over a fixed
SCOP validation split, then measures family, superfamily, and fold recovery before the
first different-fold hit.  This module keeps that protocol while replacing its Linux-only
shell and AWK orchestration with validated Python code.
"""

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.hashing import sha256_file
from .encode import process_pdb
from .model import AlphabetModel
from .submat import accumulate_counts, calc_alphabet_mi, merge_columns, write_mat
from .util import parse_pairfile_line, resolve_pdb_path

HISTORICAL_ALPHABET = "ABCDEFGHIJKLMNOPQRST"
TMALIGN_REFERENCE = (0.928162, 0.662063, 0.275436)


@dataclass(frozen=True)
class ScopClass:
    """SCOP family, superfamily, and fold labels for one domain."""

    family: str
    superfamily: str
    fold: str


@dataclass(frozen=True)
class Roc1Row:
    """Per-query values emitted by the historical ``roc1.awk`` benchmark."""

    name: str
    scop: str
    family: float
    superfamily: float
    fold: float
    false_positives: int
    family_count: int
    superfamily_count: int
    fold_count: int


def read_sid_list(path: str | Path) -> list[str]:
    """Read a one-SID-per-line manifest, rejecting duplicates and extra columns."""
    sids: list[str] = []
    seen: set[str] = set()
    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 1:
                raise ValueError(
                    f"{path}:{line_number}: expected one SID, got {len(fields)} fields."
                )
            sid = fields[0]
            if sid in seen:
                raise ValueError(f"{path}:{line_number}: duplicate SID {sid!r}.")
            seen.add(sid)
            sids.append(sid)
    if not sids:
        raise ValueError(f"SID list {path} is empty.")
    return sids


def read_sequences(path: str | Path) -> dict[str, str]:
    """Read a strict SID-to-sequence table for artifact benchmark mode."""
    sequences: dict[str, str] = {}
    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected SID and sequence, got {len(fields)} fields."
                )
            sid, sequence = fields
            if sid in sequences:
                raise ValueError(f"{path}:{line_number}: duplicate sequence SID {sid!r}.")
            sequences[sid] = sequence
    if not sequences:
        raise ValueError(f"Sequence file {path} is empty.")
    return sequences


def load_scop_lookup(path: str | Path, sids: set[str] | None = None) -> dict[str, ScopClass]:
    """Load SCOP labels, optionally restricting them to a benchmark SID set."""
    lookup: dict[str, ScopClass] = {}
    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected SID and SCOP family, got {len(fields)} fields."
                )
            sid, family = fields
            if sids is not None and sid not in sids:
                continue
            parts = family.rsplit(".", maxsplit=2)
            if len(parts) != 3 or not parts[1] or not parts[2]:
                raise ValueError(f"{path}:{line_number}: malformed SCOP family {family!r}.")
            superfamily = family.rsplit(".", maxsplit=1)[0]
            fold = superfamily.rsplit(".", maxsplit=1)[0]
            if sid in lookup:
                raise ValueError(f"{path}:{line_number}: duplicate SCOP SID {sid!r}.")
            lookup[sid] = ScopClass(family, superfamily, fold)

    if sids is not None:
        missing = sorted(sids - lookup.keys())
        if missing:
            raise ValueError(
                f"SCOP lookup {path} is missing {len(missing)} requested SIDs; "
                f"sample: {missing[:10]}."
            )
    if not lookup:
        raise ValueError(f"SCOP lookup {path} contains no selected domains.")
    return lookup


def iter_hits(paths: Iterable[str | Path]) -> Iterator[tuple[str, str, int]]:
    """Yield ranked ``(query, target, score)`` hits from one or more whitespace tables."""
    for path in paths:
        with open(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) < 3:
                    raise ValueError(f"{path}:{line_number}: expected query, target, and score.")
                try:
                    score = int(fields[2])
                except ValueError as error:
                    raise ValueError(
                        f"{path}:{line_number}: score {fields[2]!r} is not an integer."
                    ) from error
                yield fields[0], fields[1], score


def score_roc1(lookup: dict[str, ScopClass], hits: Iterable[tuple[str, str, int]]) -> list[Roc1Row]:
    """Reproduce ``roc1.awk`` with deterministic output and input validation.

    Hits must be grouped by query and sorted by non-increasing score.  Self-hits are
    deliberately retained because the published benchmark included them in both numerator
    and family denominator.
    """
    family_counts: dict[str, int] = {}
    superfamily_counts: dict[str, int] = {}
    fold_counts: dict[str, int] = {}
    for scop in lookup.values():
        family_counts[scop.family] = family_counts.get(scop.family, 0) + 1
        superfamily_counts[scop.superfamily] = superfamily_counts.get(scop.superfamily, 0) + 1
        fold_counts[scop.fold] = fold_counts.get(scop.fold, 0) + 1

    found: dict[str, list[int]] = {}
    current_query: str | None = None
    current_score: int | None = None
    current_targets: set[str] = set()
    completed_queries: set[str] = set()

    for query, target, score in hits:
        if query not in lookup:
            raise ValueError(f"Hit query {query!r} is absent from filtered SCOP lookup.")
        if target not in lookup:
            raise ValueError(f"Hit target {target!r} is absent from filtered SCOP lookup.")
        if query != current_query:
            if current_query is not None:
                completed_queries.add(current_query)
            if query in completed_queries:
                raise ValueError(f"Hits for query {query!r} are not contiguous.")
            current_query = query
            current_score = None
            current_targets = set()
        if current_score is not None and score > current_score:
            raise ValueError(f"Hits for query {query!r} are not sorted by descending score.")
        current_score = score
        if target in current_targets:
            raise ValueError(f"Duplicate hit pair {query!r} -> {target!r}.")
        current_targets.add(target)

        # found = [family, superfamily, fold, first false-positive seen]
        query_found = found.setdefault(query, [0, 0, 0, 0])
        if query_found[3]:
            continue
        query_scop = lookup[query]
        target_scop = lookup[target]
        if query_scop.fold != target_scop.fold:
            query_found[3] = 1
        elif query_scop.family == target_scop.family:
            query_found[0] += 1
        elif query_scop.superfamily == target_scop.superfamily:
            query_found[1] += 1
        else:
            query_found[2] += 1

    rows: list[Roc1Row] = []
    for sid, scop in sorted(lookup.items()):
        family_count = family_counts[scop.family]
        superfamily_count = superfamily_counts[scop.superfamily]
        fold_count = fold_counts[scop.fold]
        superfamily_denominator = superfamily_count - family_count
        fold_denominator = fold_count - superfamily_count
        # Same eligibility condition as roc1.awk.
        if superfamily_denominator <= 0 or fold_denominator <= 0:
            continue
        family_found, superfamily_found, fold_found, false_positive = found.get(sid, [0, 0, 0, 0])
        rows.append(
            Roc1Row(
                name=sid,
                scop=scop.family,
                family=family_found / family_count,
                superfamily=superfamily_found / superfamily_denominator,
                fold=fold_found / fold_denominator,
                false_positives=false_positive,
                family_count=family_count,
                superfamily_count=superfamily_count,
                fold_count=fold_count,
            )
        )
    if not rows:
        raise ValueError("No SCOP queries meet the historical ROC1 denominator requirements.")
    return rows


def summarize_roc1(rows: Sequence[Roc1Row]) -> dict[str, float | int]:
    """Average ROC1 columns and expose the original header-counting result for parity."""
    family = float(np.mean([row.family for row in rows]))
    superfamily = float(np.mean([row.superfamily for row in rows]))
    fold = float(np.mean([row.fold for row in rows]))
    combined = float(
        np.mean(
            [
                family / TMALIGN_REFERENCE[0],
                superfamily / TMALIGN_REFERENCE[1],
                fold / TMALIGN_REFERENCE[2],
            ]
        )
    )
    # run-benchmark.sh averaged result.rocx without skipping roc1.awk's header. AWK
    # coerced the three header strings to zero while NR still included that row.
    legacy_factor = len(rows) / (len(rows) + 1)
    legacy_family = family * legacy_factor
    legacy_superfamily = superfamily * legacy_factor
    legacy_fold = fold * legacy_factor
    legacy_combined = float(
        np.mean(
            [
                legacy_family / TMALIGN_REFERENCE[0],
                legacy_superfamily / TMALIGN_REFERENCE[1],
                legacy_fold / TMALIGN_REFERENCE[2],
            ]
        )
    )
    return {
        "n_eligible_queries": len(rows),
        "family_roc1": family,
        "superfamily_roc1": superfamily,
        "fold_roc1": fold,
        "tmalign_normalized_mean": combined,
        "legacy_header_biased_family_roc1": legacy_family,
        "legacy_header_biased_superfamily_roc1": legacy_superfamily,
        "legacy_header_biased_fold_roc1": legacy_fold,
        "legacy_header_biased_tmalign_normalized_mean": legacy_combined,
    }


def write_roc1(path: str | Path, rows: Sequence[Roc1Row]) -> None:
    """Write historical columns as deterministic tab-separated output."""
    with open(path, "w") as handle:
        handle.write("NAME\tSCOP\tFAM\tSFAM\tFOLD\tFP\tFAMCNT\tSFAMCNT\tFOLDCNT\n")
        for row in rows:
            handle.write(
                f"{row.name}\t{row.scop}\t{row.family:.12g}\t{row.superfamily:.12g}\t"
                f"{row.fold:.12g}\t{row.false_positives}\t{row.family_count}\t"
                f"{row.superfamily_count}\t{row.fold_count}\n"
            )


def _pairfile_sids(path: str | Path, allowed_sids: set[str]) -> set[str]:
    required: set[str] = set()
    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            parsed = parse_pairfile_line(line)
            if parsed is None:
                raise ValueError(f"{path}:{line_number}: expected at least three fields.")
            sid1, sid2, _ = parsed
            if sid1 not in allowed_sids or sid2 not in allowed_sids:
                raise ValueError(
                    f"{path}:{line_number}: pair {sid1!r}, {sid2!r} crosses train SID manifest."
                )
            required.update((sid1, sid2))
    if not required:
        raise ValueError(f"Training pairfile {path} contains no pairs.")
    return required


def _encode_sids(
    sids: Sequence[str],
    model: AlphabetModel,
    pdb_dir: str | Path,
    virtual_center: tuple[float, float, float],
    mean: np.ndarray | None,
    std: np.ndarray | None,
) -> tuple[dict[str, str], list[str]]:
    sequences: dict[str, str] = {}
    failed: list[str] = []
    for index, sid in enumerate(sids, start=1):
        try:
            pdb_path = resolve_pdb_path(pdb_dir, sid)
            _, sequence = process_pdb(
                pdb_path.name,
                model,
                None,
                str(pdb_path.parent),
                virtual_center,
                model.invalid_state,
                mean=mean,
                std=std,
            )
            sequences[sid] = sequence
        except Exception as error:
            print(f"Error encoding {sid}: {error}", file=sys.stderr)
            failed.append(sid)
        if index % 500 == 0 or index == len(sids):
            print(f"  encoded {index}/{len(sids)} structures ({len(failed)} failed)")
    return sequences, failed


def _replace_invalid_states(
    sequences: dict[str, str],
    pairfile: str | Path,
    alphabet: str,
    invalid_state: str,
) -> tuple[dict[str, str], str | None]:
    """Choose invalid-state replacement using training MI_tot, matching final Foldseek training."""
    if invalid_state in alphabet:
        return sequences.copy(), invalid_state
    invalid_count = sum(sequence.count(invalid_state) for sequence in sequences.values())
    if invalid_count == 0:
        return sequences.copy(), None

    extended = alphabet + invalid_state
    letter2idx = {letter: index for index, letter in enumerate(extended)}
    counts, counts_prev = accumulate_counts(str(pairfile), sequences, letter2idx, len(extended))
    invalid_index = len(alphabet)
    best_state: str | None = None
    best_mi_tot = -math.inf
    for state_index, state in enumerate(alphabet):
        merged_counts = merge_columns(counts, invalid_index, state_index)
        merged_prev = merge_columns(counts_prev, invalid_index, state_index)
        _, mi_tot = calc_alphabet_mi(merged_counts, merged_prev)
        if mi_tot > best_mi_tot:
            best_mi_tot = mi_tot
            best_state = state
    if best_state is None:
        raise RuntimeError("Could not select a replacement for invalid structural states.")
    return {
        sid: sequence.replace(invalid_state, best_state) for sid, sequence in sequences.items()
    }, best_state


def _make_submat(
    pairfile: str | Path, sequences: dict[str, str], alphabet: str, path: str | Path
) -> dict[str, float | int]:
    letter2idx = {letter: index for index, letter in enumerate(alphabet)}
    counts, counts_prev = accumulate_counts(str(pairfile), sequences, letter2idx, len(alphabet))
    if counts.sum() == 0:
        raise ValueError("Training alignments produced no valid state-pair counts.")
    mi, mi_tot = calc_alphabet_mi(counts, counts_prev)
    p_ab = counts / counts.sum()
    p_a = p_ab.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = 2 * np.log2(p_ab / (p_a * p_a[:, np.newaxis]))
    scores[~np.isfinite(scores)] = 0
    integer_scores = np.rint(scores).astype(int)
    if integer_scores.min() < -128 or integer_scores.max() > 127:
        raise ValueError("Substitution scores exceed SSW's signed 8-bit score range.")
    with open(path, "w") as handle:
        write_mat(handle, list(alphabet), integer_scores)
    return {"training_counts": int(counts.sum()), "training_mi": mi, "training_mi_tot": mi_tot}


def _write_sequences(path: str | Path, sequences: dict[str, str], order: Sequence[str]) -> None:
    with open(path, "w") as handle:
        for sid in order:
            handle.write(f"{sid} {sequences[sid]}\n")


def _validate_submat(path: str | Path, sequences: dict[str, str]) -> None:
    with open(path) as handle:
        lines = [line.split() for line in handle if line.strip()]
    if not lines:
        raise ValueError(f"Substitution matrix {path} is empty.")
    alphabet = lines[0]
    if len(alphabet) != 20 or "".join(alphabet) != HISTORICAL_ALPHABET:
        raise ValueError(
            f"Historical benchmark requires matrix alphabet {HISTORICAL_ALPHABET}, "
            f"got {''.join(alphabet)!r}."
        )
    if len(lines[1:]) != len(alphabet):
        raise ValueError(f"Substitution matrix {path} is not square.")
    for row_index, row in enumerate(lines[1:]):
        if len(row) != len(alphabet) + 1 or row[0] != alphabet[row_index]:
            raise ValueError(f"Substitution matrix {path} has malformed row {row_index + 1}.")
        try:
            values = [int(value) for value in row[1:]]
        except ValueError as error:
            raise ValueError(f"Substitution matrix {path} contains a non-integer score.") from error
        if min(values) < -128 or max(values) > 127:
            raise ValueError(f"Substitution matrix {path} exceeds SSW's signed 8-bit range.")
    unexpected = sorted(set("".join(sequences.values())) - set(alphabet))
    if unexpected:
        raise ValueError(
            f"Sequences contain states absent from substitution matrix: {''.join(unexpected)!r}."
        )


def _write_fasta(path: Path, sequences: dict[str, str], sids: Sequence[str]) -> None:
    with open(path, "w") as handle:
        for sid in sids:
            handle.write(f">{sid}\n{sequences[sid]}\n")


def _parse_ssw_output(lines: Iterable[str], output_path: Path) -> int:
    query: str | None = None
    target: str | None = None
    count = 0
    saw_output = False
    with open(output_path, "w") as output:
        for line in lines:
            fields = line.split()
            if not fields:
                continue
            saw_output = True
            # SSW releases have used both bare and colon-suffixed labels. Historical
            # parser matched line prefixes, so retain that compatibility.
            if fields[0].startswith("target"):
                if len(fields) < 2:
                    raise ValueError("Malformed SSW target line.")
                target = fields[1]
            elif fields[0].startswith("query"):
                if len(fields) < 2:
                    raise ValueError("Malformed SSW query line.")
                query = fields[1]
            elif fields[0].startswith("optimal_alignment_score"):
                if query is None or target is None or len(fields) < 2:
                    raise ValueError("SSW score appeared before query and target identifiers.")
                try:
                    score = int(fields[1])
                except ValueError as error:
                    raise ValueError(f"Malformed SSW score {fields[1]!r}.") from error
                output.write(f"{query}\t{target}\t{score}\n")
                count += 1
    if saw_output and count == 0:
        raise ValueError("SSW emitted output, but no alignment-score records were recognized.")
    return count


def _run_ssw_shard(
    ssw: Path,
    target_fasta: Path,
    query_fasta: Path,
    submat: Path,
    gap_open: int,
    gap_extend: int,
    min_score: int,
    hit_path: Path,
) -> int:
    raw_path = hit_path.with_suffix(".raw")
    stderr_path = hit_path.with_suffix(".stderr")
    command = [
        str(ssw),
        "-o",
        str(gap_open),
        "-e",
        str(gap_extend),
        "-a",
        submat.name,
        "-p",
        "-f",
        str(min_score),
        str(target_fasta.resolve()),
        str(query_fasta.resolve()),
    ]
    with open(stderr_path, "w") as stderr:
        process = subprocess.Popen(
            command,
            cwd=submat.parent,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
        )
        assert process.stdout is not None
        try:
            count = _parse_ssw_output(process.stdout, raw_path)
        except Exception:
            process.terminate()
            process.wait()
            raise
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"SSW failed for {query_fasta.name} with exit code {return_code}; see {stderr_path}."
        )
    sort_executable = shutil.which("sort")
    if sort_executable is None:
        raise RuntimeError("POSIX sort is required to rank SSW hits.")
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    with open(hit_path, "w") as output:
        result = subprocess.run(
            [sort_executable, "-k1,1", "-k3,3nr", "-k2,2", str(raw_path)],
            stdout=output,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Sorting {raw_path} failed: {result.stderr.strip()}")
    raw_path.unlink()
    if stderr_path.stat().st_size == 0:
        stderr_path.unlink()
    return count


def run_search(
    ssw: Path,
    sequences: dict[str, str],
    sid_order: Sequence[str],
    submat: Path,
    out_dir: Path,
    gap_open: int,
    gap_extend: int,
    min_score: int,
    jobs: int,
) -> list[Path]:
    """Run all-versus-all SSW in portable record-aligned shards."""
    if jobs < 1:
        raise ValueError("jobs must be at least 1.")
    work_dir = out_dir / "search"
    shard_dir = work_dir / "queries"
    hit_dir = work_dir / "hits"
    shard_dir.mkdir(parents=True)
    hit_dir.mkdir()
    target_fasta = work_dir / "target.fasta"
    # ssw_test historically uses a short fixed-size matrix-name buffer.
    search_submat = work_dir / "s.mat"
    shutil.copyfile(submat, search_submat)
    _write_fasta(target_fasta, sequences, sid_order)

    shard_size = math.ceil(len(sid_order) / min(jobs, len(sid_order)))
    shards: list[tuple[Path, Path]] = []
    for shard_index, start in enumerate(range(0, len(sid_order), shard_size)):
        query_path = shard_dir / f"query_{shard_index:03d}.fasta"
        hit_path = hit_dir / f"query_{shard_index:03d}.m8"
        _write_fasta(query_path, sequences, sid_order[start : start + shard_size])
        shards.append((query_path, hit_path))

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(jobs, len(shards))) as executor:
        futures = {
            executor.submit(
                _run_ssw_shard,
                ssw,
                target_fasta,
                query_path,
                search_submat,
                gap_open,
                gap_extend,
                min_score,
                hit_path,
            ): query_path
            for query_path, hit_path in shards
        }
        for future in concurrent.futures.as_completed(futures):
            query_path = futures[future]
            hit_count = future.result()
            print(f"  {query_path.name}: {hit_count} hits")
    return [hit_path for _, hit_path in shards]


def _resolve_executable(value: str | Path) -> Path:
    text = str(value)
    resolved = shutil.which(text)
    if resolved is None:
        candidate = Path(text)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            resolved = str(candidate.resolve())
    if resolved is None:
        raise FileNotFoundError(
            f"SSW executable {text!r} was not found or is not executable. Build "
            "Complete-Striped-Smith-Waterman-Library's ssw_test and pass --ssw PATH."
        )
    return Path(resolved)


def run_roc1(args: argparse.Namespace) -> dict[str, float | int]:
    """Run standalone ROC1 scoring from argparse-compatible arguments."""
    sid_set = set(read_sid_list(args.sid_list))
    lookup = load_scop_lookup(args.scop_lookup, sid_set)
    rows = score_roc1(lookup, iter_hits(args.hits))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_roc1(output, rows)
    summary = summarize_roc1(rows)
    print(json.dumps(summary, indent=2))
    return summary


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """Prepare alphabet artifacts, run SSW, and score historical ROC1 sensitivity."""
    out_dir = Path(args.out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise FileExistsError(f"Benchmark output path {out_dir} is not a directory.")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Benchmark output directory {out_dir} is not empty.")
    out_dir.mkdir(parents=True, exist_ok=True)

    gap_open = int(args.gap_open)
    gap_extend = int(args.gap_extend)
    min_score = int(args.min_score)
    jobs = int(args.jobs)
    if gap_open <= 0 or gap_extend <= 0 or min_score <= 0:
        raise ValueError("gap_open, gap_extend, and min_score must be positive integers.")
    if jobs <= 0:
        raise ValueError("jobs must be at least 1.")

    ssw = _resolve_executable(args.ssw)
    val_sids = read_sid_list(args.val_sids)
    lookup = load_scop_lookup(args.scop_lookup, set(val_sids))
    model_dir = args.model_dir
    sequence_path_arg = args.sequences
    submat_arg = args.submat
    prep_report: dict[str, object] = {}

    sequences_path = out_dir / "sequences.txt"
    submat_path = out_dir / "submat.txt"
    if model_dir is not None:
        if sequence_path_arg is not None or submat_arg is not None:
            raise ValueError("Use either --model-dir or --sequences with --submat, not both.")
        model, mean, std = AlphabetModel.load(model_dir)
        if model.quantizer_name == "continuous":
            raise ValueError("Continuous bypass has no alphabet and cannot run search benchmark.")
        alphabet = model.letters[: model.n_states]
        if model.n_states != 20 or alphabet != HISTORICAL_ALPHABET:
            raise ValueError(
                f"Historical benchmark requires 20 states {HISTORICAL_ALPHABET}, got "
                f"{model.n_states} states {alphabet!r}."
            )
        if len(model.invalid_state) != 1:
            raise ValueError("Model invalid_state must be exactly one character.")
        virtual_center_arg = args.virt
        virtual_center_value = virtual_center_arg or model.virtual_center
        if virtual_center_value is None:
            raise ValueError("Model has no virtual center; pass --virt ALPHA BETA DISTANCE.")
        if len(virtual_center_value) != 3:
            raise ValueError("Virtual center must contain exactly three values.")
        virtual_center = (
            float(virtual_center_value[0]),
            float(virtual_center_value[1]),
            float(virtual_center_value[2]),
        )
        train_manifest = set(read_sid_list(args.train_sids))
        train_required = _pairfile_sids(args.train_pairfile, train_manifest)
        encode_order = sorted(train_required | set(val_sids))
        print(f"Encoding {len(encode_order)} train/validation structures...")
        all_sequences, failed = _encode_sids(
            encode_order,
            model,
            args.pdb_dir,
            virtual_center,
            mean,
            std,
        )
        missing_train = sorted(train_required - all_sequences.keys())
        missing_val = sorted(set(val_sids) - all_sequences.keys())
        if missing_train or missing_val:
            raise RuntimeError(
                f"Benchmark requires complete splits; missing {len(missing_train)} training and "
                f"{len(missing_val)} validation structures."
            )
        replaced_sequences, replacement = _replace_invalid_states(
            all_sequences,
            args.train_pairfile,
            alphabet,
            model.invalid_state,
        )
        prep_report.update(
            _make_submat(args.train_pairfile, replaced_sequences, alphabet, submat_path)
        )
        val_sequences = {sid: replaced_sequences[sid] for sid in val_sids}
        _write_sequences(sequences_path, val_sequences, val_sids)
        prep_report.update(
            {
                "source": "model",
                "model_dir": str(Path(model_dir)),
                "n_encoded": len(all_sequences),
                "n_failed": len(failed),
                "invalid_state": model.invalid_state,
                "invalid_state_replacement": replacement,
                "invalid_state_policy": "training_mi_tot_merge",
                "virtual_center": list(virtual_center),
            }
        )
    else:
        if sequence_path_arg is None or submat_arg is None:
            raise ValueError("Pass --model-dir, or pass both --sequences and --submat.")
        loaded = read_sequences(sequence_path_arg)
        missing_val = sorted(set(val_sids) - loaded.keys())
        if missing_val:
            raise ValueError(
                f"Sequence file is missing {len(missing_val)} validation SIDs; "
                f"sample: {missing_val[:10]}."
            )
        val_sequences = {sid: loaded[sid] for sid in val_sids}
        _write_sequences(sequences_path, val_sequences, val_sids)
        shutil.copyfile(submat_arg, submat_path)
        prep_report.update(
            {
                "source": "artifacts",
                "source_sequences": str(sequence_path_arg),
                "source_submat": str(submat_arg),
            }
        )

    _validate_submat(submat_path, val_sequences)
    print(f"Running all-versus-all SSW over {len(val_sids)} validation structures...")
    hit_paths = run_search(
        ssw,
        val_sequences,
        val_sids,
        submat_path,
        out_dir,
        gap_open,
        gap_extend,
        min_score,
        jobs,
    )
    rows = score_roc1(lookup, iter_hits(hit_paths))
    roc1_path = out_dir / "roc1.tsv"
    write_roc1(roc1_path, rows)
    summary = summarize_roc1(rows)
    input_records = {
        "validation_sids": {
            "path": str(args.val_sids),
            "sha256": sha256_file(args.val_sids),
        },
        "scop_lookup": {
            "path": str(args.scop_lookup),
            "sha256": sha256_file(args.scop_lookup),
        },
    }
    if model_dir is not None:
        input_records.update(
            {
                "training_sids": {
                    "path": str(args.train_sids),
                    "sha256": sha256_file(args.train_sids),
                },
                "training_pairfile": {
                    "path": str(args.train_pairfile),
                    "sha256": sha256_file(args.train_pairfile),
                },
            }
        )
    report: dict[str, object] = {
        **prep_report,
        **summary,
        "n_validation_domains": len(val_sids),
        "gap_open": gap_open,
        "gap_extend": gap_extend,
        "min_score": min_score,
        "ssw": str(ssw),
        "ssw_sha256": sha256_file(ssw),
        "metric": (
            "historical ROC1-like recovery before first different-fold hit; self-hit included"
        ),
        "tmalign_reference": {
            "family": TMALIGN_REFERENCE[0],
            "superfamily": TMALIGN_REFERENCE[1],
            "fold": TMALIGN_REFERENCE[2],
        },
        "inputs": input_records,
        "artifacts": {
            "sequences_sha256": sha256_file(sequences_path),
            "submat_sha256": sha256_file(submat_path),
            "roc1_sha256": sha256_file(roc1_path),
        },
    }
    with open(out_dir / "benchmark_report.json", "w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))
    return report
