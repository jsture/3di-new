#!/usr/bin/env python3
"""
Fetch and wrangle the Foldseek SCOPe40 benchmark structure archive.

This script performs only procurement/wrangling:
  - download archive
  - safe extraction
  - copy/link structures into a stable SID-indexed directory
  - write a manifest and summary

It does not:
  - compute 3Di descriptors
  - parse CIGAR strings
  - generate training pairs
  - filter aligned residue pairs
  - split train/validation sets
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_URLS = [
    # Name used by the Foldseek benchmark-data README.
    "https://wwwuser.gwdg.de/~compbiol/foldseek/scop40pdb.tar.gz",
    # Some older scripts/repos have used this spelling.
    "https://wwwuser.gwdg.de/~compbiol/foldseek/scp40pdb.tar.gz",
]


@dataclass(frozen=True)
class ManifestRow:
    sid: str
    normalized_path: str
    original_path: str
    size_bytes: int
    sha256: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_bytes(n_bytes: int) -> str:
    mib = n_bytes / (1024 * 1024)
    return f"{mib:.1f} MiB"


def copy_response_with_progress(response, out, chunk_size: int = 1024 * 1024) -> None:
    content_length = response.headers.get("Content-Length")
    total = int(content_length) if content_length and content_length.isdigit() else None

    downloaded = 0
    last_update = 0.0
    next_log_percent = 10
    next_log_bytes = 25 * 1024 * 1024
    stderr_is_tty = sys.stderr.isatty()

    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break

        out.write(chunk)
        downloaded += len(chunk)

        now = time.monotonic()
        should_update_tty = stderr_is_tty and now - last_update >= 0.2
        should_log_percent = (
            not stderr_is_tty
            and total is not None
            and next_log_percent < 100
            and downloaded < total
            and total - downloaded >= chunk_size
            and downloaded * 100 // total >= next_log_percent
        )
        should_log_bytes = not stderr_is_tty and total is None and downloaded >= next_log_bytes

        if should_update_tty:
            if total is not None:
                percent = downloaded * 100 / total
                print(
                    f"\rDownloaded {format_bytes(downloaded)} / "
                    f"{format_bytes(total)} ({percent:.1f}%)",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"\rDownloaded {format_bytes(downloaded)}",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
            last_update = now

        elif should_log_percent and total is not None:
            percent = downloaded * 100 / total
            print(
                f"Downloaded {format_bytes(downloaded)} / {format_bytes(total)} ({percent:.1f}%)",
                file=sys.stderr,
            )
            while downloaded * 100 // total >= next_log_percent:
                next_log_percent += 10

        elif should_log_bytes:
            print(f"Downloaded {format_bytes(downloaded)}", file=sys.stderr)
            next_log_bytes += 25 * 1024 * 1024

    if stderr_is_tty:
        if total is not None:
            print(
                f"\rDownloaded {format_bytes(downloaded)} / {format_bytes(total)} (100.0%)        ",
                file=sys.stderr,
            )
        else:
            print(f"\rDownloaded {format_bytes(downloaded)}        ", file=sys.stderr)
    elif total is not None:
        print(
            f"Downloaded {format_bytes(downloaded)} / {format_bytes(total)} (100.0%)",
            file=sys.stderr,
        )
    else:
        print(f"Downloaded {format_bytes(downloaded)}", file=sys.stderr)


def download_file(urls: list[str], destination: Path, force: bool = False) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        print(f"Using existing download: {destination}", file=sys.stderr)
        return "existing"

    last_error: Exception | None = None

    for url in urls:
        try:
            print(f"Downloading: {url}", file=sys.stderr)
            tmp = destination.with_suffix(destination.suffix + ".tmp")

            with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
                copy_response_with_progress(response, out)

            tmp.replace(destination)
            return url

        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            last_error = exc
            print(f"Failed: {url} ({exc})", file=sys.stderr)

    raise RuntimeError(f"All download URLs failed. Last error: {last_error}")


def safe_extract_tar(archive: Path, destination: Path, force: bool = False) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    marker = destination / ".extracted"
    if marker.exists() and not force:
        print(f"Using existing extraction: {destination}", file=sys.stderr)
        return

    if force and destination.exists():
        shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)

    destination_resolved = destination.resolve()

    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()

        for member in members:
            target = (destination / member.name).resolve()
            if not str(target).startswith(str(destination_resolved)):
                raise RuntimeError(f"Unsafe path in tar archive: {member.name}")

        tar.extractall(destination, filter="data")

    marker.write_text("ok\n")


def is_probable_pdb_domain_file(path: Path) -> bool:
    if not path.is_file():
        return False

    name = path.name
    suffix = path.suffix.lower()

    if name.startswith("."):
        return False

    # SCOP domain identifiers commonly start with "d".
    # The Foldseek archive may contain files without a .pdb extension.
    if name.startswith("d"):
        return True

    if suffix in {".pdb", ".ent"}:
        return True

    return False


def sid_from_path(path: Path) -> str:
    if path.suffix.lower() in {".pdb", ".ent"}:
        return path.stem
    return path.name


def iter_structure_files(extracted_dir: Path) -> Iterable[Path]:
    for path in sorted(extracted_dir.rglob("*")):
        if is_probable_pdb_domain_file(path):
            yield path


def wrangle_structures(
    extracted_dir: Path,
    normalized_dir: Path,
    copy_mode: str = "copy",
    force: bool = False,
) -> list[ManifestRow]:
    if force and normalized_dir.exists():
        shutil.rmtree(normalized_dir)

    normalized_dir.mkdir(parents=True, exist_ok=True)

    rows: list[ManifestRow] = []
    seen: set[str] = set()

    for source in iter_structure_files(extracted_dir):
        sid = sid_from_path(source)

        if sid in seen:
            raise RuntimeError(
                f"Duplicate SID after normalization: {sid}\n"
                f"First duplicate encountered at: {source}"
            )

        seen.add(sid)
        dest = normalized_dir / sid

        if dest.exists():
            dest.unlink()

        if copy_mode == "copy":
            shutil.copy2(source, dest)
        elif copy_mode == "symlink":
            dest.symlink_to(source.resolve())
        elif copy_mode == "hardlink":
            os.link(source, dest)
        else:
            raise ValueError(f"Unsupported copy_mode: {copy_mode}")

        rows.append(
            ManifestRow(
                sid=sid,
                normalized_path=str(dest),
                original_path=str(source),
                size_bytes=dest.stat().st_size,
                sha256=sha256_file(dest),
            )
        )

    if not rows:
        raise RuntimeError(f"No probable PDB/domain files found under: {extracted_dir}")

    return rows


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("sid\tnormalized_path\toriginal_path\tsize_bytes\tsha256\n")
        for row in rows:
            handle.write(
                f"{row.sid}\t{row.normalized_path}\t{row.original_path}\t"
                f"{row.size_bytes}\t{row.sha256}\n"
            )


def read_sid_file(path: Path) -> set[str]:
    sids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Accept either "sid" or "sid classification".
            sid = stripped.split()[0]
            sids.add(sid)

    return sids


def validate_sid_file(name: str, path: Path, manifest_sids: set[str]) -> dict[str, object]:
    requested = read_sid_file(path)
    missing = sorted(requested - manifest_sids)

    return {
        "name": name,
        "path": str(path),
        "n_requested": len(requested),
        "n_missing": len(missing),
        "missing_examples": missing[:20],
    }


def validate_pairfile(
    path: Path, manifest_sids: set[str], max_examples: int = 20
) -> dict[str, object]:
    n_lines = 0
    n_pairs = 0
    missing: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            n_lines += 1
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            sid1, sid2 = parts[0], parts[1]
            n_pairs += 1

            if sid1 not in manifest_sids:
                missing.add(sid1)
            if sid2 not in manifest_sids:
                missing.add(sid2)

    missing_sorted = sorted(missing)

    return {
        "name": "pairfile",
        "path": str(path),
        "n_lines": n_lines,
        "n_pairs": n_pairs,
        "n_missing_sids": len(missing_sorted),
        "missing_examples": missing_sorted[:max_examples],
    }


def write_summary(
    *,
    path: Path,
    archive_path: Path,
    download_source: str,
    rows: list[ManifestRow],
    validation_results: list[dict[str, object]],
) -> None:
    total_bytes = sum(row.size_bytes for row in rows)

    summary = {
        "archive_path": str(archive_path),
        "download_source": download_source,
        "n_structures": len(rows),
        "total_structure_bytes": total_bytes,
        "sid_examples": [row.sid for row in rows[:10]],
        "validation": validation_results,
    }

    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and wrangle the Foldseek SCOPe40 structure archive."
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/external/foldseek_scop40"),
        help="Output directory for downloaded, extracted, and normalized files.",
    )

    parser.add_argument(
        "--url",
        action="append",
        default=None,
        help=(
            "Archive URL. Can be supplied multiple times. "
            "Defaults to Foldseek SCOPe40 benchmark URLs."
        ),
    )

    parser.add_argument(
        "--copy-mode",
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help="How to populate pdb_by_sid.",
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload archive even if it already exists.",
    )

    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract archive even if extracted marker exists.",
    )

    parser.add_argument(
        "--force-wrangle",
        action="store_true",
        help="Rebuild pdb_by_sid even if it already exists.",
    )

    parser.add_argument(
        "--train-sids",
        type=Path,
        default=None,
        help="Optional train SID file to validate against fetched structures.",
    )

    parser.add_argument(
        "--val-sids",
        type=Path,
        default=None,
        help="Optional validation SID file to validate against fetched structures.",
    )

    parser.add_argument(
        "--pairfile",
        type=Path,
        default=None,
        help="Optional pairfile to validate against fetched structures.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir: Path = args.out_dir
    downloads_dir = out_dir / "downloads"
    extracted_dir = out_dir / "extracted"
    normalized_dir = out_dir / "pdb_by_sid"
    manifest_path = out_dir / "manifest.tsv"
    summary_path = out_dir / "dataset_summary.json"
    archive_path = downloads_dir / "scop40pdb.tar.gz"

    urls = args.url if args.url else DEFAULT_URLS

    download_source = download_file(
        urls=urls,
        destination=archive_path,
        force=args.force_download,
    )

    safe_extract_tar(
        archive=archive_path,
        destination=extracted_dir,
        force=args.force_extract,
    )

    rows = wrangle_structures(
        extracted_dir=extracted_dir,
        normalized_dir=normalized_dir,
        copy_mode=args.copy_mode,
        force=args.force_wrangle,
    )

    write_manifest(rows, manifest_path)

    manifest_sids = {row.sid for row in rows}
    validation_results: list[dict[str, object]] = []

    if args.train_sids is not None:
        validation_results.append(validate_sid_file("train_sids", args.train_sids, manifest_sids))

    if args.val_sids is not None:
        validation_results.append(validate_sid_file("val_sids", args.val_sids, manifest_sids))

    if args.pairfile is not None:
        validation_results.append(validate_pairfile(args.pairfile, manifest_sids))

    write_summary(
        path=summary_path,
        archive_path=archive_path,
        download_source=download_source,
        rows=rows,
        validation_results=validation_results,
    )

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary:  {summary_path}")
    print(f"PDB-by-SID dir: {normalized_dir}")
    print(f"Structures:     {len(rows)}")


if __name__ == "__main__":
    main()
