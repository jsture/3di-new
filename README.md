# 3Di-new

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Orcid: Jakob](https://img.shields.io/badge/Jakob-bar?style=flat&logo=orcid&labelColor=white&color=grey)](https://orcid.org/0000-0002-2841-7284)

# Data layout

## `raw/`

Immutable baseline SCOPe/Foldseek-style metadata.

- `pdbs_train.txt`: training domain SIDs.
- `pdbs_val.txt`: validation domain SIDs.
- `scop_lookup.tsv`: SID-to-SCOP classification table.
- `tmaln-06.out`: original filtered structural-alignment pairfile.

These files should not be edited in place.

## `derived/pairfiles/`

Pairfiles derived from `raw/tmaln-06.out`.

- `tmaln-06.assigned.out`: alignments where both SIDs are in train or validation.
- `tmaln-06.train.out`: train/train alignments only.
- `tmaln-06.val.out`: validation/validation alignments only.
- `tmaln-06.cross_split.out`: train/validation alignments; diagnostic only, not used for training.

## `external/`

Downloaded and extracted structure archives. Ignored by git.

## Policy

Training uses `derived/pairfiles/tmaln-06.train.out`.

Validation diagnostics use `derived/pairfiles/tmaln-06.val.out`.

Cross-split and unassigned rows are ignored.

Residue-level C-alpha distance filtering is applied later during CIGAR expansion / feature generation, not when creating these pairfiles.
