"""Quarantined experiment: alignment-balanced batch sampler.

Removed from the core (which uses a plain shuffled loader), this keeps the self-contained
``AlignmentBatchSampler`` as a runnable snapshot. It depends only on numpy + torch's ``Sampler``
base, nothing in ``tdi.v2``; pass per-row ``alignment_ids`` to draw several distinct alignments
per batch.
"""

from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class AlignmentBatchSampler(Sampler[list[int]]):
    """Batch sampler drawing several distinct alignments per batch.

    Flat random sampling fills batches with correlated residues from one structure pair, which
    biases code-usage statistics. This sampler picks ``alignments_per_batch`` distinct
    alignments per batch, then draws rows within them, so each batch spans many alignments.
    Reproducible under a fixed seed; vary with epoch via ``set_epoch``.

    Stochastic sampling, not a partition: across one epoch some rows may be drawn more than once
    and others not at all. Each yielded batch is filled to exactly ``batch_size``.
    """

    def __init__(
        self,
        alignment_ids: Sequence[object] | np.ndarray,
        batch_size: int,
        alignments_per_batch: int,
        seed: int = 0,
    ) -> None:
        """Initialize the sampler.

        Args:
            alignment_ids: Per-row alignment identifier (length == dataset length).
            batch_size: Rows per batch.
            alignments_per_batch: Target number of distinct alignments per batch.
            seed: Base seed for reproducible draws.
        """
        if alignments_per_batch < 1:
            raise ValueError("alignments_per_batch must be >= 1")
        self.batch_size = batch_size
        self.alignments_per_batch = alignments_per_batch
        self.seed = seed
        self.epoch = 0

        groups: dict[object, list[int]] = defaultdict(list)
        for idx, aid in enumerate(alignment_ids):
            groups[aid].append(idx)
        self.groups = [np.asarray(v, dtype=np.int64) for v in groups.values()]
        self.n = len(alignment_ids)
        self.num_batches = self.n // batch_size
        # Rows drawn per alignment so that alignments_per_batch of them fill a batch.
        self.per_alignment = max(1, batch_size // alignments_per_batch)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so batch composition varies per epoch but stays reproducible."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng([self.seed, self.epoch])
        n_groups = len(self.groups)
        for _ in range(self.num_batches):
            order = rng.permutation(n_groups)
            batch: list[int] = []
            pos = 0
            # Consume distinct alignments until the batch is full, re-permuting the alignment
            # order if we run out before reaching batch_size.
            while len(batch) < self.batch_size:
                if pos >= n_groups:
                    order = rng.permutation(n_groups)
                    pos = 0
                members = self.groups[order[pos]]
                pos += 1
                take = min(self.per_alignment, len(members), self.batch_size - len(batch))
                sel = rng.choice(len(members), size=take, replace=False)
                batch.extend(members[sel].tolist())
            yield batch
