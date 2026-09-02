#!/usr/bin/env python3
"""Score a no-learning baseline: k-means straight on the raw 10-D residue descriptors.

No encoder, no decoder, no quantizer, no training. Each residue is assigned to the nearest
of K centroids fitted on the standardized training descriptors, and the resulting sequences
go through the *same* evaluation path as a trained run -- same structures, same pairfile,
same MI and substitution-matrix code.

That makes the output directly comparable to a trained arm and answers the question no
relative comparison between trained arms can: how much does the learned pipeline actually
buy over clustering the inputs?

Nothing in ``tdi`` is modified. The script writes a run-shaped directory, so
``scripts/evaluate_runs.py`` tabulates it alongside the trained runs without special-casing.

Usage::

    uv run python scripts/kmeans_baseline.py --out runs/ablation/kmeans_raw
"""

import argparse
import contextlib
import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

import tdi.v2.cli
from tdi.v2.cli import run_evaluate
from tdi.v2.model import AlphabetModel
from tdi.v2.train import _read_provenance


class RawDescriptorAlphabet(AlphabetModel):
    """An AlphabetModel whose states are nearest-centroid assignments on the raw input.

    Subclassing matters rather than duck-typing: ``encode.discretize`` dispatches to
    ``encode_states`` only for a real ``AlphabetModel``, so this keeps the encoding path
    byte-identical to a trained run. The inherited encoder, decoder, and codebook are built
    but never consulted -- ``encode_states`` bypasses them entirely.
    """

    def __init__(self, centroids: np.ndarray) -> None:
        """Wrap fitted centroids in the model interface the evaluation path expects.

        Args:
            centroids: Fitted cluster centres, shape (n_states, input_dim), in standardized
                descriptor space.
        """
        n_states, input_dim = centroids.shape
        # hidden_dim/z_dim are placeholders: the inherited nets are never run.
        super().__init__(
            input_dim=input_dim, hidden_dim=8, z_dim=2, n_states=n_states, quantizer="vq"
        )
        self.register_buffer("raw_centroids", torch.tensor(centroids, dtype=torch.float32))

    @torch.no_grad()
    def encode_states(self, x: torch.Tensor) -> torch.Tensor:
        """Assign each standardized descriptor to its nearest centroid."""
        centroids = self.get_buffer("raw_centroids")
        return torch.cdist(x, centroids).argmin(dim=-1)


@contextlib.contextmanager
def _stub_export_loader(model: AlphabetModel, mean: np.ndarray, std: np.ndarray) -> Iterator[None]:
    """Make ``run_evaluate`` use the in-memory baseline instead of reading an export.

    ``run_evaluate`` is reused unchanged so the MI, substitution matrix, and usage
    diagnostics are computed by exactly the same code as for trained runs; only the source
    of the model is redirected.
    """

    class _StubLoader:
        @staticmethod
        def load(export_dir: Path | str) -> tuple[AlphabetModel, np.ndarray, np.ndarray]:
            return model, mean, std

    original = tdi.v2.cli.AlphabetModel
    setattr(tdi.v2.cli, "AlphabetModel", _StubLoader)
    try:
        yield
    finally:
        setattr(tdi.v2.cli, "AlphabetModel", original)


def _fit_centroids(
    processed_dir: Path, n_states: int, sample: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit k-means on standardized training descriptors.

    Returns (centroids, scaler mean, scaler std). Standardization uses the dataset's
    train-only scaler, so the baseline sees precisely the inputs a trained encoder sees.
    """
    descriptors = np.load(processed_dir / "train_x_raw.npy")
    scaler = np.load(processed_dir / "scaler.npz")
    mean, std = scaler["mean"], scaler["std"]

    rng = np.random.default_rng(seed)
    if sample and sample < len(descriptors):
        # Fitting on every row is unnecessary for 20 centroids and dominates runtime.
        rows = rng.choice(len(descriptors), size=sample, replace=False)
        descriptors = descriptors[rows]

    standardized = (descriptors - mean) / std
    print(f"Fitting k-means: K={n_states} on {len(standardized):,} standardized descriptors...")
    kmeans = KMeans(n_clusters=n_states, random_state=seed, n_init="auto").fit(standardized)
    return kmeans.cluster_centers_.astype(np.float32), mean, std


def _write_run_shell(
    out_dir: Path, centroids: np.ndarray, mean: np.ndarray, std: np.ndarray, n_states: int
) -> None:
    """Write the run-shaped metadata that ``evaluate_runs.py`` reads when tabulating."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "centroids.npy", centroids)
    with open(out_dir / "scaler.json", "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                "input_dim": int(centroids.shape[1]),
                "hidden_dim": 0,
                # z_dim is the descriptor width: clustering happens in the input space itself.
                "z_dim": int(centroids.shape[1]),
                "n_states": n_states,
                "quantizer": "kmeans_raw",
                "levels": None,
                "loss": "none",
                "note": "No-learning baseline: nearest centroid on raw standardized descriptors.",
            },
            f,
            indent=2,
        )


def main() -> None:
    """Fit the baseline, evaluate it on the validation alignments, and print the headline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/ablation/kmeans_raw", help="Output run directory.")
    parser.add_argument(
        "--processed-dir",
        default="data/processed/scop_ca5_v1",
        help="Processed dataset providing descriptors and the train-only scaler.",
    )
    parser.add_argument(
        "--pdb-dir",
        default="data/external/foldseek_scop40/pdb_by_sid",
        help="Directory of structures to encode.",
    )
    parser.add_argument(
        "--pairfile",
        default="data/derived/pairfiles/tmaln-06.val.out",
        help="Validation alignment pairfile.",
    )
    parser.add_argument("--n-states", type=int, default=20, help="Number of clusters.")
    parser.add_argument(
        "--sample", type=int, default=500_000, help="Rows to fit on (0 = all rows)."
    )
    parser.add_argument("--seed", type=int, default=0, help="k-means seed.")
    parser.add_argument(
        "--virt",
        type=float,
        nargs=3,
        default=None,
        help="Virtual center override; otherwise read from the dataset manifest.",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out)

    centroids, mean, std = _fit_centroids(processed_dir, args.n_states, args.sample, args.seed)
    _write_run_shell(out_dir, centroids, mean, std, args.n_states)

    model = RawDescriptorAlphabet(centroids)
    # Provenance: the baseline must encode structures with the same geometry as the runs it
    # is compared against, so take the virtual center from the dataset that produced them.
    virtual_center, max_ca_dist = _read_provenance(processed_dir)
    model.virtual_center = list(args.virt) if args.virt is not None else virtual_center
    model.max_ca_dist = max_ca_dist
    if model.virtual_center is None:
        parser.error("No virtual center in the dataset manifest; pass --virt explicitly.")

    with _stub_export_loader(model, mean, std):
        run_evaluate(
            argparse.Namespace(
                model_dir=str(out_dir),
                pdb_dir=args.pdb_dir,
                pairfile=args.pairfile,
                out_dir=str(out_dir / "eval"),
                virt=None,
                invalid_state=None,
                max_failure_rate=1.0,
            )
        )

    with open(out_dir / "eval" / "evaluation_report.json") as f:
        report = json.load(f)
    capacity = math.log2(args.n_states)
    entropy_bits = report["normalized_entropy"] * capacity
    print(
        f"\nk-means baseline (K={args.n_states}, no learning):\n"
        f"  mi      {report['mi']:.4f} bits\n"
        f"  mi_tot  {report['mi_tot']:.4f} bits   ({100 * report['mi_tot'] / capacity:.1f}% of "
        f"log2(K))\n"
        f"  eff_K   {2**entropy_bits:.2f}   dead_state_fraction {report['dead_state_fraction']:.3f}"
    )
    print(
        "\nCompare mi_tot against the trained arms. A learned model close to this number is "
        "not\nearning its complexity; a large margin over it means the pipeline works and the "
        "ceiling\nlies elsewhere."
    )


if __name__ == "__main__":
    main()
