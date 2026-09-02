#!/usr/bin/env python3
"""Score the published Foldseek 3Di alphabet through this repo's evaluation path.

The assets in ``data/v1`` are the original released model: ``encoder.pt`` is a pickled
``nn.Sequential`` (10 -> 10 -> 10 -> 2), ``states.txt`` holds its 20 centroids in that 2-D
latent, and ``sub_score.mat`` is the published substitution matrix. Together they define the
alphabet every trained run here is trying to beat, so scoring them on the same validation
pairfile with the same MI code turns a relative comparison between our own arms into an
absolute one.

Two conventions of the original pipeline matter and are honoured here:

* **No standardization.** v1 fed raw descriptors straight to the encoder; there is no scaler
  in ``data/v1``. Passing our train-fit z-scores would silently shift every input off the
  distribution the released weights were trained on, so ``mean``/``std`` are left ``None``.
* **Virtual center (270, 0, 2).** Read from the processed dataset's manifest, which records
  the same value the released model was built with, so the descriptors this script computes
  are the ones the encoder expects.

Nothing in ``tdi`` is modified. The script writes a run-shaped directory, so
``scripts/evaluate_runs.py`` tabulates the reference alongside the trained runs, and it
additionally diffs the substitution matrix it derives against the published one -- a
end-to-end check that our encoding and MI code reproduce the original result.

Usage::

    uv run python scripts/foldseek_v1_reference.py --out runs/ablation/foldseek_v1
"""

import argparse
import contextlib
import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import tdi.v2.cli
from tdi.v2.cli import run_evaluate
from tdi.v2.model import AlphabetModel
from tdi.v2.train import _read_provenance


class FoldseekV1Alphabet(AlphabetModel):
    """The released v1 encoder plus its published centroids, in the v2 model interface.

    Subclassing matters rather than duck-typing: ``run_evaluate`` always calls ``process_pdb``
    with ``centroids=None``, and ``encode.discretize`` only reaches its nearest-centroid
    lookup for a real ``AlphabetModel``. Wrapping the released ``Sequential`` here keeps the
    encoding path byte-identical to a trained run. The inherited encoder, decoder, and
    codebook are constructed but never consulted.
    """

    def __init__(self, v1_encoder: nn.Module, centroids: np.ndarray) -> None:
        """Wrap the released encoder and centroids.

        Args:
            v1_encoder: The pickled ``nn.Sequential`` from ``data/v1/encoder.pt``.
            centroids: Published state coordinates, shape (n_states, z_dim), from
                ``data/v1/states.txt``.
        """
        n_states, z_dim = centroids.shape
        # input_dim/hidden_dim mirror the released topology; the inherited nets are never run.
        super().__init__(
            input_dim=10, hidden_dim=10, z_dim=z_dim, n_states=n_states, quantizer="vq"
        )
        self.v1_encoder = v1_encoder
        self.v1_encoder.eval()
        self.register_buffer("v1_centroids", torch.tensor(centroids, dtype=torch.float32))

    @torch.no_grad()
    def encode_states(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw descriptors and assign each to its nearest published centroid."""
        z = self.v1_encoder.forward(x)
        centroids = self.get_buffer("v1_centroids")
        return torch.cdist(z, centroids).argmin(dim=-1)


@contextlib.contextmanager
def _stub_export_loader(model: AlphabetModel) -> Iterator[None]:
    """Make ``run_evaluate`` use the in-memory reference instead of reading an export.

    ``run_evaluate`` is reused unchanged so MI, the substitution matrix, and the usage
    diagnostics come from exactly the same code as for trained runs; only the source of the
    model is redirected. ``mean``/``std`` are returned as ``None`` so ``process_pdb`` skips
    standardization, matching the original v1 convention.
    """

    class _StubLoader:
        @staticmethod
        def load(export_dir: Path | str) -> tuple[AlphabetModel, None, None]:
            return model, None, None

    original = tdi.v2.cli.AlphabetModel
    setattr(tdi.v2.cli, "AlphabetModel", _StubLoader)
    try:
        yield
    finally:
        setattr(tdi.v2.cli, "AlphabetModel", original)


def _load_reference(v1_dir: Path) -> tuple[nn.Module, np.ndarray]:
    """Load the released encoder object and its centroid table."""
    # weights_only=False is required: encoder.pt stores a pickled nn.Sequential, not a state
    # dict. The file is a checked-in repo asset, not untrusted input. (decoder.pt is a
    # different case -- it references a long-gone `vae_training` module and will not
    # unpickle; the evaluation path never needs a decoder, so it is not loaded.)
    encoder = torch.load(v1_dir / "encoder.pt", map_location="cpu", weights_only=False)
    if not isinstance(encoder, nn.Module):
        raise TypeError(f"{v1_dir / 'encoder.pt'} did not unpickle to an nn.Module.")
    centroids = np.loadtxt(v1_dir / "states.txt", dtype=np.float32)
    if centroids.ndim != 2:
        raise ValueError(f"Expected a 2-D centroid table in states.txt, got {centroids.shape}.")
    return encoder, centroids


def _read_submat(path: Path) -> tuple[list[str], np.ndarray]:
    """Parse a substitution matrix file into (state letters, score matrix)."""
    with open(path) as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    names = lines[0].split()
    rows = [[int(value) for value in line.split()[1:]] for line in lines[1:]]
    return names, np.array(rows, dtype=np.int64)


def _compare_submat(derived_path: Path, published_path: Path) -> None:
    """Report how closely the derived substitution matrix reproduces the published one.

    This is the sharpest available check that the encoding and MI machinery in this repo
    behaves like the original: same inputs and same model should give near-identical scores.
    Exact equality is not expected -- the published matrix was fit on the full training
    alignment set, whereas this runs on the validation pairfile.
    """
    derived_names, derived = _read_submat(derived_path)
    published_names, published = _read_submat(published_path)

    if derived_names != published_names or derived.shape != published.shape:
        print(
            f"\nSubstitution matrix not comparable: derived {derived.shape} over "
            f"{''.join(derived_names)} vs published {published.shape} over "
            f"{''.join(published_names)}."
        )
        return

    flat_derived = derived.astype(np.float64).ravel()
    flat_published = published.astype(np.float64).ravel()
    correlation = float(np.corrcoef(flat_derived, flat_published)[0, 1])
    mean_abs_diff = float(np.abs(flat_derived - flat_published).mean())
    print(
        f"\nDerived vs published substitution matrix ({published_path}):\n"
        f"  pearson        {correlation:.4f}\n"
        f"  mean |diff|    {mean_abs_diff:.2f} score units\n"
        f"  A high correlation means this repo's encoding and scoring path reproduces the\n"
        f"  released result; a low one points at a feature or convention mismatch, not at\n"
        f"  the alphabet itself."
    )


def _write_run_shell(out_dir: Path, centroids: np.ndarray, n_states: int) -> None:
    """Write the run-shaped metadata that ``evaluate_runs.py`` reads when tabulating."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "centroids.npy", centroids)
    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                "input_dim": 10,
                "hidden_dim": 10,
                "z_dim": int(centroids.shape[1]),
                "n_states": n_states,
                "quantizer": "foldseek_v1",
                "levels": None,
                "loss": "none",
                "note": "Published Foldseek 3Di release from data/v1; raw (unstandardized) "
                "descriptors, no retraining.",
            },
            f,
            indent=2,
        )


def main() -> None:
    """Evaluate the released alphabet on the validation alignments and print the headline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/ablation/foldseek_v1", help="Output run directory.")
    parser.add_argument("--v1-dir", default="data/v1", help="Directory of released v1 assets.")
    parser.add_argument(
        "--processed-dir",
        default="data/processed/scop_ca5_v1",
        help="Processed dataset supplying the virtual-center provenance.",
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
    parser.add_argument(
        "--virt",
        type=float,
        nargs=3,
        default=None,
        help="Virtual center override; otherwise read from the dataset manifest.",
    )
    args = parser.parse_args()

    v1_dir = Path(args.v1_dir)
    out_dir = Path(args.out)

    encoder, centroids = _load_reference(v1_dir)
    n_states, z_dim = centroids.shape
    print(f"Loaded released encoder ({z_dim}-D latent) and {n_states} published centroids.")
    _write_run_shell(out_dir, centroids, n_states)

    model = FoldseekV1Alphabet(encoder, centroids)
    # Provenance: the reference must encode structures with the same geometry as the runs it
    # is compared against, so take the virtual center from the dataset that produced them.
    virtual_center, max_ca_dist = _read_provenance(Path(args.processed_dir))
    model.virtual_center = list(args.virt) if args.virt is not None else virtual_center
    model.max_ca_dist = max_ca_dist
    if model.virtual_center is None:
        parser.error("No virtual center in the dataset manifest; pass --virt explicitly.")

    with _stub_export_loader(model):
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
    capacity = math.log2(n_states)
    entropy_bits = report["normalized_entropy"] * capacity
    print(
        f"\nFoldseek v1 reference (K={n_states}, published weights, no retraining):\n"
        f"  mi      {report['mi']:.4f} bits\n"
        f"  mi_tot  {report['mi_tot']:.4f} bits   ({100 * report['mi_tot'] / capacity:.1f}% of "
        f"log2(K))\n"
        f"  eff_K   {2**entropy_bits:.2f}   dead_state_fraction {report['dead_state_fraction']:.3f}"
    )

    published_submat = v1_dir / "sub_score.mat"
    if published_submat.exists():
        _compare_submat(out_dir / "eval" / "submat.txt", published_submat)

    print(
        "\nThis is the upper reference point. Trained arms above it beat the published\n"
        "alphabet on this benchmark; arms below it do not, however they compare to each other."
    )


if __name__ == "__main__":
    main()
