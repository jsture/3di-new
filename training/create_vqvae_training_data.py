"""Script to generate VQ-VAE training data from alignments of PDB files."""

import argparse
import os
from pathlib import Path
import random
from typing import Dict, Tuple, List
import numpy as np

import extract_pdb_features
import util

# Feature cache to avoid recomputing descriptors for the same PDB file
# maps PDB path to a tuple of (features, valid_mask)
FEATURE_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}


def encoder_features(
    pdb_path: str, virt_cb: Tuple[float, float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate 3D descriptors for each residue of a PDB file.

    Args:
        pdb_path: Absolute or relative path to the PDB file.
        virt_cb: Virtual center coordinate offset parameters (alpha, beta, d).

    Returns:
        A tuple of (vae_features, valid_mask):
            - vae_features: Descriptors shape (N, 10) including local angles and seq log distance.
            - valid_mask: Boolean mask of valid residues (shape: (N,)).
    """
    cached = FEATURE_CACHE.get(pdb_path)
    if cached is not None:
        return cached

    coords, valid_mask = extract_pdb_features.get_coords_from_pdb(
        pdb_path, full_backbone=True
    )
    coords = extract_pdb_features.move_CB(coords, virt_cb=virt_cb)
    partner_idx = extract_pdb_features.find_nearest_residues(coords, valid_mask)
    features, valid_mask2 = extract_pdb_features.calc_angles_forloop(
        coords, partner_idx, valid_mask
    )

    seq_dist = (partner_idx - np.arange(len(partner_idx)))[:, np.newaxis]
    log_dist = np.sign(seq_dist) * np.log(np.abs(seq_dist) + 1)

    vae_features = np.hstack([features, log_dist])
    FEATURE_CACHE[pdb_path] = vae_features, valid_mask2

    return vae_features, valid_mask2


def align_features(
    pdb_dir: str,
    virtual_center: Tuple[float, float, float],
    sid1: str,
    sid2: str,
    cigar_string: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return aligned descriptors for a given alignment between two PDBs.

    Args:
        pdb_dir: Directory where PDB files are stored.
        virtual_center: Virtual center offsets (alpha, beta, d).
        sid1: Structural ID of first protein.
        sid2: Structural ID of second protein.
        cigar_string: The alignment CIGAR string mapping residues.

    Returns:
        A tuple of (feat_x, feat_y):
            - feat_x: Aligned features of protein 1 (shape: (M, 10)).
            - feat_y: Aligned features of protein 2 (shape: (M, 10)).
    """
    idx_1, idx_2 = util.parse_cigar(cigar_string).T

    feat1, mask1 = encoder_features(
        os.path.join(pdb_dir, sid1), virtual_center
    )
    feat2, mask2 = encoder_features(
        os.path.join(pdb_dir, sid2), virtual_center
    )

    valid_mask = mask1[idx_1] & mask2[idx_2]
    idx_1 = idx_1[valid_mask]
    idx_2 = idx_2[valid_mask]

    x = np.vstack([feat1[idx_1], feat2[idx_2]])
    y = np.vstack([feat2[idx_2], feat1[idx_1]])
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract features from aligned structure pairs to compile VQ-VAE training data."
    )
    parser.add_argument("pdb_dir", type=str, help="Directory containing PDB files.")
    parser.add_argument(
        "pairfile", type=str, help="Path to structural alignment pairfile."
    )
    parser.add_argument("alpha", type=float, help="Virtual center offset angle alpha.")
    parser.add_argument("beta", type=float, help="Virtual center offset angle beta.")
    parser.add_argument("d", type=float, help="Virtual center offset distance d.")
    parser.add_argument(
        "out", type=str, help="Output path for the .npy training data file."
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent / "data"
    pdbs_train_path = data_dir / "pdbs_train.txt"

    with open(pdbs_train_path, "r") as file:
        pdbs_train = set(file.read().splitlines())

    # Find alignments between PDBs of the training set
    alignments: List[Tuple[str, str, str]] = []
    with open(args.pairfile, "r") as file:
        for line in file:
            parts = line.rstrip("\n").split()
            if len(parts) >= 3:
                sid1, sid2, cigar_string = parts[0], parts[1], parts[2]
                if sid1 in pdbs_train and sid2 in pdbs_train:
                    alignments.append((sid1, sid2, cigar_string))

    # Shuffle to ensure reproducibility
    random.Random(123).shuffle(alignments)

    virtual_center = (args.alpha, args.beta, args.d)

    xy: List[Tuple[np.ndarray, np.ndarray]] = []
    for sid1, sid2, cigar_string in alignments:
        xy.append(
            align_features(
                args.pdb_dir, virtual_center, sid1, sid2, cigar_string
            )
        )

    # Compile and write features to disk
    if xy:
        x_feat = np.vstack([x for x, y in xy])
        y_feat = np.vstack([y for x, y in xy])
        idx = np.arange(len(x_feat))
        np.random.RandomState(123).shuffle(idx)

        np.save(args.out, np.dstack([x_feat[idx], y_feat[idx]]))
    else:
        print("Warning: No alignments matched training PDB set. Output not written.")


if __name__ == "__main__":
    main()
