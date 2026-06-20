"""Feature extraction: parse PDB files and compute 3Di structural descriptors for v2.

This module provides geometric functions to calculate angles and distances from PDB structures
to construct 3Di structural alphabet descriptors.
"""

import sys

import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Residue import Residue
from numpy.linalg import norm
from scipy.spatial.distance import cdist

# Standard distance between C_alpha and C_beta in Angstroms
DISTANCE_ALPHA_BETA: float = 1.5336


def approx_c_beta_position(
    c_alpha: np.ndarray, n: np.ndarray, c_carboxyl: np.ndarray
) -> np.ndarray:
    """Approximate C beta position from C alpha, N and C positions.

    Assumes the four ligands of the C alpha form a regular tetrahedron.

    Args:
        c_alpha: 3D coordinates of C_alpha atom (shape: (3,)).
        n: 3D coordinates of N atom (shape: (3,)).
        c_carboxyl: 3D coordinates of C (carboxyl) atom (shape: (3,)).

    Returns:
        3D coordinates of approximated C_beta atom (shape: (3,)).
    """
    # Compute normalized bond vectors
    v1 = c_carboxyl - c_alpha
    v1 = v1 / norm(v1)
    v2 = n - c_alpha
    v2 = v2 / norm(v2)

    # Compute orthonormal basis vectors for the CB plane
    b1 = v2 + (1.0 / 3.0) * v1
    b2 = np.cross(v1, b1)

    u1 = b1 / norm(b1)
    u2 = b2 / norm(b2)

    # Reconstruct coordinate vector pointing to CB position
    v4 = -(1.0 / 3.0) * v1 + np.sqrt(8.0) / 3.0 * norm(v1) * (
        -(1.0 / 2.0) * u1 - np.sqrt(3.0) / 2.0 * u2
    )

    return c_alpha + DISTANCE_ALPHA_BETA * v4


def get_atom_coordinates(
    chain: list[Residue], verbose: bool = False, full_backbone: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Get CA/CB coordinates from list of biopython residues.

    C betas from GLY are approximated.

    Args:
        chain: List of Bio.PDB Residue objects.
        verbose: If True, prints warnings for missing atoms.
        full_backbone: If True, also returns C and N coordinates.

    Returns:
        A tuple of (coords, valid_mask):
            - coords: np.ndarray of shape (n_residues, 6) or (n_residues, 12).
            - valid_mask: np.ndarray of boolean values (shape: (n_residues,)).
    """
    n_res = len(chain)
    n_cols = 12 if full_backbone else 6
    coords = np.full((n_res, n_cols), np.nan, dtype=np.float32)

    for i, res in enumerate(chain):
        is_hetatm = len(res.id[0].strip())
        if is_hetatm:
            continue

        # Find Alpha carbon coordinate
        ca_atoms = [atom for atom in res if atom.name == "CA"]
        if len(ca_atoms) != 1:
            if verbose:
                print(f"No CA found [{i}]")
        else:
            coords[i, 0:3] = ca_atoms[0].coord

        # Find or approximate Beta carbon coordinate
        cb_atoms = [atom for atom in res if atom.name == "CB"]
        if res.resname != "GLY" and cb_atoms:
            # Use the first CB (handles alternate-location duplicates instead of dropping).
            coords[i, 3:6] = cb_atoms[0].coord
        else:
            # Approximate CB for Glycine or missing CB residues
            n_atoms = [atom for atom in res if atom.name == "N"]
            co_atoms = [atom for atom in res if atom.name == "C"]
            if len(ca_atoms) != 1 or len(n_atoms) != 1 or len(co_atoms) != 1:
                if verbose:
                    print(f"Failed to approx CB ({ca_atoms}, {n_atoms}, {co_atoms})")
            else:
                cb_coord = approx_c_beta_position(
                    ca_atoms[0].coord, n_atoms[0].coord, co_atoms[0].coord
                )
                coords[i, 3:6] = cb_coord

        # Optional: Save additional backbone coordinates for verification/superposition
        if full_backbone:
            n_atoms = [atom for atom in res if atom.name == "N"]
            co_atoms = [atom for atom in res if atom.name == "C"]
            if len(n_atoms) == 1 and len(co_atoms) == 1:
                coords[i, 6:9] = n_atoms[0].coord
                coords[i, 9:12] = co_atoms[0].coord

    # Create mask indicating valid residues (no NaN values in the row)
    valid_mask = np.asarray(~np.any(np.isnan(coords), axis=1))
    return coords, valid_mask


def distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances between two coordinate sets.

    Args:
        a: Array of shape (M, D).
        b: Array of shape (N, D).

    Returns:
        Distance matrix of shape (M, N).
    """
    # cdist avoids materializing the (M, N, D) broadcast difference, cutting memory and
    # time on large domains; numerically equivalent to the sqrt-of-squared-sum form.
    return cdist(a, b)


def find_nearest_residues(
    coords: np.ndarray,
    valid_mask: np.ndarray,
    k: int = 1,
    return_dist: bool = False,
    min_seq_dist: int | None = 1,
    fall_back_dist: float = 10.0,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Find indices of the k-th nearest neighbors comparing C_beta distances.

    Args:
        coords: Coordinates array of shape (N, 6) or (N, 12).
        valid_mask: Boolean mask indicating valid residues of shape (N,).
        k: Neighbor index to retrieve (1 for nearest, etc.).
        return_dist: If True, returns neighbor distances alongside indices.
        min_seq_dist: Minimum sequence separation between matching residues.
        fall_back_dist: Distance threshold for lifting sequence separation constraints.

    Returns:
        Indices of nearest residues (shape: (N,)), and optionally the distances.
    """
    assert not np.isnan(coords[valid_mask, 3:6]).any()
    dist = distance_matrix(coords[:, 3:6], coords[:, 3:6])

    # Mask self-distances and invalid coordinate indices
    dist[np.eye(dist.shape[0], dtype=bool)] = np.inf
    dist[~valid_mask, :] = np.inf
    dist[:, ~valid_mask] = np.inf
    dist[:, 0] = np.inf
    dist[0, :] = np.inf
    dist[:, -1] = np.inf
    dist[-1, :] = np.inf

    # Residues whose entire distance column is inf after self/invalid/boundary masking have
    # no valid partner (tiny/degenerate structures); argmin would return a bogus index 0.
    # Capture this here, before any min_seq_dist offset masking mutates the columns, so a
    # legitimate fall-back partner is never misread as having no partner.
    no_partner = np.isinf(dist).all(axis=0)

    # Filter by minimum sequence distance separation if required
    if min_seq_dist is not None and min_seq_dist != 1:
        n = dist.shape[0]

        j_no_min_seq = dist.argmin(axis=0)

        for offset in range(-min_seq_dist + 1, min_seq_dist):
            i_idx, j_idx = np.where(np.eye(n, k=offset, dtype=bool))
            dist[i_idx, j_idx] = np.inf

        j = dist.argmin(axis=0)
        fall_back_mask = dist.min(axis=0) >= fall_back_dist
        j[fall_back_mask] = j_no_min_seq[fall_back_mask]
    else:
        current_k = k
        while current_k > 1:
            j = dist.argmin(axis=0)
            dist[j, np.arange(dist.shape[0])] = np.inf
            current_k -= 1
        j = dist.argmin(axis=0)

    # Mark partnerless residues with a -1 sentinel so calc_angles_forloop drops them
    # instead of silently pairing to residue 0.
    j = j.copy()
    j[no_partner] = -1

    if return_dist:
        return j, dist[j, np.arange(dist.shape[0])]
    return j


def unit_vec(v: np.ndarray) -> np.ndarray:
    """Calculate the unit vector of v.

    Args:
        v: 1D input array.

    Returns:
        Normalized vector.
    """
    return v / np.linalg.norm(v)


def calc_angles(coords: np.ndarray, i: int, j: int) -> np.ndarray:
    """Calculate the 9-dimensional 3Di descriptor between residues i and j.

    Args:
        coords: Coordinates array of shape (N, 6) or (N, 12).
        i: Source residue index.
        j: Target residue index.

    Returns:
        Feature vector of shape (9,).
    """
    ca = coords[:, 0:3]

    # Calculate direction unit vectors
    u_1 = unit_vec(ca[i] - ca[i - 1])
    u_2 = unit_vec(ca[i + 1] - ca[i])
    u_3 = unit_vec(ca[j] - ca[j - 1])
    u_4 = unit_vec(ca[j + 1] - ca[j])
    u_5 = unit_vec(ca[j] - ca[i])

    # Dot products for angles
    cos_phi_12 = u_1.dot(u_2)
    cos_phi_34 = u_3.dot(u_4)
    cos_phi_15 = u_1.dot(u_5)
    cos_phi_35 = u_3.dot(u_5)
    cos_phi_14 = u_1.dot(u_4)
    cos_phi_23 = u_2.dot(u_3)
    cos_phi_13 = u_1.dot(u_3)

    d = np.linalg.norm(ca[i] - ca[j])
    seq_dist = np.clip(j - i, -4, 4)

    return np.array(
        [
            cos_phi_12,
            cos_phi_34,
            cos_phi_15,
            cos_phi_35,
            cos_phi_14,
            cos_phi_23,
            cos_phi_13,
            d,
            seq_dist,
        ]
    )


def calc_angles_forloop(
    coords: np.ndarray, partner_idx: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate angles for all valid residues and their nearest partners.

    Args:
        coords: Coordinates array of shape (N, 6) or (N, 12).
        partner_idx: Array of nearest neighbor indices of shape (N,).
        valid_mask: Array mask of valid residues of shape (N,).

    Returns:
        A tuple of (features, new_valid_mask):
            - features: Descriptor array of shape (N, 9).
            - new_valid_mask: Boolean mask indicating which residues have valid features.
    """
    n_res = coords.shape[0]
    out = np.full((n_res, 9), np.nan, dtype=np.float32)

    # Exclude boundary residues which lack sequence neighbors for directions.
    # Note: find_nearest_residues forces the first/last columns to inf, so the partner
    # j is always in [1, n_res - 2]; this keeps j-1 / j+1 below in bounds.
    for i in range(1, n_res - 1):
        if valid_mask[i - 1] and valid_mask[i] and valid_mask[i + 1]:
            j = partner_idx[i]
            # -1 sentinel from find_nearest_residues: residue had no valid partner.
            if j < 0:
                continue
            if valid_mask[j + 1] and valid_mask[j - 1]:
                out[i] = calc_angles(coords, i, j)

    new_valid_mask = np.asarray(~np.isnan(out).any(axis=1))
    return out, new_valid_mask


def get_coords_from_pdb(path: str, full_backbone: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Read a PDB file and return CA and CB coordinates (+ optional N and C).

    Args:
        path: Path to the PDB file.
        full_backbone: If True, also extracts N and C atom coordinates.

    Returns:
        A tuple of (coords, valid_mask).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("None", path)
    assert structure is not None

    # Keep the first-model/first-chain selection, but make silent truncation observable:
    # multi-chain / multi-model inputs (mmCIF, AlphaFold, NMR ensembles) would otherwise
    # drop every chain/model past the first without any signal.
    models = list(structure.get_models())
    model = models[0]
    chains = list(model.get_chains())
    chain = chains[0]
    if len(models) > 1 or len(chains) > 1:
        print(
            f"Warning: {path}: {len(models)} model(s), {len(chains)} chain(s); "
            f"using model 0 chain {chain.id}",
            file=sys.stderr,
        )

    coords, valid_mask = get_atom_coordinates(
        list(chain.get_residues()), full_backbone=full_backbone
    )
    return coords, valid_mask


def move_CB(
    coords: np.ndarray,
    c_alpha_beta_distance_scale: float = 1.0,
    virt_cb: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Adjust C_beta coordinates based on a distance scale or virtual angle projections.

    Args:
        coords: Coordinates array of shape (N, 12).
        c_alpha_beta_distance_scale: Multiplier for moving CB along the CA-CB vector.
        virt_cb: Optional tuple of (alpha, beta, d) specifying spherical offsets.

    Returns:
        Adjusted coordinates array.
    """
    if c_alpha_beta_distance_scale != 1.0 and virt_cb is None:
        ca = coords[:, 0:3]
        cb = coords[:, 3:6]
        coords[:, 3:6] = (cb - ca) * c_alpha_beta_distance_scale + ca

    if virt_cb is not None:
        alpha_deg, beta_deg, d = virt_cb
        alpha = np.radians(alpha_deg)
        beta = np.radians(beta_deg)

        ca = coords[:, 0:3]
        cb = coords[:, 3:6]
        n_atm = coords[:, 6:9]

        v = cb - ca

        a = cb - ca
        b = n_atm - ca
        cross_prod = np.cross(a, b)
        # Floor the norm to avoid divide-by-zero on degenerate (collinear) geometry.
        cross_norm = np.maximum(np.linalg.norm(cross_prod, axis=1, keepdims=True), 1e-8)
        k = cross_prod / cross_norm

        v = (
            v * np.cos(alpha)
            + np.cross(k, v) * np.sin(alpha)
            + k * (k * v).sum(axis=1, keepdims=True) * (1 - np.cos(alpha))
        )

        k = (n_atm - ca) / np.linalg.norm(n_atm - ca, axis=1, keepdims=True)
        v = (
            v * np.cos(beta)
            + np.cross(k, v) * np.sin(beta)
            + k * (k * v).sum(axis=1, keepdims=True) * (1 - np.cos(beta))
        )

        coords[:, 3:6] = ca + v * d

    return coords
