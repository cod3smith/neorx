"""
Structure Comparison — L vs D Proteins
========================================

Aligns and compares the predicted structures of L- and D-protein
forms using RMSD, TM-score, and secondary structure analysis.

RMSD vs TM-score
-----------------
* **RMSD** (Root Mean Square Deviation) is the most intuitive
  metric: the average Cα displacement in Ångströms after optimal
  superposition.  But RMSD has a flaw — a single badly-aligned
  loop can inflate the score even if the rest of the structure
  matches perfectly.

* **TM-score** (Template Modelling score) solves this by
  down-weighting large distances.  It ranges from 0 to 1:

  - **TM > 0.5** → same fold (statistically significant)
  - **TM > 0.7** → very similar structures
  - **TM > 0.9** → essentially identical

  Formula: TM = (1/L) Σ 1/(1 + (dᵢ/d₀)²)

  where L is target length, dᵢ is per-residue distance, and
  d₀ = 1.24 ∛(L − 15) − 1.8 is a length-dependent normaliser.

  Reference: Zhang & Skolnick (2004) *Proteins* 57:702-710.

Secondary structure comparison
-------------------------------
We assign each residue to one of three states based on backbone
torsion angles (φ, ψ) from the Ramachandran plot:

* **H** (helix): −160° < φ < −20° and −80° < ψ < 0°
* **E** (sheet): −180° < φ < −40° and 50° < ψ < 180°
* **C** (coil): everything else

For D-proteins, the Ramachandran plot is mirrored: the allowed
regions have **positive** φ values (the exact reflection of the
L-protein regions).  When comparing secondary structure between
L and D forms, we compare the topology (helix vs sheet vs coil)
regardless of the sign of the torsion angles.
"""

from __future__ import annotations

import io
import logging
import math
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from .models import (
    StructurePrediction,
    ComparisonReport,
    ResidueComparison,
)

logger = logging.getLogger(__name__)


# ── PDB Parsing Helpers ─────────────────────────────────────────────

def _extract_ca_coords(pdb_string: str) -> np.ndarray:
    """Extract Cα atom coordinates from a PDB string.

    Parameters
    ----------
    pdb_string : str
        PDB-format string.

    Returns
    -------
    ndarray [N, 3]
        Cα coordinates in Ångströms.
    """
    coords: list[list[float]] = []
    seen: set[int] = set()

    for line in pdb_string.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        res_seq = int(line[22:26].strip())
        if res_seq in seen:
            continue
        seen.add(res_seq)
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        coords.append([x, y, z])

    return np.array(coords, dtype=np.float64)


def _extract_backbone_atoms(pdb_string: str) -> dict[int, dict[str, np.ndarray]]:
    """Extract N, CA, C backbone atoms per residue.

    Returns a dict mapping residue index → {atom_name: coords}.
    Needed for torsion angle calculation.
    """
    backbone: dict[int, dict[str, np.ndarray]] = {}

    for line in pdb_string.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name not in ("N", "CA", "C"):
            continue
        res_seq = int(line[22:26].strip())
        coords = np.array([
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        ])
        if res_seq not in backbone:
            backbone[res_seq] = {}
        backbone[res_seq][atom_name] = coords

    return backbone


# ── Structural Alignment ────────────────────────────────────────────

def superimpose(
    coords_fixed: np.ndarray,
    coords_moving: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Optimal superposition using the Kabsch algorithm.

    Finds the rotation and translation that minimises RMSD
    between two sets of corresponding points.

    Parameters
    ----------
    coords_fixed : ndarray [N, 3]
        Reference coordinates (will not move).
    coords_moving : ndarray [N, 3]
        Mobile coordinates (will be rotated/translated).

    Returns
    -------
    coords_aligned : ndarray [N, 3]
        The moving coordinates after optimal superposition.
    rmsd : float
        RMSD in Ångströms.
    """
    assert len(coords_fixed) == len(coords_moving), (
        f"Coordinate count mismatch: {len(coords_fixed)} vs {len(coords_moving)}"
    )

    n = len(coords_fixed)

    # Centre both coordinate sets
    centroid_fixed = coords_fixed.mean(axis=0)
    centroid_moving = coords_moving.mean(axis=0)
    p = coords_fixed - centroid_fixed
    q = coords_moving - centroid_moving

    # Cross-covariance matrix
    H = q.T @ p

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])

    # Optimal rotation
    R = Vt.T @ sign_matrix @ U.T

    # Apply rotation and translation
    aligned = (q @ R.T) + centroid_fixed

    # RMSD
    diff = coords_fixed - aligned
    rmsd = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

    return aligned, rmsd


# ── TM-score ────────────────────────────────────────────────────────

def calculate_tm_score(
    coords_ref: np.ndarray,
    coords_aligned: np.ndarray,
    target_length: Optional[int] = None,
) -> float:
    """Calculate the TM-score between two aligned structures.

    TM-score is more robust than RMSD because it uses a
    length-dependent distance cutoff ``d₀`` that down-weights
    large per-residue deviations.

    Parameters
    ----------
    coords_ref : ndarray [N, 3]
        Reference Cα coordinates.
    coords_aligned : ndarray [N, 3]
        Aligned Cα coordinates (after superposition).
    target_length : int, optional
        Normalisation length.  Defaults to the number of residues.

    Returns
    -------
    float
        TM-score in [0, 1].
    """
    L = target_length or len(coords_ref)
    if L < 16:
        # d0 formula is unstable for very short proteins
        d0 = 0.5
    else:
        d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
        d0 = max(d0, 0.5)  # floor at 0.5 Å

    # Per-residue distances
    distances = np.sqrt(np.sum((coords_ref - coords_aligned) ** 2, axis=1))

    # TM-score sum
    tm_sum = np.sum(1.0 / (1.0 + (distances / d0) ** 2))
    tm_score = float(tm_sum / L)

    return min(tm_score, 1.0)


# ── Per-residue distances ───────────────────────────────────────────

def per_residue_distances(
    coords_ref: np.ndarray,
    coords_aligned: np.ndarray,
) -> list[float]:
    """Calculate per-residue Cα–Cα distances after alignment.

    Parameters
    ----------
    coords_ref : ndarray [N, 3]
    coords_aligned : ndarray [N, 3]

    Returns
    -------
    list[float]
        Distance in Å for each residue.
    """
    diffs = coords_ref - coords_aligned
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    return distances.tolist()


# ── Secondary Structure Assignment ──────────────────────────────────

def _dihedral_angle(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> float:
    """Calculate the dihedral angle (in degrees) defined by four points.

    Uses the atan2 formula for numerical stability.
    """
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)

    if norm_n1 < 1e-8 or norm_n2 < 1e-8:
        return 0.0

    n1 = n1 / norm_n1
    n2 = n2 / norm_n2

    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))

    return math.degrees(math.atan2(y, x))


def assign_secondary_structure(pdb_string: str) -> list[str]:
    """Assign secondary structure from backbone torsion angles.

    Uses a simplified Ramachandran classification:
    * **H** (helix): φ ∈ [−160, −20] and ψ ∈ [−80, 0]
    * **E** (sheet): φ ∈ [−180, −40] and ψ ∈ [50, 180]
    * **C** (coil): everything else

    For D-proteins, the φ/ψ distributions are mirrored (positive
    φ values).  We account for this by taking the absolute value
    of φ when classifying topology.

    Parameters
    ----------
    pdb_string : str
        PDB-format string.

    Returns
    -------
    list[str]
        Per-residue assignment: ``"H"``, ``"E"``, or ``"C"``.
    """
    backbone = _extract_backbone_atoms(pdb_string)
    residue_indices = sorted(backbone.keys())

    assignments: list[str] = []

    for i, res_idx in enumerate(residue_indices):
        atoms = backbone[res_idx]
        if not all(k in atoms for k in ("N", "CA", "C")):
            assignments.append("C")
            continue

        # Calculate φ: C(i-1) — N(i) — CA(i) — C(i)
        # Calculate ψ: N(i) — CA(i) — C(i) — N(i+1)
        phi = None
        psi = None

        if i > 0:
            prev_idx = residue_indices[i - 1]
            prev_atoms = backbone.get(prev_idx, {})
            if "C" in prev_atoms:
                phi = _dihedral_angle(
                    prev_atoms["C"], atoms["N"], atoms["CA"], atoms["C"]
                )

        if i < len(residue_indices) - 1:
            next_idx = residue_indices[i + 1]
            next_atoms = backbone.get(next_idx, {})
            if "N" in next_atoms:
                psi = _dihedral_angle(
                    atoms["N"], atoms["CA"], atoms["C"], next_atoms["N"]
                )

        # Classify using absolute φ (works for both L and D)
        if phi is not None and psi is not None:
            abs_phi = abs(phi)
            abs_psi = abs(psi)

            if 20 <= abs_phi <= 160 and abs_psi <= 80:
                assignments.append("H")
            elif 40 <= abs_phi <= 180 and 50 <= abs_psi <= 180:
                assignments.append("E")
            else:
                assignments.append("C")
        else:
            assignments.append("C")  # terminal residues

    return assignments


def _secondary_structure_match(ss_l: list[str], ss_d: list[str]) -> float:
    """Fraction of residues with matching secondary structure.

    Parameters
    ----------
    ss_l : list[str]
        L-protein secondary structure assignments.
    ss_d : list[str]
        D-protein secondary structure assignments.

    Returns
    -------
    float
        Match fraction in [0, 1].
    """
    n = min(len(ss_l), len(ss_d))
    if n == 0:
        return 0.0

    matches = sum(1 for i in range(n) if ss_l[i] == ss_d[i])
    return matches / n


# ── Region Identification ───────────────────────────────────────────

def _identify_regions(
    distances: list[float],
    threshold: float = 2.0,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Identify conserved and divergent residue regions.

    Parameters
    ----------
    distances : list[float]
        Per-residue Cα distances.
    threshold : float
        Distance threshold in Å.  Below = conserved, above = divergent.

    Returns
    -------
    conserved : list[tuple[int, int]]
        Residue index ranges that are structurally similar.
    divergent : list[tuple[int, int]]
        Residue index ranges that differ.
    """
    conserved: list[tuple[int, int]] = []
    divergent: list[tuple[int, int]] = []

    if not distances:
        return conserved, divergent

    # Group consecutive residues
    current_type = "conserved" if distances[0] < threshold else "divergent"
    start = 0

    for i in range(1, len(distances)):
        is_conserved = distances[i] < threshold
        new_type = "conserved" if is_conserved else "divergent"

        if new_type != current_type:
            region = (start, i - 1)
            if current_type == "conserved":
                conserved.append(region)
            else:
                divergent.append(region)
            start = i
            current_type = new_type

    # Final region
    region = (start, len(distances) - 1)
    if current_type == "conserved":
        conserved.append(region)
    else:
        divergent.append(region)

    return conserved, divergent


# ── Full Comparison Pipeline ────────────────────────────────────────

def compare_structures(
    l_prediction: StructurePrediction,
    d_prediction: StructurePrediction,
    distance_threshold: float = 2.0,
) -> ComparisonReport:
    """Compare L- and D-protein predicted structures.

    Performs:
    1. Extract Cα coordinates from both PDB strings
    2. Superimpose using the Kabsch algorithm
    3. Calculate RMSD and TM-score
    4. Compute per-residue distances
    5. Assign and compare secondary structure
    6. Identify conserved and divergent regions

    Parameters
    ----------
    l_prediction : StructurePrediction
        Predicted L-protein structure.
    d_prediction : StructurePrediction
        Predicted D-protein structure.
    distance_threshold : float
        Cα distance threshold (Å) for classifying conserved vs
        divergent regions.  Default 2.0 Å.

    Returns
    -------
    ComparisonReport
        Full structural comparison report.
    """
    # Extract Cα coordinates
    coords_l = _extract_ca_coords(l_prediction.pdb_string)
    coords_d = _extract_ca_coords(d_prediction.pdb_string)

    # Handle length mismatch (truncate to shorter)
    n = min(len(coords_l), len(coords_d))
    if n == 0:
        logger.warning("No Cα atoms found in one or both structures.")
        return ComparisonReport(
            l_structure=l_prediction,
            d_structure=d_prediction,
        )

    coords_l = coords_l[:n]
    coords_d = coords_d[:n]

    # Superimpose D onto L
    coords_d_aligned, rmsd = superimpose(coords_l, coords_d)

    # TM-score
    tm = calculate_tm_score(coords_l, coords_d_aligned, target_length=n)

    # Per-residue distances
    distances = per_residue_distances(coords_l, coords_d_aligned)

    # Secondary structure
    ss_l = assign_secondary_structure(l_prediction.pdb_string)
    ss_d = assign_secondary_structure(d_prediction.pdb_string)
    ss_match = _secondary_structure_match(ss_l[:n], ss_d[:n])

    # Conserved / divergent regions
    conserved, divergent = _identify_regions(distances, distance_threshold)

    logger.info(
        "Comparison: RMSD=%.2f Å, TM=%.3f, SS-match=%.1f%%, "
        "%d conserved / %d divergent regions.",
        rmsd, tm, ss_match * 100,
        len(conserved), len(divergent),
    )

    return ComparisonReport(
        l_structure=l_prediction,
        d_structure=d_prediction,
        rmsd=rmsd,
        tm_score=tm,
        per_residue_distances=distances,
        secondary_structure_match=ss_match,
        conserved_regions=conserved,
        divergent_regions=divergent,
        aligned_length=n,
    )
