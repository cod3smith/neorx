"""
Biophysical Property Analysis
==============================

Computes physicochemical properties for L- and D-protein forms.

All calculations are first-principles: they use the amino acid
sequence and (optionally) the predicted structure.  No external
databases are queried.

Net charge & isoelectric point
-------------------------------
At a given pH, each ionisable side chain has a fractional charge
determined by the Henderson–Hasselbalch equation:

    charge = q / (1 + 10^(q × (pH − pKa)))

where *q* = +1 for basic groups (Lys, Arg, His, N-terminus) and
*q* = −1 for acidic groups (Asp, Glu, Cys, Tyr, C-terminus).

The **isoelectric point** (pI) is the pH at which net charge = 0,
found by bisection search over pH 0–14.

D-protein note
---------------
D-proteins have **identical** net charge, pI, molecular weight, and
amino acid composition as their L-counterparts.  The chirality of
the backbone does not affect these scalar properties.  What *does*
differ is folding stability (reflected in pLDDT), surface topology,
and interaction with chiral biological macromolecules (enzymes,
antibodies).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .models import PropertyProfile, StructurePrediction

logger = logging.getLogger(__name__)


# ── Amino Acid Physical Constants ───────────────────────────────────

# Average molecular weights (Da) for the 20 standard amino acids
# (residue weight = amino acid weight − water)
AA_MW: dict[str, float] = {
    "G": 57.0519, "A": 71.0788, "V": 99.1326, "L": 113.1594,
    "I": 113.1594, "P": 97.1167, "F": 147.1766, "W": 186.2132,
    "M": 131.1926, "S": 87.0782, "T": 101.1051, "C": 103.1388,
    "Y": 163.1760, "H": 137.1411, "D": 115.0886, "E": 129.1155,
    "N": 114.1038, "Q": 128.1307, "K": 128.1741, "R": 156.1875,
}

# Water added once for full protein (N-terminal H + C-terminal OH)
WATER_MW = 18.01524

# pKa values for ionisable groups
PKA: dict[str, float] = {
    "D": 3.65,   # Asp side chain
    "E": 4.25,   # Glu side chain
    "C": 8.18,   # Cys side chain
    "Y": 10.07,  # Tyr side chain
    "H": 6.00,   # His side chain
    "K": 10.53,  # Lys side chain
    "R": 12.48,  # Arg side chain
}
PKA_NTERM = 9.69   # α-amino group
PKA_CTERM = 2.34   # α-carboxyl group

# Kyte–Doolittle hydrophobicity scale (higher = more hydrophobic)
HYDROPHOBICITY: dict[str, float] = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}


# ── Molecular Weight ────────────────────────────────────────────────

def molecular_weight(sequence: str) -> float:
    """Calculate protein molecular weight in Daltons.

    MW = Σ(residue weights) + H₂O

    Unknown residues are silently skipped.

    Parameters
    ----------
    sequence : str
        One-letter amino acid sequence.

    Returns
    -------
    float
        Molecular weight in Da.
    """
    mw = WATER_MW
    for aa in sequence.upper():
        mw += AA_MW.get(aa, 0.0)
    return round(mw, 2)


# ── Net Charge ──────────────────────────────────────────────────────

def net_charge(sequence: str, ph: float = 7.4) -> float:
    """Calculate net charge at a given pH.

    Uses the Henderson–Hasselbalch equation for each ionisable
    group.  At physiological pH 7.4, most Asp/Glu are
    deprotonated (−1) and most Lys/Arg are protonated (+1).

    Parameters
    ----------
    sequence : str
        One-letter amino acid sequence.
    ph : float
        pH value (default 7.4).

    Returns
    -------
    float
        Net charge at given pH.
    """
    charge = 0.0

    # N-terminus (basic, +1 when protonated)
    charge += 1.0 / (1.0 + 10 ** (ph - PKA_NTERM))

    # C-terminus (acidic, −1 when deprotonated)
    charge -= 1.0 / (1.0 + 10 ** (PKA_CTERM - ph))

    for aa in sequence.upper():
        if aa in ("K", "R", "H"):
            # Basic side chains: +1 when protonated
            charge += 1.0 / (1.0 + 10 ** (ph - PKA[aa]))
        elif aa in ("D", "E", "C", "Y"):
            # Acidic side chains: −1 when deprotonated
            charge -= 1.0 / (1.0 + 10 ** (PKA[aa] - ph))

    return round(charge, 3)


# ── Isoelectric Point ──────────────────────────────────────────────

def isoelectric_point(sequence: str, tol: float = 0.01) -> float:
    """Find the isoelectric point (pI) by bisection search.

    pI is the pH where the net charge equals zero.

    Parameters
    ----------
    sequence : str
        One-letter amino acid sequence.
    tol : float
        Convergence tolerance.

    Returns
    -------
    float
        Estimated isoelectric point.
    """
    lo, hi = 0.0, 14.0

    for _ in range(100):  # max iterations
        mid = (lo + hi) / 2.0
        q = net_charge(sequence, mid)

        if abs(q) < tol:
            return round(mid, 2)

        if q > 0:
            lo = mid  # too acidic, raise pH
        else:
            hi = mid  # too basic, lower pH

    return round((lo + hi) / 2.0, 2)


# ── Hydrophobic Fraction ────────────────────────────────────────────

def hydrophobic_fraction(sequence: str) -> float:
    """Fraction of residues that are hydrophobic (Kyte–Doolittle > 0).

    A rough proxy for the tendency of a protein to bury
    hydrophobic residues in its core.

    Parameters
    ----------
    sequence : str
        One-letter amino acid sequence.

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    if not sequence:
        return 0.0

    n_hydrophobic = sum(
        1 for aa in sequence.upper()
        if HYDROPHOBICITY.get(aa, 0.0) > 0.0
    )
    return round(n_hydrophobic / len(sequence), 4)


# ── Surface / Buried Residue Estimation ─────────────────────────────

def estimate_surface_buried(
    pdb_string: str,
    plddt_surface_threshold: float = 70.0,
) -> tuple[int, int]:
    """Estimate surface vs buried residues from pLDDT and geometry.

    This is a simplified estimation:
    * Residues whose Cα is within 7.5 Å of the centroid and have
      pLDDT ≥ 70 are classified as **buried**.
    * All others are classified as **surface**.

    This heuristic captures two observations:
    1. Well-predicted residues (high pLDDT) tend to be in the
       well-packed core.
    2. Residues far from the centroid are more likely to be on the
       surface.

    For a more rigorous estimate, use DSSP or SASA calculators.

    Parameters
    ----------
    pdb_string : str
        PDB-format string.
    plddt_surface_threshold : float
        pLDDT below this → classified as surface.

    Returns
    -------
    tuple[int, int]
        (n_surface, n_buried).
    """
    coords: list[list[float]] = []
    plddts: list[float] = []

    for line in pdb_string.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue

        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        bfactor = float(line[60:66])

        coords.append([x, y, z])
        plddts.append(bfactor)

    if not coords:
        return 0, 0

    coords_arr = np.array(coords)
    centroid = coords_arr.mean(axis=0)
    distances = np.sqrt(np.sum((coords_arr - centroid) ** 2, axis=1))
    median_dist = float(np.median(distances))

    n_surface = 0
    n_buried = 0

    for i in range(len(coords)):
        is_core = distances[i] < median_dist
        is_confident = plddts[i] >= plddt_surface_threshold

        if is_core and is_confident:
            n_buried += 1
        else:
            n_surface += 1

    return n_surface, n_buried


# ── Stability Estimation ───────────────────────────────────────────

def estimate_stability(mean_plddt: float) -> str:
    """Map mean pLDDT to a qualitative stability label.

    pLDDT reflects the model's confidence in each residue's
    position.  Higher confidence generally correlates with a
    more stable, well-folded structure.

    * pLDDT > 90  → "Very High"
    * pLDDT > 70  → "High"
    * pLDDT > 50  → "Moderate"
    * pLDDT ≤ 50  → "Low" (may indicate disorder)

    Parameters
    ----------
    mean_plddt : float
        Mean predicted LDDT confidence score.

    Returns
    -------
    str
        Stability label.
    """
    if mean_plddt > 90:
        return "Very High"
    elif mean_plddt > 70:
        return "High"
    elif mean_plddt > 50:
        return "Moderate"
    else:
        return "Low"


# ── Full Property Profile ──────────────────────────────────────────

def compute_property_profile(
    prediction: StructurePrediction,
    ph: float = 7.4,
) -> PropertyProfile:
    """Compute a complete biophysical property profile.

    Parameters
    ----------
    prediction : StructurePrediction
        A predicted structure with PDB string and pLDDT scores.
    ph : float
        pH for net charge calculation.

    Returns
    -------
    PropertyProfile
        Complete property profile.
    """
    seq = prediction.sequence
    mean_plddt_val = prediction.mean_plddt or 0.0

    n_surface, n_buried = estimate_surface_buried(prediction.pdb_string)

    return PropertyProfile(
        sequence=seq,
        chirality=prediction.chirality,
        mean_plddt=mean_plddt_val,
        estimated_stability=estimate_stability(mean_plddt_val),
        net_charge=net_charge(seq, ph),
        isoelectric_point=isoelectric_point(seq),
        molecular_weight=molecular_weight(seq),
        hydrophobic_fraction=hydrophobic_fraction(seq),
        n_surface_residues=n_surface,
        n_buried_residues=n_buried,
    )


# ── L vs D Property Comparison ─────────────────────────────────────

def compare_properties(
    l_profile: PropertyProfile,
    d_profile: PropertyProfile,
) -> dict:
    """Compare property profiles between L and D forms.

    D-proteins have identical charge, pI, MW, and hydrophobicity
    (these are chirality-independent).  Differences arise in
    structural stability (pLDDT) and surface topology.

    Parameters
    ----------
    l_profile : PropertyProfile
    d_profile : PropertyProfile

    Returns
    -------
    dict
        Summary of property differences.
    """
    return {
        "sequence_length": len(l_profile.sequence),
        "molecular_weight": l_profile.molecular_weight,
        "net_charge_L": l_profile.net_charge,
        "net_charge_D": d_profile.net_charge,
        "isoelectric_point": l_profile.isoelectric_point,
        "hydrophobic_fraction": l_profile.hydrophobic_fraction,
        "stability_L": l_profile.estimated_stability,
        "stability_D": d_profile.estimated_stability,
        "plddt_L": round(l_profile.mean_plddt, 1),
        "plddt_D": round(d_profile.mean_plddt, 1),
        "plddt_delta": round(l_profile.mean_plddt - d_profile.mean_plddt, 1),
        "surface_residues_L": l_profile.n_surface_residues,
        "surface_residues_D": d_profile.n_surface_residues,
        "buried_residues_L": l_profile.n_buried_residues,
        "buried_residues_D": d_profile.n_buried_residues,
        "chirality_independent_properties": [
            "net_charge", "isoelectric_point", "molecular_weight",
            "hydrophobic_fraction",
        ],
        "chirality_dependent_properties": [
            "plddt", "stability", "surface_topology",
            "protease_susceptibility", "immunogenicity",
        ],
    }
