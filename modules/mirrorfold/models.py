"""
MirrorFold Data Models
=======================

Pydantic models for structure predictions, comparisons, and
therapeutic assessments of L- and D-protein pairs.

Chirality in proteins
---------------------
Amino acids (except glycine) have a chiral α-carbon — a carbon
bonded to four different groups: –NH₂, –COOH, –H, and –R (side
chain).  The two mirror-image forms are called **L** (levo) and
**D** (dextro):

* **L-amino acids** — used by all known life on Earth.
* **D-amino acids** — the mirror image.  Same atoms, same bonds,
  but arranged as a non-superimposable reflection.

A protein built entirely from D-amino acids folds into the
mirror image of the L-protein's structure.  This has profound
implications for drug design and biosecurity.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class StructurePrediction(BaseModel):
    """A predicted 3D structure for a protein sequence.

    Contains the PDB-format string, per-residue confidence
    scores, and metadata about the prediction method.

    Attributes
    ----------
    sequence : str
        The amino-acid sequence (one-letter codes).
    chirality : str
        ``"L"`` for natural proteins, ``"D"`` for mirror-image.
    pdb_string : str
        The predicted structure in PDB format.
    plddt_scores : list[float]
        Per-residue pLDDT (predicted Local Distance Difference
        Test) confidence scores.  Range 0–100; >90 is very high
        confidence, 70–90 is confident, 50–70 is low, <50 is
        very low / likely disordered.
    mean_plddt : float
        Average pLDDT across all residues.
    prediction_method : str
        How the structure was predicted: ``"esmfold_local"``,
        ``"esmfold_api"``, or ``"mock"`` (for testing).
    """

    sequence: str
    chirality: str = Field(pattern=r"^[LD]$")
    pdb_string: str
    plddt_scores: list[float] = Field(default_factory=list)
    mean_plddt: float = 0.0
    prediction_method: str = "esmfold_local"


class ResidueComparison(BaseModel):
    """Per-residue structural comparison between L and D forms.

    Attributes
    ----------
    residue_index : int
        Zero-based residue index.
    residue_name : str
        One-letter amino acid code.
    ca_distance : float
        Cα–Cα distance (Å) after superposition.
    l_secondary : str
        Secondary structure assignment for L-form
        (``"H"`` helix, ``"E"`` sheet, ``"C"`` coil).
    d_secondary : str
        Secondary structure assignment for D-form.
    """

    residue_index: int
    residue_name: str
    ca_distance: float
    l_secondary: str = "C"
    d_secondary: str = "C"


class ComparisonReport(BaseModel):
    """Full structural comparison between L- and D-protein forms.

    Attributes
    ----------
    l_structure : StructurePrediction
        Predicted L-protein structure.
    d_structure : StructurePrediction
        Predicted D-protein structure (via reversed-sequence proxy).
    rmsd : float
        Root-mean-square deviation (Å) of aligned Cα atoms.
        Lower is more similar.  <2 Å is a good match.
    tm_score : float
        Template Modelling score (0–1).  >0.5 indicates the same
        fold; >0.7 is a close structural match.  More robust than
        RMSD for comparing protein folds because it normalises by
        length and down-weights large local deviations.
    per_residue_distances : list[float]
        Cα–Cα distance (Å) for each residue after alignment.
    secondary_structure_match : float
        Fraction of residues with the same secondary structure
        assignment in both L and D forms (0–1).
    conserved_regions : list[tuple[int, int]]
        Residue index ranges where L and D are structurally similar
        (Cα distance < 2 Å).
    divergent_regions : list[tuple[int, int]]
        Residue index ranges where L and D structures diverge
        (Cα distance ≥ 2 Å).
    aligned_length : int
        Number of residues that were aligned.
    """

    l_structure: StructurePrediction
    d_structure: StructurePrediction
    rmsd: float = 0.0
    tm_score: float = 0.0
    per_residue_distances: list[float] = Field(default_factory=list)
    secondary_structure_match: float = 0.0
    conserved_regions: list[tuple[int, int]] = Field(default_factory=list)
    divergent_regions: list[tuple[int, int]] = Field(default_factory=list)
    aligned_length: int = 0


class PropertyProfile(BaseModel):
    """Biophysical properties of a protein structure.

    Attributes
    ----------
    sequence : str
        Amino acid sequence.
    chirality : str
        ``"L"`` or ``"D"``.
    mean_plddt : float
        Average pLDDT (proxy for predicted stability).
    estimated_stability : str
        Qualitative stability: ``"high"``, ``"moderate"``, ``"low"``.
    net_charge : float
        Net charge at pH 7.4 (physiological).
    isoelectric_point : float
        pH at which the protein has zero net charge.
    molecular_weight : float
        Molecular weight in Daltons.
    hydrophobic_fraction : float
        Fraction of residues that are hydrophobic.
    n_surface_residues : int
        Estimated number of surface-exposed residues.
    n_buried_residues : int
        Estimated number of buried residues.
    """

    sequence: str
    chirality: str = "L"
    mean_plddt: float = 0.0
    estimated_stability: str = "moderate"
    net_charge: float = 0.0
    isoelectric_point: float = 7.0
    molecular_weight: float = 0.0
    hydrophobic_fraction: float = 0.0
    n_surface_residues: int = 0
    n_buried_residues: int = 0


class TherapeuticAssessment(BaseModel):
    """D-protein therapeutic viability assessment.

    Evaluates whether the mirror-image (D-amino acid) version of
    a protein is viable as a therapeutic molecule, considering
    fold conservation, protease resistance, and immunogenicity.

    Attributes
    ----------
    protein_name : str
        Human-readable protein name.
    pdb_id : str or None
        PDB identifier if the protein was fetched from PDB.
    comparison : ComparisonReport
        Full L vs D structural comparison.
    l_properties : PropertyProfile
        Biophysical properties of the L-form.
    d_properties : PropertyProfile
        Biophysical properties of the D-form.
    binding_pocket_conserved : bool
        Whether the predicted binding pocket geometry is conserved
        between L and D forms.
    estimated_protease_resistance : str
        ``"high"`` — D-proteins are intrinsically resistant to
        all natural proteases because protease active sites are
        stereospecific for L-peptide bonds.
    estimated_immunogenicity : str
        ``"very low"`` — D-proteins cannot be processed by the
        proteasome or presented by MHC class I/II molecules,
        making them essentially invisible to the adaptive immune
        system.
    therapeutic_viability : str
        Overall assessment: ``"promising"``, ``"possible"``,
        or ``"unlikely"``.
    rationale : str
        Human-readable explanation of the assessment.
    """

    protein_name: str = ""
    pdb_id: Optional[str] = None
    comparison: ComparisonReport
    l_properties: PropertyProfile
    d_properties: PropertyProfile
    binding_pocket_conserved: bool = False
    estimated_protease_resistance: str = "high"
    estimated_immunogenicity: str = "very low"
    therapeutic_viability: str = "possible"
    rationale: str = ""
