"""
MirrorFold — Mirror-Image Protein Structure Analysis
======================================================

Predict and compare structures of natural (L-amino acid) proteins
versus their mirror-image (D-amino acid) counterparts.

Background
-----------
All known life uses L-amino acids.  Mirror life — if it exists —
would use D-amino acids.  D-proteins are also of tremendous
therapeutic interest: they resist proteases, evade the immune
system, and can have dramatically extended half-lives.

This module provides:

* **mirror** — L→D sequence and coordinate transformations
* **predictor** — Structure prediction (ESMFold local/API/mock)
* **compare** — Structural alignment, RMSD, TM-score
* **analysis** — Biophysical property calculations
* **therapeutic** — D-protein therapeutic potential assessment
* **viz** — Interactive 3D visualisation and Ramachandran plots

Quick start
-----------
>>> from modules.mirrorfold import predict_pair, compare_structures
>>> l_pred, d_pred = predict_pair("YGGFL", allow_mock=True)
>>> report = compare_structures(l_pred, d_pred)
>>> print(f"RMSD: {report.rmsd:.2f} Å, TM: {report.tm_score:.3f}")
"""

from __future__ import annotations

# Data models
from .models import (
    StructurePrediction,
    ResidueComparison,
    ComparisonReport,
    PropertyProfile,
    TherapeuticAssessment,
)

# Mirror transformations
from .mirror import (
    mirror_sequence,
    is_valid_sequence,
    reflect_coordinates,
    get_l_smiles,
    get_d_smiles,
    verify_mirror_smiles,
)

# Structure prediction
from .predictor import (
    predict_structure,
    predict_pair,
    fetch_pdb,
    extract_sequence_from_pdb,
)

# Structural comparison
from .compare import (
    compare_structures,
    superimpose,
    calculate_tm_score,
    assign_secondary_structure,
)

# Property analysis
from .analysis import (
    compute_property_profile,
    compare_properties,
    molecular_weight,
    net_charge,
    isoelectric_point,
    hydrophobic_fraction,
)

# Therapeutic assessment
from .therapeutic import (
    assess_therapeutic_potential,
    assess_protease_resistance,
    assess_immunogenicity,
)

__all__ = [
    # Models
    "StructurePrediction",
    "ResidueComparison",
    "ComparisonReport",
    "PropertyProfile",
    "TherapeuticAssessment",
    # Mirror
    "mirror_sequence",
    "is_valid_sequence",
    "reflect_coordinates",
    "get_l_smiles",
    "get_d_smiles",
    "verify_mirror_smiles",
    # Predictor
    "predict_structure",
    "predict_pair",
    "fetch_pdb",
    "extract_sequence_from_pdb",
    # Compare
    "compare_structures",
    "superimpose",
    "calculate_tm_score",
    "assign_secondary_structure",
    # Analysis
    "compute_property_profile",
    "compare_properties",
    "molecular_weight",
    "net_charge",
    "isoelectric_point",
    "hydrophobic_fraction",
    # Therapeutic
    "assess_therapeutic_potential",
    "assess_protease_resistance",
    "assess_immunogenicity",
]
