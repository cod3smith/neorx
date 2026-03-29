"""
D-Protein Therapeutic Assessment
=================================

Evaluates the potential of D-protein (mirror-image) variants as
therapeutics.  This is one of the most exciting areas of mirror
biology — D-proteins have several inherent advantages:

Why D-protein therapeutics?
----------------------------
1. **Protease resistance** — Natural proteases (trypsin, chymotrypsin,
   pepsin, etc.) evolved to recognise L-amino acid peptide bonds.
   They literally cannot grip D-peptide bonds because the active site
   geometry is wrong.  D-proteins are essentially invisible to the
   body's proteolytic machinery.

2. **Immune evasion** — MHC class I and II molecules present peptide
   fragments to T cells.  These molecules bind peptides in a specific
   L-amino acid configuration.  D-peptides cannot be presented,
   making D-proteins practically non-immunogenic.

3. **Extended half-life** — Because D-proteins resist proteases and
   evade the immune system, they have dramatically longer serum
   half-lives than L-protein drugs.  An L-peptide that survives
   minutes in blood can last *days* as a D-peptide.

4. **Oral bioavailability** — Resistance to digestive proteases
   opens the possibility of oral administration, which is the
   holy grail for protein therapeutics.

The catch
----------
The D-protein must fold into the correct 3D structure — specifically,
the mirror image of the L-protein's fold.  If it folds correctly,
it will bind the **same** target (because the mirror-image binding
surface is complementary to the same binding pocket).

This module quantifies how likely a D-protein is to maintain its
fold and function, based on structural comparison.
"""

from __future__ import annotations

import logging
from typing import Optional

from .models import (
    ComparisonReport,
    PropertyProfile,
    TherapeuticAssessment,
)

logger = logging.getLogger(__name__)


# ── Protease Resistance Assessment ─────────────────────────────────

def assess_protease_resistance(chirality: str) -> str:
    """Estimate protease resistance based on chirality.

    This is straightforward: L-proteins are susceptible to all
    natural proteases.  D-proteins are resistant to essentially
    all of them.

    Parameters
    ----------
    chirality : str
        ``"L"`` or ``"D"``.

    Returns
    -------
    str
        Qualitative resistance level.
    """
    if chirality == "D":
        return "Very High"
    return "Normal (susceptible to natural proteases)"


# ── Immunogenicity Assessment ──────────────────────────────────────

def assess_immunogenicity(chirality: str) -> str:
    """Estimate immunogenicity based on chirality.

    MHC class I molecules bind peptide fragments of 8–10 residues
    in a groove that is stereospecific for L-amino acids.  MHC
    class II binds 13–25-mers, also stereospecifically.  Since
    D-peptides cannot fit these grooves, D-proteins evade
    adaptive immunity almost completely.

    Innate immunity (e.g. toll-like receptors) is less chirality-
    dependent, so some innate immune response is possible.

    Parameters
    ----------
    chirality : str
        ``"L"`` or ``"D"``.

    Returns
    -------
    str
        Qualitative immunogenicity level.
    """
    if chirality == "D":
        return "Very Low (MHC cannot present D-peptides)"
    return "Normal (subject to standard immune surveillance)"


# ── Binding Pocket Conservation ────────────────────────────────────

def assess_binding_pocket_conservation(
    comparison: ComparisonReport,
    pocket_residues: Optional[list[int]] = None,
) -> bool:
    """Determine whether the binding pocket is structurally conserved.

    If specific pocket residues are known, we check whether those
    residues fall in conserved regions.  Otherwise, we use the
    global TM-score as a proxy:

    * TM-score > 0.5 → binding pocket likely conserved
    * TM-score ≤ 0.5 → substantial structural rearrangement

    Parameters
    ----------
    comparison : ComparisonReport
        Structural comparison between L and D forms.
    pocket_residues : list[int], optional
        Indices of residues known to form the binding pocket.

    Returns
    -------
    bool
        Whether the binding pocket is conserved.
    """
    if comparison.tm_score is None:
        return False

    if pocket_residues and comparison.per_residue_distances:
        # Check if the average distance of pocket residues < 2 Å
        pocket_dists = [
            comparison.per_residue_distances[i]
            for i in pocket_residues
            if i < len(comparison.per_residue_distances)
        ]
        if pocket_dists:
            mean_pocket_dist = sum(pocket_dists) / len(pocket_dists)
            return mean_pocket_dist < 2.0

    # Global proxy
    return comparison.tm_score > 0.5


# ── Therapeutic Viability Score ────────────────────────────────────

def therapeutic_viability_score(
    tm_score: float,
    pocket_conserved: bool,
    sequence_length: int,
) -> str:
    """Compute overall therapeutic viability.

    Considers:
    * Fold conservation (TM-score)
    * Binding pocket geometry
    * Protein size (smaller D-proteins are easier to synthesise)

    Rating scale:
    * "Excellent"  — TM > 0.7 + pocket conserved + small size
    * "Good"       — TM > 0.5 + pocket conserved
    * "Moderate"   — TM > 0.5 but pocket uncertain
    * "Poor"       — TM ≤ 0.5 or synthesis impractical

    Parameters
    ----------
    tm_score : float
    pocket_conserved : bool
    sequence_length : int

    Returns
    -------
    str
        Viability category.
    """
    # Very large proteins are currently impractical for D-protein
    # synthesis (solid-phase peptide synthesis limit ~50-100 AAs,
    # native chemical ligation can extend to ~200 AAs)
    synthesis_feasible = sequence_length <= 200

    if tm_score > 0.7 and pocket_conserved and synthesis_feasible:
        return "Excellent"
    elif tm_score > 0.7 and pocket_conserved:
        return "Good (but synthesis challenging for this size)"
    elif tm_score > 0.5 and pocket_conserved:
        return "Good"
    elif tm_score > 0.5:
        return "Moderate"
    else:
        return "Poor"


# ── Rationale Generation ──────────────────────────────────────────

def _generate_rationale(
    tm_score: float,
    pocket_conserved: bool,
    viability: str,
    sequence_length: int,
    plddt_l: float,
    plddt_d: float,
) -> str:
    """Generate a human-readable rationale for the assessment.

    Parameters
    ----------
    tm_score : float
    pocket_conserved : bool
    viability : str
    sequence_length : int
    plddt_l, plddt_d : float

    Returns
    -------
    str
        Multi-line rationale text.
    """
    lines = []

    # Fold conservation
    if tm_score > 0.7:
        lines.append(
            f"FOLD CONSERVATION: Strong (TM-score = {tm_score:.3f}). "
            f"The D-protein is predicted to adopt essentially the same "
            f"fold as the L-protein."
        )
    elif tm_score > 0.5:
        lines.append(
            f"FOLD CONSERVATION: Moderate (TM-score = {tm_score:.3f}). "
            f"The D-protein adopts the same general fold with some "
            f"local structural differences."
        )
    else:
        lines.append(
            f"FOLD CONSERVATION: Weak (TM-score = {tm_score:.3f}). "
            f"The D-protein may not fold into the same topology."
        )

    # Binding pocket
    if pocket_conserved:
        lines.append(
            "BINDING POCKET: Conserved. The functional binding "
            "surface geometry is maintained in the D-form."
        )
    else:
        lines.append(
            "BINDING POCKET: May be altered. The D-protein's binding "
            "surface may differ from the L-form."
        )

    # Prediction confidence
    plddt_delta = abs(plddt_l - plddt_d)
    if plddt_delta < 5:
        lines.append(
            f"PREDICTION CONFIDENCE: Both forms predicted with similar "
            f"confidence (ΔpLDDT = {plddt_delta:.1f})."
        )
    else:
        lines.append(
            f"PREDICTION CONFIDENCE: Significant confidence difference "
            f"(ΔpLDDT = {plddt_delta:.1f}). The less confident "
            f"prediction should be interpreted cautiously."
        )

    # Synthesis feasibility
    if sequence_length <= 50:
        lines.append(
            f"SYNTHESIS: Feasible ({sequence_length} residues). "
            f"Standard SPPS can produce D-proteins of this size."
        )
    elif sequence_length <= 200:
        lines.append(
            f"SYNTHESIS: Feasible with ligation ({sequence_length} "
            f"residues). Native chemical ligation or similar methods "
            f"are needed."
        )
    else:
        lines.append(
            f"SYNTHESIS: Challenging ({sequence_length} residues). "
            f"Current chemical synthesis methods are limited to "
            f"~200 residues for D-proteins."
        )

    # Inherent D-protein advantages
    lines.append(
        "PROTEASE RESISTANCE: Very High. D-peptide bonds are not "
        "recognised by natural proteases (trypsin, chymotrypsin, "
        "pepsin, etc.)."
    )
    lines.append(
        "IMMUNOGENICITY: Very Low. MHC class I and II molecules "
        "cannot present D-peptide fragments, effectively evading "
        "adaptive immunity."
    )

    # Overall verdict
    lines.append(f"OVERALL VIABILITY: {viability}.")

    return "\n\n".join(lines)


# ── Full Assessment Pipeline ───────────────────────────────────────

def assess_therapeutic_potential(
    protein_name: str,
    comparison: ComparisonReport,
    l_properties: PropertyProfile,
    d_properties: PropertyProfile,
    pdb_id: Optional[str] = None,
    pocket_residues: Optional[list[int]] = None,
) -> TherapeuticAssessment:
    """Run a full D-protein therapeutic assessment.

    Integrates structural comparison and property analysis to
    produce a comprehensive evaluation of whether a D-protein
    variant is a viable therapeutic candidate.

    Parameters
    ----------
    protein_name : str
        Human-readable name (e.g. "Insulin B chain").
    comparison : ComparisonReport
        Structural comparison between L and D forms.
    l_properties : PropertyProfile
        L-protein biophysical properties.
    d_properties : PropertyProfile
        D-protein biophysical properties.
    pdb_id : str, optional
        PDB identifier if available.
    pocket_residues : list[int], optional
        Residue indices forming the binding pocket.

    Returns
    -------
    TherapeuticAssessment
        Complete therapeutic assessment.
    """
    tm = comparison.tm_score or 0.0
    seq_len = len(comparison.l_structure.sequence)

    pocket_conserved = assess_binding_pocket_conservation(
        comparison, pocket_residues
    )

    viability = therapeutic_viability_score(tm, pocket_conserved, seq_len)

    rationale = _generate_rationale(
        tm_score=tm,
        pocket_conserved=pocket_conserved,
        viability=viability,
        sequence_length=seq_len,
        plddt_l=l_properties.mean_plddt,
        plddt_d=d_properties.mean_plddt,
    )

    logger.info(
        "Assessment for %s: viability=%s, TM=%.3f, pocket=%s",
        protein_name, viability, tm, pocket_conserved,
    )

    return TherapeuticAssessment(
        protein_name=protein_name,
        pdb_id=pdb_id,
        comparison=comparison,
        l_properties=l_properties,
        d_properties=d_properties,
        binding_pocket_conserved=pocket_conserved,
        estimated_protease_resistance=assess_protease_resistance("D"),
        estimated_immunogenicity=assess_immunogenicity("D"),
        therapeutic_viability=viability,
        rationale=rationale,
    )
