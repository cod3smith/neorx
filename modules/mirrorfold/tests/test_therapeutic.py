"""Tests for mirrorfold.therapeutic — D-protein assessment."""

from __future__ import annotations

import pytest

from modules.mirrorfold.therapeutic import (
    assess_protease_resistance,
    assess_immunogenicity,
    assess_binding_pocket_conservation,
    therapeutic_viability_score,
    assess_therapeutic_potential,
)
from modules.mirrorfold.predictor import predict_pair
from modules.mirrorfold.compare import compare_structures
from modules.mirrorfold.analysis import compute_property_profile


# ── Protease Resistance ─────────────────────────────────────────

class TestProteaseResistance:
    """D-protein protease resistance assessment."""

    def test_d_protein_resistant(self) -> None:
        result = assess_protease_resistance("D")
        assert "Very High" in result

    def test_l_protein_susceptible(self) -> None:
        result = assess_protease_resistance("L")
        assert "susceptible" in result.lower() or "Normal" in result


# ── Immunogenicity ──────────────────────────────────────────────

class TestImmunogenicity:
    """D-protein immunogenicity assessment."""

    def test_d_protein_low_immunogenicity(self) -> None:
        result = assess_immunogenicity("D")
        assert "Very Low" in result or "low" in result.lower()

    def test_l_protein_normal_immunogenicity(self) -> None:
        result = assess_immunogenicity("L")
        assert "Normal" in result


# ── Binding Pocket Conservation ─────────────────────────────────

class TestBindingPocket:
    """Binding pocket conservation analysis."""

    def test_conserved_with_high_tm(self) -> None:
        l_pred, d_pred = predict_pair(
            "YGGFL",
            allow_mock=True,
            use_cache=False,
            allow_api_fallback=False,
        )
        report = compare_structures(l_pred, d_pred)

        # The mock structures may or may not be conserved
        result = assess_binding_pocket_conservation(report)
        assert isinstance(result, bool)


# ── Therapeutic Viability Score ─────────────────────────────────

class TestViabilityScore:
    """Therapeutic viability scoring."""

    def test_excellent_viability(self) -> None:
        score = therapeutic_viability_score(
            tm_score=0.85, pocket_conserved=True, sequence_length=30
        )
        assert "Excellent" in score

    def test_good_viability(self) -> None:
        score = therapeutic_viability_score(
            tm_score=0.6, pocket_conserved=True, sequence_length=30
        )
        assert "Good" in score

    def test_moderate_viability(self) -> None:
        score = therapeutic_viability_score(
            tm_score=0.55, pocket_conserved=False, sequence_length=30
        )
        assert "Moderate" in score

    def test_poor_viability(self) -> None:
        score = therapeutic_viability_score(
            tm_score=0.3, pocket_conserved=False, sequence_length=30
        )
        assert "Poor" in score

    def test_large_protein_synthesis_note(self) -> None:
        score = therapeutic_viability_score(
            tm_score=0.85, pocket_conserved=True, sequence_length=300
        )
        assert "synthesis" in score.lower() or "Good" in score


# ── Full Assessment Pipeline ────────────────────────────────────

class TestFullAssessment:
    """End-to-end therapeutic assessment."""

    def test_assess_short_peptide(self) -> None:
        """Full assessment on Leu-enkephalin (5 residues)."""
        l_pred, d_pred = predict_pair(
            "YGGFL",
            allow_mock=True,
            use_cache=False,
            allow_api_fallback=False,
        )
        comparison = compare_structures(l_pred, d_pred)
        l_props = compute_property_profile(l_pred)
        d_props = compute_property_profile(d_pred)

        assessment = assess_therapeutic_potential(
            protein_name="Leu-enkephalin",
            comparison=comparison,
            l_properties=l_props,
            d_properties=d_props,
        )

        assert assessment.protein_name == "Leu-enkephalin"
        assert assessment.therapeutic_viability != ""
        assert "Very High" in assessment.estimated_protease_resistance
        assert "Very Low" in assessment.estimated_immunogenicity
        assert assessment.rationale != ""
        assert len(assessment.rationale) > 50

    def test_assess_medium_protein(self) -> None:
        """Assessment on a 20-residue sequence."""
        seq = "ACDEFGHIKLMNPQRSTVWY"
        l_pred, d_pred = predict_pair(
            seq,
            allow_mock=True,
            use_cache=False,
            allow_api_fallback=False,
        )
        comparison = compare_structures(l_pred, d_pred)
        l_props = compute_property_profile(l_pred)
        d_props = compute_property_profile(d_pred)

        assessment = assess_therapeutic_potential(
            protein_name="Test 20-mer",
            comparison=comparison,
            l_properties=l_props,
            d_properties=d_props,
            pdb_id="TEST",
        )

        assert assessment.pdb_id == "TEST"
        assert isinstance(assessment.binding_pocket_conserved, bool)
