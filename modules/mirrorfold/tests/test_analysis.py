"""Tests for mirrorfold.analysis — Biophysical property calculations."""

from __future__ import annotations

import pytest

from modules.mirrorfold.analysis import (
    molecular_weight,
    net_charge,
    isoelectric_point,
    hydrophobic_fraction,
    estimate_stability,
    estimate_surface_buried,
    compute_property_profile,
    compare_properties,
)
from modules.mirrorfold.predictor import predict_pair, _generate_mock_pdb
from modules.mirrorfold.models import StructurePrediction


# ── Molecular Weight ──────────────────────────────────────────────

class TestMolecularWeight:
    """Protein molecular weight calculation."""

    def test_glycine_mw(self) -> None:
        """Single glycine: residue weight + H2O."""
        mw = molecular_weight("G")
        assert mw == pytest.approx(57.05 + 18.02, abs=0.1)

    def test_longer_sequence(self) -> None:
        mw = molecular_weight("YGGFL")
        assert mw > 500  # Leu-enkephalin is ~555 Da

    def test_empty_sequence(self) -> None:
        mw = molecular_weight("")
        assert mw == pytest.approx(18.02, abs=0.1)  # just water


# ── Net Charge ───────────────────────────────────────────────────

class TestNetCharge:
    """Henderson–Hasselbalch net charge calculation."""

    def test_neutral_glycine(self) -> None:
        """Single glycine near neutral pH should be ~0."""
        charge = net_charge("G", ph=6.0)
        assert abs(charge) < 1.5

    def test_acidic_residue(self) -> None:
        """Aspartate at pH 7.4 should contribute negative charge."""
        charge = net_charge("D", ph=7.4)
        assert charge < 0

    def test_basic_residue(self) -> None:
        """Lysine at pH 7.4 should contribute positive charge."""
        charge = net_charge("K", ph=7.4)
        assert charge > 0

    def test_extreme_ph(self) -> None:
        """At very low pH, charge should be very positive."""
        charge = net_charge("ACDEFGHIKLMNPQRSTVWY", ph=1.0)
        assert charge > 3


# ── Isoelectric Point ───────────────────────────────────────────

class TestIsoelectricPoint:
    """Isoelectric point (pI) estimation."""

    def test_pi_in_range(self) -> None:
        pi = isoelectric_point("ACDEFGHIKLMNPQRSTVWY")
        assert 0 < pi < 14

    def test_acidic_protein(self) -> None:
        """All-Asp protein should have low pI."""
        pi = isoelectric_point("DDDDDD")
        assert pi < 5

    def test_basic_protein(self) -> None:
        """All-Lys protein should have high pI."""
        pi = isoelectric_point("KKKKKK")
        assert pi > 9


# ── Hydrophobic Fraction ────────────────────────────────────────

class TestHydrophobicFraction:
    """Hydrophobic residue fraction."""

    def test_all_hydrophobic(self) -> None:
        frac = hydrophobic_fraction("IVLFC")
        assert frac == pytest.approx(1.0)

    def test_all_hydrophilic(self) -> None:
        frac = hydrophobic_fraction("DERK")
        assert frac == pytest.approx(0.0)

    def test_mixed(self) -> None:
        frac = hydrophobic_fraction("ADEK")
        assert 0 < frac < 1

    def test_empty(self) -> None:
        assert hydrophobic_fraction("") == 0.0


# ── Stability Estimation ────────────────────────────────────────

class TestStability:
    """pLDDT → stability label mapping."""

    def test_very_high(self) -> None:
        assert estimate_stability(95.0) == "Very High"

    def test_high(self) -> None:
        assert estimate_stability(80.0) == "High"

    def test_moderate(self) -> None:
        assert estimate_stability(60.0) == "Moderate"

    def test_low(self) -> None:
        assert estimate_stability(40.0) == "Low"


# ── Surface / Buried Estimation ─────────────────────────────────

class TestSurfaceBuried:
    """Surface vs buried residue estimation."""

    def test_mock_pdb_has_both(self) -> None:
        pdb = _generate_mock_pdb("ACDEFGHIKLMNPQRSTVWY")
        n_surf, n_buried = estimate_surface_buried(pdb)
        assert n_surf + n_buried == 20

    def test_empty_pdb(self) -> None:
        n_surf, n_buried = estimate_surface_buried("")
        assert n_surf == 0
        assert n_buried == 0


# ── Property Profile ────────────────────────────────────────────

class TestPropertyProfile:
    """Full property profile computation."""

    def test_compute_profile(self) -> None:
        pred = StructurePrediction(
            sequence="YGGFL",
            chirality="L",
            pdb_string=_generate_mock_pdb("YGGFL"),
            plddt_scores=[80.0] * 5,
            mean_plddt=80.0,
            prediction_method="mock",
        )
        profile = compute_property_profile(pred)
        assert profile.sequence == "YGGFL"
        assert profile.chirality == "L"
        assert profile.molecular_weight > 0
        assert 0 < profile.isoelectric_point < 14


# ── L vs D Property Comparison ──────────────────────────────────

class TestCompareProperties:
    """Compare L and D property profiles."""

    def test_compare_mock_properties(self) -> None:
        l_pred, d_pred = predict_pair(
            "YGGFL",
            allow_mock=True,
            use_cache=False,
            allow_api_fallback=False,
        )
        l_prof = compute_property_profile(l_pred)
        d_prof = compute_property_profile(d_pred)
        diff = compare_properties(l_prof, d_prof)

        assert diff["molecular_weight"] > 0
        assert "chirality_independent_properties" in diff
        assert "chirality_dependent_properties" in diff
