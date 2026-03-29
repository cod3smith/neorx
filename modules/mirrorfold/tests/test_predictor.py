"""Tests for mirrorfold.predictor — Structure prediction."""

from __future__ import annotations

import pytest

from modules.mirrorfold.predictor import (
    predict_structure,
    predict_pair,
    extract_sequence_from_pdb,
    _generate_mock_pdb,
    _extract_plddt_from_pdb,
)
from modules.mirrorfold.models import StructurePrediction


# ── Mock PDB Generation ───────────────────────────────────────────

class TestMockPDB:
    """Mock PDB generator tests."""

    def test_mock_pdb_length(self) -> None:
        """Mock PDB has correct number of CA atoms."""
        pdb = _generate_mock_pdb("ACGT")
        ca_lines = [l for l in pdb.splitlines() if l.startswith("ATOM") and "CA" in l]
        assert len(ca_lines) == 4

    def test_mock_pdb_residue_names(self) -> None:
        """Mock PDB uses correct 3-letter codes."""
        pdb = _generate_mock_pdb("AG")
        lines = [l for l in pdb.splitlines() if l.startswith("ATOM") and "CA" in l]
        assert "ALA" in lines[0]
        assert "GLY" in lines[1]

    def test_mock_pdb_deterministic(self) -> None:
        """Same sequence → same mock PDB."""
        pdb1 = _generate_mock_pdb("YGGFL")
        pdb2 = _generate_mock_pdb("YGGFL")
        assert pdb1 == pdb2

    def test_mock_pdb_has_end(self) -> None:
        pdb = _generate_mock_pdb("AV")
        assert pdb.strip().endswith("END")

    def test_mock_pdb_plddt_range(self) -> None:
        """B-factor (pLDDT proxy) should be in [50, 95]."""
        pdb = _generate_mock_pdb("ACDEFGHIKLMNPQRSTVWY")
        plddts = _extract_plddt_from_pdb(pdb)
        assert all(50 <= p <= 95 for p in plddts)


# ── pLDDT Extraction ──────────────────────────────────────────────

class TestPLDDT:
    """pLDDT extraction from B-factor column."""

    def test_extract_from_mock(self) -> None:
        pdb = _generate_mock_pdb("YGGFL")
        plddts = _extract_plddt_from_pdb(pdb)
        assert len(plddts) == 5
        assert all(isinstance(p, float) for p in plddts)

    def test_extract_from_manual_pdb(self) -> None:
        pdb = (
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 85.50\n"
            "ATOM      2  CB  ALA A   1       2.000   3.000   4.000  1.00 85.50\n"
            "ATOM      3  CA  GLY A   2       4.800   2.000   3.000  1.00 72.30\n"
        )
        plddts = _extract_plddt_from_pdb(pdb)
        assert len(plddts) == 2  # only CA atoms
        assert plddts[0] == pytest.approx(85.5)
        assert plddts[1] == pytest.approx(72.3)

    def test_empty_pdb(self) -> None:
        plddts = _extract_plddt_from_pdb("")
        assert plddts == []


# ── Structure Prediction (mock mode) ──────────────────────────────

class TestPredictStructure:
    """Structure prediction with mock fallback."""

    def test_predict_l_mock(self) -> None:
        pred = predict_structure(
            "YGGFL", chirality="L", allow_mock=True,
            use_cache=False, allow_api_fallback=False,
        )
        assert isinstance(pred, StructurePrediction)
        assert pred.chirality == "L"
        assert pred.sequence == "YGGFL"
        assert pred.prediction_method == "mock"
        assert pred.mean_plddt > 0

    def test_predict_d_mock(self) -> None:
        pred = predict_structure(
            "YGGFL", chirality="D", allow_mock=True,
            use_cache=False, allow_api_fallback=False,
        )
        assert pred.chirality == "D"
        assert pred.prediction_method == "mock"
        # D-protein uses reversed sequence internally
        assert pred.sequence == "YGGFL"  # original sequence stored

    def test_predict_plddt_scores(self) -> None:
        pred = predict_structure(
            "ACDEF", chirality="L", allow_mock=True,
            use_cache=False, allow_api_fallback=False,
        )
        assert len(pred.plddt_scores) == 5
        assert pred.mean_plddt == pytest.approx(
            sum(pred.plddt_scores) / len(pred.plddt_scores), abs=0.1
        )

    def test_predict_has_pdb_string(self) -> None:
        pred = predict_structure(
            "AG", chirality="L", allow_mock=True,
            use_cache=False, allow_api_fallback=False,
        )
        assert "ATOM" in pred.pdb_string
        assert "END" in pred.pdb_string


# ── Predict Pair ──────────────────────────────────────────────────

class TestPredictPair:
    """Paired L+D prediction."""

    def test_predict_pair(self) -> None:
        l_pred, d_pred = predict_pair(
            "YGGFL",
            allow_mock=True,
            use_cache=False,
            allow_api_fallback=False,
        )
        assert l_pred.chirality == "L"
        assert d_pred.chirality == "D"
        assert l_pred.sequence == d_pred.sequence

    def test_pair_both_have_pdb(self) -> None:
        l_pred, d_pred = predict_pair(
            "AGA",
            allow_mock=True,
            use_cache=False,
            allow_api_fallback=False,
        )
        assert "ATOM" in l_pred.pdb_string
        assert "ATOM" in d_pred.pdb_string


# ── PDB Sequence Extraction ──────────────────────────────────────

class TestExtractSequence:
    """Extract amino acid sequence from PDB strings."""

    def test_extract_from_mock(self) -> None:
        pdb = _generate_mock_pdb("YGGFL")
        seq = extract_sequence_from_pdb(pdb)
        assert seq == "YGGFL"

    def test_extract_single_residue(self) -> None:
        pdb = _generate_mock_pdb("A")
        seq = extract_sequence_from_pdb(pdb)
        assert seq == "A"

    def test_extract_empty(self) -> None:
        seq = extract_sequence_from_pdb("")
        assert seq == ""
