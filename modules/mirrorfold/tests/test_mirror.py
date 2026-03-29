"""Tests for mirrorfold.mirror — L→D transformations."""

from __future__ import annotations

import pytest

from modules.mirrorfold.mirror import (
    STANDARD_AA,
    L_AMINO_ACID_SMILES,
    D_AMINO_ACID_SMILES,
    AA_3TO1,
    AA_1TO3,
    is_valid_sequence,
    mirror_sequence,
    get_l_smiles,
    get_d_smiles,
    sequence_to_smiles,
    verify_mirror_smiles,
    reflect_coordinates,
)


# ── Sequence Validation ────────────────────────────────────────────

class TestValidation:
    """Sequence validation tests."""

    def test_valid_sequence(self) -> None:
        assert is_valid_sequence("MKFLILF") is True

    def test_valid_single_residue(self) -> None:
        assert is_valid_sequence("A") is True

    def test_empty_sequence_invalid(self) -> None:
        assert is_valid_sequence("") is False

    def test_non_standard_residue(self) -> None:
        assert is_valid_sequence("MKXFLILF") is False

    def test_lowercase_valid(self) -> None:
        # is_valid_sequence normalises to uppercase
        assert is_valid_sequence("mkflilf") is True

    def test_numbers_invalid(self) -> None:
        assert is_valid_sequence("MK123") is False


# ── Mirror Sequence ────────────────────────────────────────────────

class TestMirrorSequence:
    """D-protein sequence generation (reversal)."""

    def test_reverse_simple(self) -> None:
        """D-protein approximation reverses the sequence."""
        assert mirror_sequence("ACDF") == "FDCA"

    def test_palindrome_unchanged(self) -> None:
        assert mirror_sequence("ACA") == "ACA"

    def test_single_residue(self) -> None:
        assert mirror_sequence("G") == "G"

    def test_enkephalin(self) -> None:
        """Leu-enkephalin YGGFL → LFGGY."""
        assert mirror_sequence("YGGFL") == "LFGGY"


# ── SMILES Lookups ─────────────────────────────────────────────────

class TestSMILES:
    """L/D amino acid SMILES generation."""

    def test_all_20_l_smiles(self) -> None:
        """Every standard AA has an L-SMILES entry."""
        for aa in STANDARD_AA:
            smiles = get_l_smiles(aa)
            assert smiles is not None, f"Missing L-SMILES for {aa}"
            assert len(smiles) > 0

    def test_all_20_d_smiles(self) -> None:
        """Every standard AA has a D-SMILES entry."""
        for aa in STANDARD_AA:
            smiles = get_d_smiles(aa)
            assert smiles is not None, f"Missing D-SMILES for {aa}"
            assert len(smiles) > 0

    def test_glycine_achiral(self) -> None:
        """Glycine has no stereocenter — L and D SMILES are equal."""
        assert get_l_smiles("G") == get_d_smiles("G")

    def test_alanine_stereo_differs(self) -> None:
        """L-Ala and D-Ala have different stereocenters."""
        l = get_l_smiles("A")
        d = get_d_smiles("A")
        assert l != d
        assert "@@" in l or "@" in l
        assert "@@" in d or "@" in d

    def test_unknown_residue_returns_none(self) -> None:
        assert get_l_smiles("X") is None
        assert get_d_smiles("X") is None

    def test_sequence_to_smiles_l(self) -> None:
        result = sequence_to_smiles("AG", "L")
        assert len(result) == 2
        assert result[0]["residue"] == "A"
        assert result[0]["chirality"] == "L"
        assert result[0]["valid"] is True
        assert result[1]["residue"] == "G"

    def test_sequence_to_smiles_d(self) -> None:
        result = sequence_to_smiles("AG", "D")
        assert len(result) == 2
        assert result[0]["chirality"] == "D"
        assert result[0]["valid"] is True


# ── SMILES Verification ───────────────────────────────────────────

class TestVerifyMirror:
    """Verify that L/D SMILES pairs are valid mirror molecules."""

    def test_verify_alanine(self) -> None:
        result = verify_mirror_smiles("A")
        assert result["valid"] is True
        assert result["same_formula"] is True

    def test_verify_glycine(self) -> None:
        result = verify_mirror_smiles("G")
        assert result["valid"] is True

    @pytest.mark.parametrize("aa", list(STANDARD_AA))
    def test_verify_all_standard(self, aa: str) -> None:
        """All 20 standard AAs should have valid mirror SMILES."""
        result = verify_mirror_smiles(aa)
        assert result["valid"] is True, f"Invalid mirror for {aa}: {result}"


# ── Coordinate Reflection ──────────────────────────────────────────

class TestReflection:
    """PDB coordinate reflection for D-protein geometry."""

    MOCK_PDB = (
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 80.00\n"
        "ATOM      2  CA  GLY A   2       4.800   5.000   6.000  1.00 75.00\n"
    )

    def test_reflect_x(self) -> None:
        reflected = reflect_coordinates(self.MOCK_PDB, axis="x")
        lines = [l for l in reflected.splitlines() if l.startswith("ATOM")]
        x1 = float(lines[0][30:38])
        x2 = float(lines[1][30:38])
        assert x1 == pytest.approx(-1.0, abs=0.01)
        assert x2 == pytest.approx(-4.8, abs=0.01)

    def test_reflect_y(self) -> None:
        reflected = reflect_coordinates(self.MOCK_PDB, axis="y")
        lines = [l for l in reflected.splitlines() if l.startswith("ATOM")]
        y1 = float(lines[0][38:46])
        assert y1 == pytest.approx(-2.0, abs=0.01)

    def test_reflect_preserves_non_atom_lines(self) -> None:
        pdb = "HEADER TEST\n" + self.MOCK_PDB + "END\n"
        reflected = reflect_coordinates(pdb, axis="x")
        assert "HEADER TEST" in reflected
        assert "END" in reflected

    def test_reflect_z(self) -> None:
        reflected = reflect_coordinates(self.MOCK_PDB, axis="z")
        lines = [l for l in reflected.splitlines() if l.startswith("ATOM")]
        z1 = float(lines[0][46:54])
        assert z1 == pytest.approx(-3.0, abs=0.01)


# ── Mapping Tables ─────────────────────────────────────────────────

class TestMappings:
    """Three-letter ↔ one-letter amino acid code mappings."""

    def test_3to1_complete(self) -> None:
        assert len(AA_3TO1) == 20

    def test_1to3_complete(self) -> None:
        assert len(AA_1TO3) == 20

    def test_roundtrip(self) -> None:
        for one, three in AA_1TO3.items():
            assert AA_3TO1[three] == one

    def test_known_mapping(self) -> None:
        assert AA_3TO1["ALA"] == "A"
        assert AA_1TO3["W"] == "TRP"
