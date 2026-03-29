"""Tests for mirrorfold.compare — Structural comparison."""

from __future__ import annotations

import numpy as np
import pytest

from modules.mirrorfold.compare import (
    superimpose,
    calculate_tm_score,
    per_residue_distances,
    assign_secondary_structure,
    compare_structures,
    _identify_regions,
    _extract_ca_coords,
)
from modules.mirrorfold.predictor import predict_pair, _generate_mock_pdb
from modules.mirrorfold.models import StructurePrediction


# ── Superimposition ───────────────────────────────────────────────

class TestSuperimpose:
    """Kabsch alignment tests."""

    def test_identical_structures(self) -> None:
        """RMSD of identical coordinates should be ~0."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        _, rmsd = superimpose(coords, coords.copy())
        assert rmsd == pytest.approx(0.0, abs=1e-6)

    def test_translated_structure(self) -> None:
        """Pure translation should give RMSD ~0 after alignment."""
        coords = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0]], dtype=np.float64)
        shifted = coords + np.array([10, 20, 30])
        _, rmsd = superimpose(coords, shifted)
        assert rmsd == pytest.approx(0.0, abs=1e-6)

    def test_rotated_structure(self) -> None:
        """90° rotation should give RMSD ~0 after alignment."""
        coords = np.array([
            [0, 0, 0],
            [5, 0, 0],
            [0, 5, 0],
            [0, 0, 5],
        ], dtype=np.float64)
        # 90° rotation around z-axis: (x,y,z) → (-y,x,z)
        rotated = np.array([
            [0, 0, 0],
            [0, 5, 0],
            [-5, 0, 0],
            [0, 0, 5],
        ], dtype=np.float64)
        _, rmsd = superimpose(coords, rotated)
        assert rmsd == pytest.approx(0.0, abs=1e-4)

    def test_different_structures_nonzero_rmsd(self) -> None:
        """Genuinely different structures should have RMSD > 0."""
        coords1 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64)
        coords2 = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0]], dtype=np.float64)
        _, rmsd = superimpose(coords1, coords2)
        assert rmsd > 0.1


# ── TM-score ─────────────────────────────────────────────────────

class TestTMScore:
    """TM-score calculation tests."""

    def test_identical_tm_score(self) -> None:
        """Identical structures should have TM ~1.0."""
        coords = np.random.rand(30, 3)
        tm = calculate_tm_score(coords, coords)
        assert tm == pytest.approx(1.0, abs=0.01)

    def test_random_tm_score_lower(self) -> None:
        """Random structures should have low TM-score."""
        np.random.seed(42)
        coords1 = np.random.rand(30, 3) * 20
        coords2 = np.random.rand(30, 3) * 20
        tm = calculate_tm_score(coords1, coords2)
        assert tm < 0.8  # should be well below 1.0

    def test_tm_score_bounded(self) -> None:
        """TM-score should be in [0, 1]."""
        np.random.seed(123)
        for _ in range(10):
            coords1 = np.random.rand(20, 3) * 10
            coords2 = np.random.rand(20, 3) * 10
            tm = calculate_tm_score(coords1, coords2)
            assert 0.0 <= tm <= 1.0

    def test_short_protein_tm_score(self) -> None:
        """TM-score should work for very short proteins."""
        coords = np.array([[0, 0, 0], [3.8, 0, 0], [7.6, 0, 0]], dtype=np.float64)
        tm = calculate_tm_score(coords, coords)
        assert tm == pytest.approx(1.0, abs=0.01)


# ── Per-residue Distances ────────────────────────────────────────

class TestPerResidueDistances:
    """Per-residue distance calculation."""

    def test_identical_zero_distances(self) -> None:
        coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        dists = per_residue_distances(coords, coords)
        assert all(d == pytest.approx(0.0, abs=1e-6) for d in dists)

    def test_known_distance(self) -> None:
        c1 = np.array([[0, 0, 0]], dtype=np.float64)
        c2 = np.array([[3, 4, 0]], dtype=np.float64)
        dists = per_residue_distances(c1, c2)
        assert dists[0] == pytest.approx(5.0, abs=1e-6)


# ── Region Identification ───────────────────────────────────────

class TestRegionIdentification:
    """Conserved vs divergent region classification."""

    def test_all_conserved(self) -> None:
        distances = [0.5, 0.8, 1.0, 1.5]
        conserved, divergent = _identify_regions(distances, threshold=2.0)
        assert len(conserved) == 1
        assert len(divergent) == 0
        assert conserved[0] == (0, 3)

    def test_all_divergent(self) -> None:
        distances = [3.0, 4.0, 5.0]
        conserved, divergent = _identify_regions(distances, threshold=2.0)
        assert len(conserved) == 0
        assert len(divergent) == 1

    def test_mixed_regions(self) -> None:
        distances = [0.5, 0.8, 3.0, 4.0, 0.5, 0.6]
        conserved, divergent = _identify_regions(distances, threshold=2.0)
        assert len(conserved) == 2  # indices 0-1 and 4-5
        assert len(divergent) == 1  # indices 2-3

    def test_empty_distances(self) -> None:
        conserved, divergent = _identify_regions([], threshold=2.0)
        assert conserved == []
        assert divergent == []


# ── Secondary Structure Assignment ───────────────────────────────

class TestSecondaryStructure:
    """Simplified secondary structure assignment."""

    def test_mock_pdb_has_assignments(self) -> None:
        """Mock PDB should produce SS assignments."""
        pdb = _generate_mock_pdb("ACDEFGHIKLMNPQRSTVWY")
        ss = assign_secondary_structure(pdb)
        assert len(ss) == 20
        assert all(s in ("H", "E", "C") for s in ss)

    def test_short_sequence(self) -> None:
        pdb = _generate_mock_pdb("AG")
        ss = assign_secondary_structure(pdb)
        assert len(ss) == 2


# ── Coordinate Extraction ────────────────────────────────────────

class TestCoordExtraction:
    """Cα coordinate extraction from PDB strings."""

    def test_extract_coords(self) -> None:
        pdb = _generate_mock_pdb("YGGFL")
        coords = _extract_ca_coords(pdb)
        assert coords.shape == (5, 3)

    def test_extract_empty(self) -> None:
        coords = _extract_ca_coords("")
        assert coords.shape == (0,)  # empty array


# ── Full Comparison Pipeline ─────────────────────────────────────

class TestCompareStructures:
    """End-to-end structure comparison."""

    def test_compare_mock_structures(self) -> None:
        """Compare L and D mock structures."""
        l_pred, d_pred = predict_pair(
            "YGGFL",
            allow_mock=True,
            use_cache=False,
            allow_api_fallback=False,
        )
        report = compare_structures(l_pred, d_pred)

        assert report.rmsd is not None
        assert report.rmsd >= 0
        assert report.tm_score is not None
        assert 0 <= report.tm_score <= 1
        assert report.aligned_length == 5
        assert len(report.per_residue_distances) == 5

    def test_compare_identical_structures(self) -> None:
        """Same structure compared to itself should give perfect scores."""
        pred = StructurePrediction(
            sequence="YGGFL",
            chirality="L",
            pdb_string=_generate_mock_pdb("YGGFL"),
            plddt_scores=[80.0] * 5,
            mean_plddt=80.0,
            prediction_method="mock",
        )
        report = compare_structures(pred, pred)
        assert report.rmsd == pytest.approx(0.0, abs=1e-4)
        assert report.tm_score == pytest.approx(1.0, abs=0.01)

    def test_compare_has_regions(self) -> None:
        """Comparison should identify conserved/divergent regions."""
        l_pred, d_pred = predict_pair(
            "ACDEFGHIKLMNPQRSTVWY",
            allow_mock=True,
            use_cache=False,
            allow_api_fallback=False,
        )
        report = compare_structures(l_pred, d_pred)

        total = len(report.conserved_regions) + len(report.divergent_regions)
        assert total > 0  # at least one region identified
