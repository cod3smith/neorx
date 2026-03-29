"""
MirrorFold FastAPI Service
===========================

REST API for structure prediction, comparison, and therapeutic
assessment of L- and D-protein pairs.

Endpoints
----------
* ``POST /predict``   — Predict a single structure
* ``POST /compare``   — Compare L vs D structures
* ``POST /assess``    — Full therapeutic assessment
* ``GET  /health``    — Health check
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MirrorFold API",
    description=(
        "🪞 Predict and compare structures of natural (L-amino acid) "
        "proteins versus their mirror-image (D-amino acid) counterparts."
    ),
    version="0.1.0",
)


# ── Request / Response Models ──────────────────────────────────────

class PredictRequest(BaseModel):
    """Request body for structure prediction."""
    sequence: str = Field(..., description="Amino acid sequence", min_length=1)
    chirality: str = Field("L", pattern=r"^[LD]$", description="L or D")
    allow_api_fallback: bool = Field(True, description="Allow ESMFold API")
    allow_mock: bool = Field(False, description="Allow mock PDB for testing")


class PredictResponse(BaseModel):
    """Response for structure prediction."""
    sequence: str
    chirality: str
    pdb_string: str
    mean_plddt: float
    prediction_method: str
    n_residues: int


class CompareRequest(BaseModel):
    """Request body for L vs D comparison."""
    sequence: str = Field(..., description="Amino acid sequence", min_length=1)
    distance_threshold: float = Field(2.0, description="Å threshold", ge=0.1)
    allow_api_fallback: bool = True
    allow_mock: bool = False


class CompareResponse(BaseModel):
    """Response for structural comparison."""
    rmsd: Optional[float] = None
    tm_score: Optional[float] = None
    secondary_structure_match: Optional[float] = None
    conserved_regions: list = []
    divergent_regions: list = []
    aligned_length: int = 0
    l_plddt: float = 0.0
    d_plddt: float = 0.0
    l_method: str = ""
    d_method: str = ""


class AssessRequest(BaseModel):
    """Request body for therapeutic assessment."""
    sequence: str = Field(..., description="Amino acid sequence", min_length=1)
    protein_name: str = Field("Unnamed", description="Human-readable name")
    pdb_id: Optional[str] = Field(None, description="PDB identifier")
    allow_api_fallback: bool = True
    allow_mock: bool = False


class AssessResponse(BaseModel):
    """Response for therapeutic assessment."""
    protein_name: str
    therapeutic_viability: str
    protease_resistance: str
    immunogenicity: str
    binding_pocket_conserved: bool
    rmsd: Optional[float] = None
    tm_score: Optional[float] = None
    rationale: str
    l_plddt: float = 0.0
    d_plddt: float = 0.0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    module: str = "mirrorfold"
    esmfold_available: bool = False


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check — verifies the module is loaded."""
    esmfold = False
    try:
        import esm  # noqa: F401
        esmfold = True
    except ImportError:
        pass

    return HealthResponse(esmfold_available=esmfold)


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(req: PredictRequest) -> PredictResponse:
    """Predict the 3D structure of a protein sequence."""
    from .mirror import is_valid_sequence
    from .predictor import predict_structure

    if not is_valid_sequence(req.sequence):
        raise HTTPException(400, "Invalid amino acid sequence.")

    prediction = predict_structure(
        sequence=req.sequence,
        chirality=req.chirality,
        allow_api_fallback=req.allow_api_fallback,
        allow_mock=req.allow_mock,
    )

    return PredictResponse(
        sequence=prediction.sequence,
        chirality=prediction.chirality,
        pdb_string=prediction.pdb_string,
        mean_plddt=prediction.mean_plddt or 0.0,
        prediction_method=prediction.prediction_method,
        n_residues=len(prediction.sequence),
    )


@app.post("/compare", response_model=CompareResponse)
async def compare_endpoint(req: CompareRequest) -> CompareResponse:
    """Compare L and D protein structures."""
    from .mirror import is_valid_sequence
    from .predictor import predict_pair
    from .compare import compare_structures

    if not is_valid_sequence(req.sequence):
        raise HTTPException(400, "Invalid amino acid sequence.")

    l_pred, d_pred = predict_pair(
        sequence=req.sequence,
        allow_api_fallback=req.allow_api_fallback,
        allow_mock=req.allow_mock,
    )

    report = compare_structures(
        l_pred, d_pred,
        distance_threshold=req.distance_threshold,
    )

    return CompareResponse(
        rmsd=report.rmsd,
        tm_score=report.tm_score,
        secondary_structure_match=report.secondary_structure_match,
        conserved_regions=report.conserved_regions,
        divergent_regions=report.divergent_regions,
        aligned_length=report.aligned_length,
        l_plddt=l_pred.mean_plddt or 0.0,
        d_plddt=d_pred.mean_plddt or 0.0,
        l_method=l_pred.prediction_method,
        d_method=d_pred.prediction_method,
    )


@app.post("/assess", response_model=AssessResponse)
async def assess_endpoint(req: AssessRequest) -> AssessResponse:
    """Full D-protein therapeutic assessment."""
    from .mirror import is_valid_sequence
    from .predictor import predict_pair
    from .compare import compare_structures
    from .analysis import compute_property_profile
    from .therapeutic import assess_therapeutic_potential

    if not is_valid_sequence(req.sequence):
        raise HTTPException(400, "Invalid amino acid sequence.")

    l_pred, d_pred = predict_pair(
        sequence=req.sequence,
        allow_api_fallback=req.allow_api_fallback,
        allow_mock=req.allow_mock,
    )

    comparison = compare_structures(l_pred, d_pred)
    l_props = compute_property_profile(l_pred)
    d_props = compute_property_profile(d_pred)

    assessment = assess_therapeutic_potential(
        protein_name=req.protein_name,
        comparison=comparison,
        l_properties=l_props,
        d_properties=d_props,
        pdb_id=req.pdb_id,
    )

    return AssessResponse(
        protein_name=assessment.protein_name,
        therapeutic_viability=assessment.therapeutic_viability,
        protease_resistance=assessment.estimated_protease_resistance,
        immunogenicity=assessment.estimated_immunogenicity,
        binding_pocket_conserved=assessment.binding_pocket_conserved,
        rmsd=comparison.rmsd,
        tm_score=comparison.tm_score,
        rationale=assessment.rationale,
        l_plddt=l_pred.mean_plddt or 0.0,
        d_plddt=d_pred.mean_plddt or 0.0,
    )
