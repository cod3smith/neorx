"""
Structure Prediction with ESMFold
==================================

Predicts 3D protein structures from amino-acid sequences using
Meta AI's ESMFold model, with automatic fallback to the ESMFold
REST API.

What is ESMFold?
----------------
ESMFold is a protein structure prediction model from Meta (FAIR)
that predicts 3D coordinates directly from a single sequence —
no multiple sequence alignment (MSA) needed.  This makes it much
faster than AlphaFold2 (seconds vs hours) while achieving
comparable accuracy for well-folded proteins.

The model is built on the ESM-2 protein language model (650M
parameters) and a structure module inspired by AlphaFold2.  It
outputs:

* **Atom coordinates** in PDB format
* **pLDDT scores** — per-residue confidence (0–100) stored in the
  B-factor column of the PDB file

pLDDT interpretation
--------------------
* **> 90** — Very high confidence.  Backbone and side-chain
  positions are reliable.
* **70–90** — Confident.  Backbone is reliable; side chains may
  have some error.
* **50–70** — Low confidence.  Possible loop region or partial
  disorder.
* **< 50** — Very low confidence.  Likely intrinsically
  disordered or prediction is unreliable.

Limitations for D-proteins
--------------------------
ESMFold was trained exclusively on L-protein structures from the
PDB.  It has **never seen** a D-protein.  We approximate the
D-protein structure using the reversed-sequence proxy (see
``mirror.py``).  This works because:

    fold(D-protein seq ABCDE) ≈ mirror(fold(L-protein seq EDCBA))

This approximation is well-validated for globular proteins
(Kent et al., HIV protease) but less reliable for:
* Intrinsically disordered regions
* Proteins whose folding depends on chaperones
* Very long sequences (> 400 residues)
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from .models import StructurePrediction
from .mirror import mirror_sequence, reflect_coordinates

logger = logging.getLogger(__name__)

# ── Cache directory ─────────────────────────────────────────────────
_CACHE_DIR = Path(__file__).parent / "cache"

# ── ESMFold API endpoint ────────────────────────────────────────────
ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"


def _cache_key(sequence: str, chirality: str) -> str:
    """Generate a deterministic cache key from sequence + chirality."""
    data = f"{chirality}:{sequence}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _load_from_cache(
    sequence: str, chirality: str
) -> Optional[StructurePrediction]:
    """Attempt to load a cached prediction."""
    key = _cache_key(sequence, chirality)
    cache_file = _CACHE_DIR / f"{key}.json"

    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            pred = StructurePrediction(**data)
            logger.info("Cache hit: %s (%s, %d residues).", key, chirality, len(sequence))
            return pred
        except Exception as e:
            logger.warning("Cache read failed for %s: %s", key, e)

    return None


def _save_to_cache(prediction: StructurePrediction) -> None:
    """Save a prediction to the file cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(prediction.sequence, prediction.chirality)
    cache_file = _CACHE_DIR / f"{key}.json"

    try:
        cache_file.write_text(prediction.model_dump_json(indent=2))
        logger.info("Cached prediction → %s", cache_file.name)
    except Exception as e:
        logger.warning("Cache write failed: %s", e)


def _extract_plddt_from_pdb(pdb_string: str) -> list[float]:
    """Extract per-residue pLDDT scores from the B-factor column.

    ESMFold stores pLDDT in the B-factor field of Cα atoms.
    PDB format: B-factor is columns 61–66 (0-indexed: 60:66).

    Parameters
    ----------
    pdb_string : str
        PDB-format string from ESMFold.

    Returns
    -------
    list[float]
        Per-residue pLDDT scores.
    """
    scores: list[float] = []
    seen_residues: set[int] = set()

    for line in pdb_string.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue

        res_seq = int(line[22:26].strip())
        if res_seq in seen_residues:
            continue
        seen_residues.add(res_seq)

        try:
            bfactor = float(line[60:66].strip())
            scores.append(bfactor)
        except (ValueError, IndexError):
            scores.append(0.0)

    return scores


def _predict_esmfold_local(sequence: str) -> Optional[str]:
    """Predict structure using local ESMFold model.

    Requires the ``fair-esm`` package with ESMFold support and
    a GPU with sufficient memory (~3 GB for short sequences,
    more for longer ones).

    Parameters
    ----------
    sequence : str
        Amino acid sequence.

    Returns
    -------
    str or None
        PDB string, or None if ESMFold is not available.
    """
    try:
        import torch
        import esm

        logger.info("Loading ESMFold model (this may take a minute)…")
        model = esm.pretrained.esmfold_v1()
        model = model.eval()

        # Use GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("ESMFold running on CUDA.")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS may not support all ESMFold ops — try and fall back
            try:
                model = model.to("mps")
                logger.info("ESMFold running on MPS.")
            except Exception:
                logger.info("MPS not supported for ESMFold, using CPU.")
        else:
            logger.info("ESMFold running on CPU (this will be slow).")

        with torch.no_grad():
            pdb_string = model.infer_pdb(sequence)

        return pdb_string

    except ImportError:
        logger.info(
            "ESMFold not available locally. Install with: "
            "pip install fair-esm[esmfold]"
        )
        return None
    except Exception as e:
        logger.warning("Local ESMFold prediction failed: %s", e)
        return None


def _predict_esmfold_api(sequence: str) -> Optional[str]:
    """Predict structure using the ESMFold REST API.

    Falls back to the public ESM Metagenomic Atlas API.  This is
    slower than local prediction but requires no GPU or model
    download.

    Parameters
    ----------
    sequence : str
        Amino acid sequence (max ~400 residues for API).

    Returns
    -------
    str or None
        PDB string, or None if the API is unavailable.
    """
    try:
        logger.info(
            "Calling ESMFold API (%d residues)…", len(sequence)
        )
        response = requests.post(
            ESMFOLD_API_URL,
            data=sequence,
            headers={"Content-Type": "text/plain"},
            timeout=120,
        )
        response.raise_for_status()
        pdb_string = response.text

        # Basic validation
        if "ATOM" not in pdb_string:
            logger.warning("ESMFold API returned invalid PDB.")
            return None

        logger.info("ESMFold API prediction successful.")
        return pdb_string

    except requests.exceptions.RequestException as e:
        logger.warning("ESMFold API unavailable: %s", e)
        return None


def _generate_mock_pdb(sequence: str) -> str:
    """Generate a minimal mock PDB for testing.

    Creates a linear chain of Cα atoms spaced 3.8 Å apart
    along the x-axis with random pLDDT values.  This is NOT
    a real structure prediction — only for unit testing and
    development.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.

    Returns
    -------
    str
        Mock PDB string.
    """
    from .mirror import AA_1TO3

    rng = np.random.RandomState(hash(sequence) % 2**31)
    lines: list[str] = []
    atom_num = 1

    for i, aa in enumerate(sequence):
        resname = AA_1TO3.get(aa, "ALA")
        x = i * 3.8
        y = rng.normal(0, 0.5)
        z = rng.normal(0, 0.5)
        plddt = rng.uniform(50, 95)

        line = (
            f"ATOM  {atom_num:5d}  CA  {resname:3s} A{i + 1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{plddt:6.2f}           C  "
        )
        lines.append(line)
        atom_num += 1

    lines.append("END")
    return "\n".join(lines)


def predict_structure(
    sequence: str,
    chirality: str = "L",
    use_cache: bool = True,
    allow_api_fallback: bool = True,
    allow_mock: bool = False,
) -> StructurePrediction:
    """Predict the 3D structure of a protein sequence.

    Attempts prediction in order:
    1. Check file cache
    2. Local ESMFold model (requires ``fair-esm[esmfold]``)
    3. ESMFold REST API (requires internet)
    4. Mock prediction (if ``allow_mock=True``, for testing)

    For D-proteins, the sequence is reversed before prediction
    (the "reverse-and-reflect" proxy), and the resulting
    coordinates are reflected to produce D-protein geometry.

    Parameters
    ----------
    sequence : str
        Amino acid sequence in one-letter codes.
    chirality : str
        ``"L"`` for natural protein, ``"D"`` for mirror-image.
    use_cache : bool
        Check/write the prediction cache.
    allow_api_fallback : bool
        Allow the ESMFold API as a fallback.
    allow_mock : bool
        Generate a mock structure if all else fails.

    Returns
    -------
    StructurePrediction
        Predicted structure with pLDDT scores.

    Raises
    ------
    RuntimeError
        If no prediction method is available.
    """
    seq = sequence.upper().strip()

    # For D-proteins: reverse the sequence for prediction
    prediction_seq = mirror_sequence(seq) if chirality.upper() == "D" else seq

    # Check cache
    if use_cache:
        cached = _load_from_cache(seq, chirality)
        if cached is not None:
            return cached

    # Try local ESMFold
    pdb_string = _predict_esmfold_local(prediction_seq)
    method = "esmfold_local"

    # Fallback to API
    if pdb_string is None and allow_api_fallback:
        pdb_string = _predict_esmfold_api(prediction_seq)
        method = "esmfold_api"

    # Fallback to mock
    if pdb_string is None and allow_mock:
        pdb_string = _generate_mock_pdb(prediction_seq)
        method = "mock"
        logger.warning(
            "Using MOCK prediction — not a real structure! "
            "Install fair-esm for real predictions."
        )

    if pdb_string is None:
        raise RuntimeError(
            "No structure prediction method available. Options:\n"
            "  1. Install ESMFold: pip install fair-esm[esmfold]\n"
            "  2. Enable API fallback: allow_api_fallback=True\n"
            "  3. Enable mock for testing: allow_mock=True"
        )

    # For D-proteins: reflect coordinates to get mirror geometry
    if chirality.upper() == "D":
        pdb_string = reflect_coordinates(pdb_string, axis="x")

    # Extract pLDDT scores
    plddt_scores = _extract_plddt_from_pdb(pdb_string)
    mean_plddt = float(np.mean(plddt_scores)) if plddt_scores else 0.0

    prediction = StructurePrediction(
        sequence=seq,
        chirality=chirality.upper(),
        pdb_string=pdb_string,
        plddt_scores=plddt_scores,
        mean_plddt=mean_plddt,
        prediction_method=method,
    )

    # Cache the result
    if use_cache:
        _save_to_cache(prediction)

    logger.info(
        "Predicted %s-structure: %d residues, mean pLDDT=%.1f (%s).",
        chirality, len(seq), mean_plddt, method,
    )

    return prediction


def predict_pair(
    sequence: str,
    use_cache: bool = True,
    allow_api_fallback: bool = True,
    allow_mock: bool = False,
) -> tuple[StructurePrediction, StructurePrediction]:
    """Predict both L- and D-protein structures.

    Convenience function that predicts the natural L-form and
    the mirror-image D-form for the same sequence.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    use_cache : bool
        Use prediction cache.
    allow_api_fallback : bool
        Allow ESMFold API fallback.
    allow_mock : bool
        Allow mock predictions for testing.

    Returns
    -------
    tuple[StructurePrediction, StructurePrediction]
        (L-structure, D-structure)
    """
    kwargs = dict(
        use_cache=use_cache,
        allow_api_fallback=allow_api_fallback,
        allow_mock=allow_mock,
    )

    l_pred = predict_structure(sequence, chirality="L", **kwargs)
    d_pred = predict_structure(sequence, chirality="D", **kwargs)

    return l_pred, d_pred


def fetch_pdb(pdb_id: str) -> Optional[str]:
    """Fetch a PDB file from the RCSB Protein Data Bank.

    Parameters
    ----------
    pdb_id : str
        4-character PDB identifier (e.g. ``"1HHP"``).

    Returns
    -------
    str or None
        PDB-format string, or None if not found.
    """
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        logger.info("Fetched PDB %s from RCSB.", pdb_id.upper())
        return response.text
    except requests.exceptions.RequestException as e:
        logger.warning("Failed to fetch PDB %s: %s", pdb_id, e)
        return None


def extract_sequence_from_pdb(pdb_string: str) -> str:
    """Extract the amino acid sequence from a PDB file.

    Reads ATOM records for Cα atoms and maps three-letter
    residue names to one-letter codes.

    Parameters
    ----------
    pdb_string : str
        PDB-format string.

    Returns
    -------
    str
        Amino acid sequence in one-letter codes.
    """
    from .mirror import AA_3TO1

    sequence: list[str] = []
    seen_residues: set[int] = set()

    for line in pdb_string.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue

        res_seq = int(line[22:26].strip())
        if res_seq in seen_residues:
            continue
        seen_residues.add(res_seq)

        resname = line[17:20].strip()
        aa = AA_3TO1.get(resname, "X")
        if aa != "X":
            sequence.append(aa)

    return "".join(sequence)
