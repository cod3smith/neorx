"""
ADMET Property Predictor
==========================

Multi-rule ADMET (Absorption, Distribution, Metabolism,
Excretion, Toxicity) scoring based on established medicinal
chemistry rules.

Rules Applied
--------------
1. **Lipinski's Rule of Five** (absorption)
2. **Veber Rules** (oral bioavailability)
3. **Ghose Filter** (drug-likeness range)
4. **Egan Egg Model** (intestinal absorption)
5. **PAINS Filter** (pan-assay interference)
6. **BBB Permeability Estimate**
7. **hERG Liability Estimate**
8. **Reactive-Group Toxicophore Alerts**

Each rule contributes to a composite ADMET score in [0, 1].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ADMETProfile:
    """Detailed ADMET assessment for a molecule."""

    absorption: float = 0.5
    distribution: float = 0.5
    metabolism: float = 0.5
    excretion: float = 0.5
    toxicity: float = 0.5
    composite: float = 0.5
    flags: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def predict_admet(smiles: str) -> ADMETProfile:
    """Predict ADMET properties for a molecule.

    Parameters
    ----------
    smiles : str
        SMILES string.

    Returns
    -------
    ADMETProfile
        Multi-dimensional ADMET assessment with composite score.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors, FilterCatalog

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ADMETProfile(composite=0.0, flags=["Invalid SMILES"])

        # ── Compute descriptors ─────────────────────────────────

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = Descriptors.TPSA(mol)
        rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
        n_atoms = mol.GetNumHeavyAtoms()
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

        flags: list[str] = []
        details: dict[str, Any] = {
            "mw": round(mw, 1),
            "logp": round(logp, 2),
            "hbd": hbd,
            "hba": hba,
            "tpsa": round(tpsa, 1),
            "rotatable_bonds": rotatable,
            "n_heavy_atoms": n_atoms,
            "n_rings": n_rings,
            "n_aromatic_rings": n_aromatic_rings,
        }

        # ── 1. Absorption (Lipinski + Veber) ────────────────────

        absorption = 1.0
        lipinski_violations = 0
        if mw > 500:
            lipinski_violations += 1
        if logp > 5:
            lipinski_violations += 1
        if hbd > 5:
            lipinski_violations += 1
        if hba > 10:
            lipinski_violations += 1

        if lipinski_violations == 0:
            absorption = 1.0
        elif lipinski_violations == 1:
            absorption = 0.7
            flags.append("Lipinski: 1 violation")
        else:
            absorption = max(0.1, 1.0 - lipinski_violations * 0.25)
            flags.append(f"Lipinski: {lipinski_violations} violations")

        # Veber rules
        if rotatable > 10:
            absorption -= 0.15
            flags.append("Veber: >10 rotatable bonds")
        if tpsa > 140:
            absorption -= 0.2
            flags.append("Veber: TPSA >140 Å²")

        details["lipinski_violations"] = lipinski_violations

        # ── 2. Distribution ─────────────────────────────────────

        distribution = 0.7

        # BBB permeability
        bbb_permeable = tpsa <= 90 and mw <= 400 and 1.0 <= logp <= 3.0
        details["bbb_permeable"] = bbb_permeable
        if bbb_permeable:
            distribution = 0.9

        # Egan egg model (intestinal absorption)
        if tpsa <= 131.6 and logp <= 5.88:
            distribution += 0.1
        else:
            distribution -= 0.1
            flags.append("Egan: poor intestinal absorption predicted")

        # Tissue accumulation risk
        if logp > 4:
            distribution -= 0.1
            flags.append("High LogP: tissue accumulation risk")

        # ── 3. Metabolism ───────────────────────────────────────

        metabolism = 0.7

        # CYP3A4 inhibition risk
        if n_aromatic_rings >= 3 and mw > 400:
            metabolism -= 0.2
            flags.append("CYP3A4 inhibition risk (aromatic + high MW)")
        elif n_aromatic_rings >= 4:
            metabolism -= 0.15
            flags.append("Multiple aromatic rings: CYP concern")

        # Metabolic stability sweet spot
        if 200 <= mw <= 450:
            metabolism += 0.1
        elif mw > 600:
            metabolism -= 0.15

        # ── 4. Excretion ────────────────────────────────────────

        excretion = 0.7

        # Renal clearance proxy
        if mw < 300 and logp < 2:
            excretion = 0.85
        elif mw > 500:
            excretion -= 0.15
            flags.append("High MW: biliary excretion likely")

        # Half-life proxy
        if 1 <= logp <= 3:
            excretion += 0.1

        # ── 5. Toxicity ─────────────────────────────────────────

        toxicity = 0.8  # Start optimistic

        # hERG liability (basic nitrogen + high LogP)
        smarts_basic_n = Chem.MolFromSmarts("[#7;+;!$([#7]~[#8])]")
        if smarts_basic_n is not None:
            has_basic_n = mol.HasSubstructMatch(smarts_basic_n)
            if has_basic_n and logp > 3.5:
                toxicity -= 0.25
                flags.append("hERG liability: basic nitrogen + high LogP")

        # PAINS check
        try:
            params = FilterCatalog.FilterCatalogParams()
            params.AddCatalog(
                FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS
            )
            catalog = FilterCatalog.FilterCatalog(params)
            if catalog.HasMatch(mol):
                toxicity -= 0.3
                flags.append("PAINS alert detected")
        except Exception:
            pass  # FilterCatalog not available in all RDKit builds

        # Reactive group alerts
        reactive_smarts = [
            ("[#6](=[#8])([F,Cl,Br,I])", "acyl halide"),
            ("[#7]=[#7]=[#7]", "azide"),
            ("[#8]-[#8]", "peroxide"),
            ("[#6]1[#6][#8]1", "epoxide"),
        ]
        for smarts_str, alert_name in reactive_smarts:
            pattern = Chem.MolFromSmarts(smarts_str)
            if pattern and mol.HasSubstructMatch(pattern):
                toxicity -= 0.2
                flags.append(f"Reactive group alert: {alert_name}")
                break

        # Ghose filter
        ghose_ok = 160 <= mw <= 480 and -0.4 <= logp <= 5.6 and 20 <= n_atoms <= 70
        details["ghose_pass"] = ghose_ok
        if not ghose_ok:
            flags.append("Ghose filter: outside drug-like range")

        # ── Composite ───────────────────────────────────────────

        absorption = max(0.0, min(1.0, absorption))
        distribution = max(0.0, min(1.0, distribution))
        metabolism = max(0.0, min(1.0, metabolism))
        excretion = max(0.0, min(1.0, excretion))
        toxicity = max(0.0, min(1.0, toxicity))

        # Weighted composite (toxicity gets extra weight)
        composite = (
            0.25 * absorption
            + 0.20 * distribution
            + 0.20 * metabolism
            + 0.15 * excretion
            + 0.20 * toxicity
        )

        return ADMETProfile(
            absorption=round(absorption, 3),
            distribution=round(distribution, 3),
            metabolism=round(metabolism, 3),
            excretion=round(excretion, 3),
            toxicity=round(toxicity, 3),
            composite=round(composite, 3),
            flags=flags,
            details=details,
        )

    except ImportError:
        logger.debug("RDKit not available for ADMET prediction.")
        return _fallback_admet(smiles)

    except Exception as e:
        logger.debug("ADMET prediction failed: %s", e)
        return _fallback_admet(smiles)


def _fallback_admet(smiles: str) -> ADMETProfile:
    """Minimal ADMET estimate when RDKit is unavailable."""
    length = len(smiles)
    if length < 20:
        score = 0.7
    elif length < 50:
        score = 0.6
    else:
        score = 0.4
    return ADMETProfile(
        composite=score,
        flags=["RDKit unavailable: heuristic only"],
    )
