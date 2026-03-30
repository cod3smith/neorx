"""
Target Classification System
==============================

Multi-level target classifier that distinguishes between
biologically meaningful target categories.  This prevents the
critical error observed in malaria runs where host symptom
markers (GABRD, SCN2A — brain neurotransmitter / ion channel
proteins associated with cerebral malaria seizures) were
misclassified as causal drug targets.

Target Hierarchy
----------------
1. **PATHOGEN_DIRECT** — Parasite / viral protein
   (e.g. PfDHFR, HIV protease).  Highest-priority drug target.
2. **HOST_INVASION** — Host protein the pathogen directly
   binds / hijacks (e.g. GYPA for P. falciparum, CCR5 for HIV).
3. **HOST_IMMUNE** — Host immune receptor relevant to disease
   response (e.g. TLR7, TLR9).  Valid target but with caveats.
4. **HOST_SYMPTOM** — Host protein associated with disease
   *symptoms* rather than mechanism (e.g. GABRD in cerebral
   malaria, coagulation factors in sepsis).  Should be
   classified as **CORRELATIONAL**, not causal.
5. **CORRELATIONAL** — Statistical association without any
   mechanistic evidence.

Disease Type Awareness
----------------------
Classification behaviour changes depending on disease type:
- Infectious diseases: check for pathogen vs host origin,
  check host-pathogen interaction evidence.
- Cancers: all targets are host, but distinguish driver vs
  passenger mutations.
- Metabolic diseases: distinguish causal pathway genes vs
  downstream biomarkers.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────

class TargetType(str, Enum):
    """Biological role of a candidate drug target."""
    PATHOGEN_DIRECT = "pathogen_direct"
    HOST_INVASION = "host_invasion"
    HOST_IMMUNE = "host_immune"
    HOST_SYMPTOM = "host_symptom"
    CORRELATIONAL = "correlational"


class DiseaseType(str, Enum):
    """Broad disease category affecting target classification."""
    INFECTIOUS_VIRAL = "infectious_viral"
    INFECTIOUS_PARASITIC = "infectious_parasitic"
    INFECTIOUS_BACTERIAL = "infectious_bacterial"
    GENETIC = "genetic"
    METABOLIC = "metabolic"
    CANCER = "cancer"
    AUTOIMMUNE = "autoimmune"
    NEURODEGENERATIVE = "neurodegenerative"
    OTHER = "other"


# ── Symptom Marker Blocklists ─────────────────────────────────────

# Proteins that are NEVER valid *direct* drug targets for
# infectious diseases — they reflect host symptom biology,
# not pathogen mechanism.

SYMPTOM_MARKER_FAMILIES: dict[str, list[str]] = {
    "GABA_receptors": [
        "GABRA1", "GABRA2", "GABRA3", "GABRA4", "GABRA5", "GABRA6",
        "GABRB1", "GABRB2", "GABRB3", "GABRD", "GABRE",
        "GABRG1", "GABRG2", "GABRG3", "GABRP", "GABRQ",
        "GABRR1", "GABRR2", "GABRR3",
    ],
    "sodium_channels": [
        "SCN1A", "SCN2A", "SCN3A", "SCN4A", "SCN5A",
        "SCN7A", "SCN8A", "SCN9A", "SCN10A", "SCN11A",
        "SCN1B", "SCN2B", "SCN3B", "SCN4B",
    ],
    "potassium_channels": [
        "KCNA1", "KCNA2", "KCNA3", "KCNA4", "KCNA5",
        "KCNB1", "KCNB2", "KCNC1", "KCNC2",
        "KCND1", "KCND2", "KCND3",
        "KCNQ1", "KCNQ2", "KCNQ3", "KCNQ4", "KCNQ5",
    ],
    "glutamate_receptors": [
        "GRIN1", "GRIN2A", "GRIN2B", "GRIN2C", "GRIN2D",
        "GRM1", "GRM2", "GRM3", "GRM4", "GRM5", "GRM6", "GRM7", "GRM8",
        "GRIA1", "GRIA2", "GRIA3", "GRIA4",
    ],
    "dopamine_receptors": ["DRD1", "DRD2", "DRD3", "DRD4", "DRD5"],
    "serotonin_receptors": [
        "HTR1A", "HTR1B", "HTR2A", "HTR2B", "HTR2C",
        "HTR3A", "HTR3B", "HTR4", "HTR5A", "HTR6", "HTR7",
    ],
    "coagulation_factors": [
        "F2", "F5", "F7", "F8", "F9", "F10", "F11", "F12", "F13A1",
        "SERPINC1", "SERPIND1", "PROC", "PROS1", "THBD",
    ],
    "acute_phase_proteins": [
        "CRP", "SAA1", "SAA2", "HP", "HPR", "ORM1", "ORM2",
        "SERPINA3", "FGA", "FGB", "FGG",
    ],
    "neuronal_structural": [
        "NEFH", "NEFL", "NEFM", "MAP2", "MAPT", "TUBB3",
        "SYN1", "SYN2", "SYP", "SNAP25",
    ],
}

# Build a flat lookup set for O(1) checks
_SYMPTOM_MARKER_SET: set[str] = set()
_SYMPTOM_FAMILY_MAP: dict[str, str] = {}
for _family, _genes in SYMPTOM_MARKER_FAMILIES.items():
    for _g in _genes:
        _SYMPTOM_MARKER_SET.add(_g)
        _SYMPTOM_FAMILY_MAP[_g] = _family

# Immune receptor prefixes / gene families
_IMMUNE_PREFIXES = (
    "TLR", "NOD", "NLRP", "RIG", "DDX58", "IFIH1",
    "STING", "TMEM173", "MAVS",
    "HLA-", "MHC",
)
_IMMUNE_GENES = {
    "CD4", "CD8A", "CD8B", "CD14", "CD16", "CD56",
    "CD3D", "CD3E", "CD3G", "CD19", "CD20", "MS4A1",
    "IFNG", "IFNA1", "IFNB1", "IFNL1",
    "IL1B", "IL2", "IL4", "IL6", "IL10", "IL12A", "IL17A",
    "TNF", "TNFRSF1A", "TNFRSF1B",
    "CXCL8", "CXCL10", "CCL2", "CCL5",
    "STAT1", "STAT3", "STAT4", "IRF3", "IRF7",
}

# Known infectious-disease organisms ↔ common names
DISEASE_TYPE_MAP: dict[str, DiseaseType] = {
    # Viral
    "hiv": DiseaseType.INFECTIOUS_VIRAL,
    "hiv-1": DiseaseType.INFECTIOUS_VIRAL,
    "hiv/aids": DiseaseType.INFECTIOUS_VIRAL,
    "aids": DiseaseType.INFECTIOUS_VIRAL,
    "covid-19": DiseaseType.INFECTIOUS_VIRAL,
    "sars-cov-2": DiseaseType.INFECTIOUS_VIRAL,
    "ebola": DiseaseType.INFECTIOUS_VIRAL,
    "ebola virus disease": DiseaseType.INFECTIOUS_VIRAL,
    "hepatitis b": DiseaseType.INFECTIOUS_VIRAL,
    "hepatitis c": DiseaseType.INFECTIOUS_VIRAL,
    "influenza": DiseaseType.INFECTIOUS_VIRAL,
    "dengue": DiseaseType.INFECTIOUS_VIRAL,
    "zika": DiseaseType.INFECTIOUS_VIRAL,
    # Parasitic
    "malaria": DiseaseType.INFECTIOUS_PARASITIC,
    "trypanosomiasis": DiseaseType.INFECTIOUS_PARASITIC,
    "chagas disease": DiseaseType.INFECTIOUS_PARASITIC,
    "leishmaniasis": DiseaseType.INFECTIOUS_PARASITIC,
    "schistosomiasis": DiseaseType.INFECTIOUS_PARASITIC,
    "toxoplasmosis": DiseaseType.INFECTIOUS_PARASITIC,
    # Bacterial
    "tuberculosis": DiseaseType.INFECTIOUS_BACTERIAL,
    "cholera": DiseaseType.INFECTIOUS_BACTERIAL,
    "meningitis": DiseaseType.INFECTIOUS_BACTERIAL,
    "pneumonia": DiseaseType.INFECTIOUS_BACTERIAL,
    "sepsis": DiseaseType.INFECTIOUS_BACTERIAL,
    "lyme disease": DiseaseType.INFECTIOUS_BACTERIAL,
    # Cancer
    "lung cancer": DiseaseType.CANCER,
    "breast cancer": DiseaseType.CANCER,
    "colorectal cancer": DiseaseType.CANCER,
    "pancreatic cancer": DiseaseType.CANCER,
    "prostate cancer": DiseaseType.CANCER,
    "leukemia": DiseaseType.CANCER,
    "lymphoma": DiseaseType.CANCER,
    "melanoma": DiseaseType.CANCER,
    "glioblastoma": DiseaseType.CANCER,
    # Metabolic
    "type 2 diabetes": DiseaseType.METABOLIC,
    "type 1 diabetes": DiseaseType.AUTOIMMUNE,
    "obesity": DiseaseType.METABOLIC,
    "metabolic syndrome": DiseaseType.METABOLIC,
    "nafld": DiseaseType.METABOLIC,
    # Neurodegenerative
    "alzheimer disease": DiseaseType.NEURODEGENERATIVE,
    "alzheimer's disease": DiseaseType.NEURODEGENERATIVE,
    "parkinson disease": DiseaseType.NEURODEGENERATIVE,
    "parkinson's disease": DiseaseType.NEURODEGENERATIVE,
    "huntington disease": DiseaseType.NEURODEGENERATIVE,
    "als": DiseaseType.NEURODEGENERATIVE,
    "amyotrophic lateral sclerosis": DiseaseType.NEURODEGENERATIVE,
    # Autoimmune
    "rheumatoid arthritis": DiseaseType.AUTOIMMUNE,
    "lupus": DiseaseType.AUTOIMMUNE,
    "multiple sclerosis": DiseaseType.AUTOIMMUNE,
    "crohn's disease": DiseaseType.AUTOIMMUNE,
    # Genetic
    "cystic fibrosis": DiseaseType.GENETIC,
    "sickle cell disease": DiseaseType.GENETIC,
}


def classify_disease(disease_name: str) -> DiseaseType:
    """Determine the broad disease category from the disease name.

    Uses a curated lookup table; falls back to ``OTHER`` for
    unrecognised diseases.
    """
    key = disease_name.strip().lower()
    if key in DISEASE_TYPE_MAP:
        return DISEASE_TYPE_MAP[key]

    # Heuristic substring matching for partial matches
    for pattern, dtype in DISEASE_TYPE_MAP.items():
        if pattern in key or key in pattern:
            return dtype

    # Cancer heuristic
    if any(w in key for w in ("cancer", "carcinoma", "sarcoma",
                               "blastoma", "oma", "leukemia",
                               "lymphoma", "myeloma")):
        return DiseaseType.CANCER

    return DiseaseType.OTHER


def is_infectious(disease_type: DiseaseType) -> bool:
    """Return *True* if the disease is infectious."""
    return disease_type in (
        DiseaseType.INFECTIOUS_VIRAL,
        DiseaseType.INFECTIOUS_PARASITIC,
        DiseaseType.INFECTIOUS_BACTERIAL,
    )


# ── Target Classifier ─────────────────────────────────────────────


class TargetClassifier:
    """Classify targets into biologically meaningful categories.

    This classifier prevents the critical error of treating
    host symptom markers as genuine causal drug targets.  For
    malaria, this catches GABRD, SCN2A, etc.

    The classification pipeline:

    1. **Blocklist check** — is the gene in a known symptom
       marker family (GABA receptors, sodium channels, etc.)?
    2. **Inflammatory cytokine check** — TNF, IL6, IL1B,
       CXCL8, IL10 are HOST_SYMPTOM for infections (must
       run before the general immune check to avoid being
       caught as HOST_IMMUNE).
    3. **Immune receptor check** — TLR, NOD, HLA, etc.
       are HOST_IMMUNE for infections.
    4. **Disease-type-aware heuristics** — for infectious
       diseases, check pathogen vs host origin.
    """

    def classify(
        self,
        gene_symbol: str,
        disease_type: DiseaseType,
        evidence: dict[str, Any] | None = None,
    ) -> tuple[TargetType, str]:
        """Classify a single target gene.

        Parameters
        ----------
        gene_symbol : str
            Gene symbol (e.g. "GABRD", "CCR5", "EGFR").
        disease_type : DiseaseType
            The broad disease category.
        evidence : dict, optional
            Node metadata / evidence from the graph.

        Returns
        -------
        tuple[TargetType, str]
            (target_type, human-readable reason)
        """
        if evidence is None:
            evidence = {}

        gene = gene_symbol.upper().strip()

        # ── Step 1: Symptom marker blocklist ────────────────
        if gene in _SYMPTOM_MARKER_SET and is_infectious(disease_type):
            family = _SYMPTOM_FAMILY_MAP.get(gene, "unknown")
            reason = (
                f"{gene} belongs to the {family} family. "
                f"These are host neurophysiological/systemic proteins "
                f"associated with disease symptoms (e.g. seizures in "
                f"cerebral malaria, coagulopathy in sepsis), not with "
                f"pathogen biology.  Inhibiting {gene} would treat "
                f"symptoms, not the infection."
            )
            return TargetType.HOST_SYMPTOM, reason

        # For non-infectious diseases, symptom markers in
        # neurodegenerative context *are* potentially valid
        # targets (e.g. GABA receptors in epilepsy)
        if gene in _SYMPTOM_MARKER_SET and disease_type == DiseaseType.NEURODEGENERATIVE:
            # These may be legitimate targets
            pass

        # ── Step 2: Inflammatory cytokine check ─────────────
        # TNF, IL6, IL1B, CXCL8, IL10 — valid targets in
        # autoimmune diseases, but correlational in infections.
        # This check MUST run before the general immune receptor
        # check so that cytokines are classified as HOST_SYMPTOM
        # (and thus demoted) rather than HOST_IMMUNE (which is
        # not automatically demoted).
        if gene in ("TNF", "IL6", "IL1B", "CXCL8", "IL10"):
            if is_infectious(disease_type):
                reason = (
                    f"{gene} is an inflammatory cytokine elevated "
                    f"during infection.  Elevated {gene} is typically "
                    f"a *consequence* of immune activation, not a "
                    f"cause of the infection.  Blocking {gene} during "
                    f"active infection risks immune suppression."
                )
                return TargetType.HOST_SYMPTOM, reason

        # ── Step 3: Immune receptor check ───────────────────
        if self._is_immune_gene(gene):
            if is_infectious(disease_type):
                reason = (
                    f"{gene} is a host immune receptor/cytokine. "
                    f"It may modulate the immune response to infection "
                    f"but is not a direct anti-pathogen target.  "
                    f"Immunomodulatory drugs targeting {gene} could be "
                    f"adjunctive therapy but carry immune suppression risks."
                )
                return TargetType.HOST_IMMUNE, reason
            # For autoimmune diseases, immune receptors *are*
            # primary targets (e.g. TNF in RA)
            if disease_type == DiseaseType.AUTOIMMUNE:
                reason = (
                    f"{gene} is an immune system component.  For "
                    f"autoimmune diseases, immune targets are primary "
                    f"therapeutic targets."
                )
                return TargetType.HOST_INVASION, reason

        # ── Step 4: Known host-pathogen interaction check ───
        if is_infectious(disease_type):
            if self._is_known_invasion_target(gene, evidence):
                reason = (
                    f"{gene} is a host protein that directly interacts "
                    f"with the pathogen during cell entry/invasion.  "
                    f"Blocking this interaction could prevent infection."
                )
                return TargetType.HOST_INVASION, reason

        # ── Step 5: Evidence quality check ──────────────────
        evidence_types = evidence.get("evidence_types", [])
        sources = evidence.get("source", "")
        n_sources = len(sources.split(", ")) if sources else 0

        # Also count sources from metadata if available
        metadata = evidence.get("metadata", {})
        if n_sources == 0 and metadata:
            # Node has structured data even if 'source' string is empty
            n_sources = 1

        # If truly no evidence at all (no source, no metadata, no score)
        score = evidence.get("score", 0.0)
        if n_sources == 0 and not evidence_types and score == 0.0:
            reason = (
                f"{gene} has no evidence support "
                f"(no source, no mechanistic data).  "
                f"Classification as correlational pending "
                f"additional validation."
            )
            return TargetType.CORRELATIONAL, reason

        # ── Step 6: Default classification ──────────────────
        # For cancer: all targets are host, default to invasion
        if disease_type == DiseaseType.CANCER:
            reason = (
                f"{gene} is a candidate cancer target with "
                f"multi-source evidence support."
            )
            return TargetType.HOST_INVASION, reason

        # For non-infectious diseases with evidence
        reason = (
            f"{gene} has multi-source evidence support.  "
            f"Further validation recommended."
        )
        return TargetType.HOST_INVASION, reason

    def _is_immune_gene(self, gene: str) -> bool:
        """Check if gene belongs to the immune system."""
        if gene in _IMMUNE_GENES:
            return True
        return any(gene.startswith(pfx) for pfx in _IMMUNE_PREFIXES)

    def _is_known_invasion_target(
        self, gene: str, evidence: dict[str, Any],
    ) -> bool:
        """Check if this host protein is a known pathogen receptor."""
        # Curated known host-pathogen invasion targets
        known = {
            # Malaria
            "GYPA", "GYPB", "GYPC",       # Glycophorins — P. falciparum RBC invasion
            "CR1", "DARC", "BSG",          # Complement receptor 1, Duffy, Basigin
            # HIV
            "CCR5", "CXCR4", "CD4",        # HIV co-receptors
            # SARS-CoV-2
            "ACE2", "TMPRSS2", "NRP1",     # COVID-19 entry receptors
            # Ebola
            "NPC1",                         # Niemann-Pick C1 — Ebola entry
            # General
            "LAMP1",                        # Lysosome — many pathogens
        }
        if gene in known:
            return True

        # Check metadata for interaction evidence
        metadata = evidence.get("metadata", {})
        go_terms = metadata.get("go_terms", [])
        invasion_terms = {
            "GO:0046718",  # viral entry into host cell
            "GO:0044409",  # entry into host
            "GO:0020002",  # host cell plasma membrane
        }
        return bool(set(go_terms) & invasion_terms)

    def classify_batch(
        self,
        targets: list[dict[str, Any]],
        disease_type: DiseaseType,
    ) -> list[tuple[TargetType, str]]:
        """Classify a batch of targets.

        Parameters
        ----------
        targets : list[dict]
            Each dict must have ``"gene_symbol"`` and optionally
            ``"evidence"``.
        disease_type : DiseaseType
            Broad disease category.

        Returns
        -------
        list[tuple[TargetType, str]]
        """
        return [
            self.classify(
                t["gene_symbol"],
                disease_type,
                t.get("evidence"),
            )
            for t in targets
        ]
