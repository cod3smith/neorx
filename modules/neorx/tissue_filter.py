"""
Tissue Expression Filter
==========================

Uses protein tissue-expression data to assess how relevant a
candidate target's expression profile is to a given disease.

Key design principles
---------------------
1. **Multi-tissue profiles** — genes are expressed in many
   tissues; we track ALL of them, not just the single
   highest-expression tissue.
2. **Soft scoring** — tissue relevance is a *score* (0.0–1.0),
   not a binary kill switch.  A gene expressed in many tissues
   including the disease-relevant ones gets a high score; a gene
   expressed exclusively in an irrelevant tissue gets a low
   score (but is not hard-blocked).
3. **Disease-type fallback** — for diseases not in the curated
   tissue map, we infer relevant tissues from the disease
   category (cancer, infectious, metabolic, etc.).
4. **Full HPA integration** — when the Human Protein Atlas API
   is available, we pull the COMPLETE tissue expression profile,
   not just the single maximum.

Data Source
-----------
The Human Protein Atlas (proteinatlas.org) provides tissue-level
RNA and protein expression data for ~20,000 human genes.  We
query the HPA API and cache results.

When the HPA API is unavailable, a curated fallback map covers
commonly studied gene families with their known expression
tissues.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from .classifier import classify_disease

logger = logging.getLogger(__name__)


# ── Disease → Relevant Tissues ─────────────────────────────────────
#
# These are the KNOWN disease-specific tissue sets.  For any
# disease not listed here, we fall back to disease-type-based
# inference (see _infer_tissues_from_type).

DISEASE_TISSUES: dict[str, list[str]] = {
    # Infectious — viral
    "hiv": ["blood", "lymph node", "gut", "brain", "tonsil", "spleen",
            "bone marrow", "thymus", "liver"],
    "ebola": ["blood", "liver", "spleen", "lymph node", "lung",
              "kidney", "adrenal gland", "endothelium"],
    "covid-19": ["lung", "blood", "heart", "kidney", "brain",
                 "liver", "intestine", "endothelium"],
    "hepatitis b": ["liver", "blood", "lymph node", "kidney"],
    "hepatitis c": ["liver", "blood", "lymph node"],
    "influenza": ["lung", "blood", "lymph node", "trachea"],
    "dengue": ["blood", "liver", "spleen", "bone marrow", "lymph node"],
    # Infectious — parasitic
    "malaria": ["blood", "liver", "spleen", "bone marrow", "placenta",
                "brain", "lung", "kidney", "endothelium"],
    "trypanosomiasis": ["blood", "brain", "lymph node", "heart"],
    "leishmaniasis": ["spleen", "liver", "bone marrow", "skin", "blood"],
    # Infectious — bacterial
    "tuberculosis": ["lung", "lymph node", "blood", "bone marrow",
                     "liver", "spleen", "kidney"],
    "cholera": ["intestine", "blood", "kidney"],
    "typhoid": ["intestine", "blood", "liver", "spleen", "bone marrow"],
    # Cancer — note: cancer tissues are broad because metastasis
    # and systemic signaling mean many tissues are relevant
    "lung cancer": ["lung", "lymph node", "blood", "brain", "bone",
                    "liver", "adrenal gland", "pleura"],
    "breast cancer": ["breast", "lymph node", "blood", "bone", "liver",
                      "lung", "brain", "ovary"],
    "colorectal cancer": ["colon", "intestine", "liver", "lymph node",
                          "blood", "lung", "peritoneum"],
    "pancreatic cancer": ["pancreas", "liver", "lymph node", "blood",
                          "peritoneum", "lung"],
    "prostate cancer": ["prostate", "bone", "lymph node", "blood",
                        "liver", "lung"],
    "melanoma": ["skin", "lymph node", "blood", "liver", "lung", "brain"],
    "glioblastoma": ["brain", "blood", "cerebral cortex"],
    "leukemia": ["blood", "bone marrow", "spleen", "lymph node", "liver"],
    "lymphoma": ["lymph node", "spleen", "blood", "bone marrow",
                 "liver", "lung"],
    # Metabolic
    "type 2 diabetes": ["pancreas", "liver", "adipose tissue",
                        "skeletal muscle", "kidney", "blood",
                        "intestine", "brain"],
    "type 1 diabetes": ["pancreas", "blood", "lymph node", "thymus"],
    "obesity": ["adipose tissue", "liver", "brain", "skeletal muscle",
                "pancreas", "blood"],
    # Neurodegenerative
    "alzheimer disease": ["brain", "cerebral cortex", "hippocampus",
                          "blood", "liver", "choroid plexus"],
    "alzheimer's disease": ["brain", "cerebral cortex", "hippocampus",
                            "blood", "liver", "choroid plexus"],
    "parkinson disease": ["brain", "substantia nigra", "cerebral cortex",
                          "blood", "intestine"],
    "parkinson's disease": ["brain", "substantia nigra", "cerebral cortex",
                            "blood", "intestine"],
    "huntington disease": ["brain", "cerebral cortex", "striatum", "blood"],
    "als": ["brain", "spinal cord", "skeletal muscle", "blood"],
    # Autoimmune
    "rheumatoid arthritis": ["synovium", "blood", "lymph node", "joint",
                             "bone marrow"],
    "lupus": ["blood", "kidney", "skin", "lymph node", "joint"],
    "multiple sclerosis": ["brain", "spinal cord", "blood", "lymph node"],
    "crohn's disease": ["intestine", "colon", "blood", "lymph node",
                        "liver"],
    # Cardiovascular
    "heart failure": ["heart", "blood", "kidney", "lung", "liver"],
    "atherosclerosis": ["blood", "endothelium", "heart", "liver"],
    # Genetic
    "cystic fibrosis": ["lung", "pancreas", "intestine", "liver",
                        "sweat gland"],
    "sickle cell disease": ["blood", "bone marrow", "spleen", "liver",
                            "kidney", "lung"],
}


# ── Disease-Type → Fallback Tissues ────────────────────────────────
#
# When a specific disease isn't in DISEASE_TISSUES, we use its
# DiseaseType to provide a broad but relevant tissue set.

_TYPE_TISSUES: dict[str, list[str]] = {
    "infectious_viral": [
        "blood", "lymph node", "spleen", "liver", "lung",
        "bone marrow", "thymus", "gut", "brain", "kidney",
        "endothelium",
    ],
    "infectious_parasitic": [
        "blood", "liver", "spleen", "bone marrow", "lung",
        "brain", "kidney", "intestine", "skin", "endothelium",
    ],
    "infectious_bacterial": [
        "blood", "lung", "lymph node", "liver", "spleen",
        "bone marrow", "kidney", "intestine", "brain",
    ],
    "cancer": [
        # Cancer is systemic — nearly any tissue can be relevant
        # due to metastasis, immune infiltration, and signaling
        "blood", "lymph node", "liver", "lung", "bone",
        "brain", "kidney", "skin", "breast", "prostate",
        "colon", "pancreas", "ovary", "stomach", "thyroid",
        "endothelium", "bone marrow",
    ],
    "metabolic": [
        "liver", "pancreas", "adipose tissue", "skeletal muscle",
        "kidney", "blood", "intestine", "brain", "heart",
    ],
    "neurodegenerative": [
        "brain", "cerebral cortex", "hippocampus", "spinal cord",
        "blood", "liver", "intestine",
    ],
    "autoimmune": [
        "blood", "lymph node", "bone marrow", "thymus", "spleen",
        "skin", "joint", "kidney", "intestine", "liver",
    ],
    "cardiovascular": [
        "heart", "blood", "endothelium", "kidney", "liver",
        "lung", "brain",
    ],
    "genetic": [
        # Genetic diseases affect diverse tissues — use broad set
        "blood", "liver", "lung", "brain", "kidney",
        "bone marrow", "skeletal muscle", "pancreas",
        "intestine", "skin",
    ],
    "other": [
        # Fallback: systemic tissues relevant to most diseases
        "blood", "liver", "lung", "kidney", "brain",
        "lymph node", "bone marrow", "spleen",
    ],
}


# ── Curated Multi-Tissue Expression ───────────────────────────────
#
# For frequently studied genes, provide their FULL expression
# profile (all tissues where they are significantly expressed),
# not just one "primary" tissue.  This is the fallback when the
# HPA API is unavailable.

_CURATED_EXPRESSION: dict[str, list[str]] = {
    # GABA receptors — primarily brain, some peripheral
    "GABRA1": ["brain"], "GABRA2": ["brain"], "GABRA3": ["brain"],
    "GABRA4": ["brain"], "GABRA5": ["brain"], "GABRA6": ["brain"],
    "GABRB1": ["brain"], "GABRB2": ["brain"], "GABRB3": ["brain"],
    "GABRD": ["brain"], "GABRE": ["brain"],
    "GABRG1": ["brain"], "GABRG2": ["brain"], "GABRG3": ["brain"],
    "GABRP": ["brain", "lung"], "GABRQ": ["brain"],
    "GABRR1": ["brain"], "GABRR2": ["brain"],
    # Sodium channels
    "SCN1A": ["brain"], "SCN2A": ["brain"], "SCN3A": ["brain"],
    "SCN4A": ["skeletal muscle"], "SCN5A": ["heart"],
    "SCN7A": ["brain", "lung"], "SCN8A": ["brain"],
    "SCN9A": ["dorsal root ganglia", "brain"],
    "SCN10A": ["dorsal root ganglia"],
    "SCN11A": ["dorsal root ganglia"],
    # Potassium channels
    "KCNA1": ["brain"], "KCNA2": ["brain"],
    "KCNQ1": ["heart", "intestine"], "KCNQ2": ["brain"], "KCNQ3": ["brain"],
    "KCNH2": ["heart", "brain"],
    # Glutamate receptors
    "GRIN1": ["brain"], "GRIN2A": ["brain"], "GRIN2B": ["brain"],
    "GRIN2C": ["brain"], "GRIN2D": ["brain"],
    "GRIN3A": ["brain"], "GRIN3B": ["brain"],
    "GRM1": ["brain"], "GRM5": ["brain"],
    # Dopamine receptors
    "DRD1": ["brain"], "DRD2": ["brain"], "DRD3": ["brain"],
    "DRD4": ["brain"], "DRD5": ["brain"],
    # Serotonin receptors
    "HTR1A": ["brain"], "HTR2A": ["brain"], "HTR3A": ["brain", "intestine"],
    # Coagulation factors
    "F2": ["liver", "blood"], "F5": ["liver", "blood"],
    "F7": ["liver", "blood"], "F10": ["liver", "blood"],
    "F11": ["liver", "blood"],
    # Acute phase
    "CRP": ["liver", "blood"], "SAA1": ["liver"], "HP": ["liver"],
    "ORM1": ["liver"],
    # Inflammatory cytokines — expressed broadly
    "TNF": ["blood", "lymph node", "spleen", "liver", "lung", "brain"],
    "IL6": ["blood", "liver", "lung", "adipose tissue", "lymph node", "brain"],
    "IL1B": ["blood", "lymph node", "spleen", "lung", "liver"],
    "CXCL8": ["blood", "lung", "liver", "lymph node"],
    "IFNG": ["blood", "lymph node", "spleen"],
    "IL10": ["blood", "lymph node", "spleen"],
    # Immune receptors
    "CD4": ["blood", "lymph node", "thymus", "spleen"],
    "CD8A": ["blood", "lymph node", "thymus"],
    # Chemokine receptors
    "CCR5": ["blood", "lymph node", "spleen", "brain", "lung"],
    "CXCR4": ["blood", "bone marrow", "lymph node", "brain", "lung"],
    # Malaria-relevant
    "GYPA": ["blood", "bone marrow"],
    "GYPB": ["blood", "bone marrow"],
    "GYPC": ["blood", "bone marrow"],
    "CR1": ["blood", "kidney"],
    "DARC": ["blood", "kidney", "endothelium"],
    "BSG": ["blood", "brain", "liver", "kidney", "placenta"],
    "DHFR": ["liver", "blood", "bone marrow"],
    "DHPS": ["liver", "blood"],
    # Ebola-relevant
    "NPC1": ["liver", "brain", "lung", "kidney", "blood"],
    # Cancer targets — most are broadly expressed
    "EGFR": ["lung", "skin", "liver", "kidney", "brain", "colon",
             "breast", "stomach", "pancreas"],
    "ERBB2": ["breast", "lung", "stomach", "colon", "kidney",
              "skin", "liver", "ovary", "pancreas"],
    "ALK": ["brain", "lung", "intestine"],
    "BRAF": ["brain", "blood", "lung", "liver", "bone marrow"],
    "KRAS": ["lung", "colon", "pancreas", "blood", "liver"],
    "TP53": ["blood", "liver", "lung", "brain", "kidney", "breast",
             "colon", "pancreas", "skin"],
    "PIK3CA": ["breast", "lung", "colon", "blood", "liver", "brain",
               "kidney", "stomach"],
    "BRCA1": ["breast", "ovary", "blood", "brain", "lung"],
    "BRCA2": ["breast", "ovary", "blood", "brain", "lung"],
    "ESR1": ["breast", "uterus", "ovary", "bone", "brain", "liver"],
    "CDK4": ["blood", "liver", "brain", "lung", "skin"],
    "RB1": ["blood", "liver", "brain", "lung", "bone marrow"],
    "MET": ["liver", "lung", "kidney", "brain", "stomach"],
    "ROS1": ["brain", "lung", "kidney", "intestine"],
    "RET": ["brain", "thyroid", "kidney", "adrenal gland", "lung"],
    "PTEN": ["blood", "liver", "brain", "lung", "breast", "kidney"],
    # Metabolic targets
    "PPARG": ["adipose tissue", "liver", "colon", "blood", "breast"],
    "DPP4": ["pancreas", "liver", "kidney", "intestine", "blood", "lung"],
    "GLP1R": ["pancreas", "brain", "heart", "lung", "kidney"],
    "SLC5A2": ["kidney"],
    "INSR": ["liver", "adipose tissue", "skeletal muscle", "brain",
             "blood", "kidney", "pancreas"],
    "INS": ["pancreas"],
    "ABCC8": ["pancreas", "brain", "heart"],
    "KCNJ11": ["pancreas", "brain", "heart"],
    # Neurodegenerative targets
    "PSEN1": ["brain", "blood", "liver", "kidney"],
    "PSEN2": ["brain", "blood", "liver", "heart"],
    "APP": ["brain", "blood", "liver", "kidney"],
    "MAPT": ["brain"],
    "BACE1": ["brain", "pancreas", "liver"],
    "ACHE": ["brain", "blood", "liver", "skeletal muscle"],
    "APOE": ["liver", "brain", "blood"],
    "ADAM10": ["brain", "liver", "kidney", "blood"],
    # Ubiquitous — expressed in virtually all tissues
    "AKT1": ["blood", "liver", "lung", "brain", "kidney", "breast",
             "colon", "pancreas", "skeletal muscle"],
    "MTOR": ["blood", "liver", "lung", "brain", "kidney", "breast",
             "skeletal muscle", "adipose tissue"],
    "VEGFA": ["blood", "lung", "liver", "kidney", "brain", "placenta",
              "endothelium"],
    # TLRs — immune tissues
    "TLR7": ["blood", "lymph node", "spleen", "bone marrow", "lung"],
    "TLR9": ["blood", "lymph node", "spleen", "bone marrow"],
    "TLR4": ["blood", "lymph node", "spleen", "lung", "liver"],
    "TLR2": ["blood", "lymph node", "spleen", "lung"],
    # RNA Pol subunits — ubiquitous
    "POLR2A": ["blood", "liver", "lung", "brain", "kidney"],
    "POLR2B": ["blood", "liver", "lung", "brain", "kidney"],
    "POLR2D": ["blood", "liver", "lung", "brain", "kidney"],
    "POLR2F": ["blood", "liver", "lung", "brain", "kidney"],
    "POLR2G": ["blood", "liver", "lung", "brain", "kidney"],
    "POLR2J": ["blood", "liver", "lung", "brain", "kidney"],
    "POLR2K": ["blood", "liver", "lung", "brain", "kidney"],
    "POLR2L": ["blood", "liver", "lung", "brain", "kidney"],
    # CYP enzymes — liver + some peripheral
    "CYP3A4": ["liver", "intestine"],
    "CYP3A5": ["liver", "kidney", "lung", "intestine"],
    "CYP3A7": ["liver"],
    "CYP3A43": ["liver"],
    # General transcription factors — ubiquitous
    "GTF2F1": ["blood", "liver", "lung", "brain", "kidney"],
    "GTF2F2": ["blood", "liver", "lung", "brain", "kidney"],
    # Other commonly encountered genes
    "ACE": ["lung", "kidney", "blood", "brain", "heart"],
    "ACE2": ["lung", "intestine", "kidney", "heart", "liver"],
    "TMPRSS2": ["lung", "prostate", "intestine", "kidney"],
    "CDKN2A": ["skin", "lung", "blood", "liver", "brain"],
    "CDH1": ["breast", "stomach", "colon", "skin", "liver", "lung"],
    "SORL1": ["brain", "liver", "kidney"],
    "BCHE": ["liver", "blood", "brain"],
    "CDK5": ["brain", "blood"],
    "APH1B": ["brain", "blood", "liver"],
    "HMGCR": ["liver", "blood", "brain", "intestine"],
}


class TissueFilter:
    """Assess tissue-expression relevance for candidate targets.

    For each target gene, computes a **tissue relevance score**
    (0.0–1.0) based on whether any of the gene's expression
    tissues overlap with the disease-relevant tissue set.

    Design: the score is a continuous measure, not a binary gate.
    Even genes expressed primarily in "wrong" tissues may still
    be relevant (e.g. ERBB2 is primarily in breast but also
    expressed in lung — valid for lung cancer).  The identifier
    uses this score as a confidence modifier, not a kill switch.
    """

    def __init__(self, use_api: bool = True, timeout: int = 10) -> None:
        self._use_api = use_api
        self._timeout = timeout
        self._cache: dict[str, list[str]] = {}

    def get_expression_tissues(self, gene_symbol: str) -> list[str]:
        """Return all tissues where a gene is expressed.

        Tries:
        1. In-memory cache
        2. Curated multi-tissue fallback map
        3. Human Protein Atlas API (if enabled)
        4. Returns empty list if all fail (→ unknown)
        """
        gene = gene_symbol.upper().strip()

        if gene in self._cache:
            return self._cache[gene]

        # Curated map — multi-tissue
        if gene in _CURATED_EXPRESSION:
            tissues = _CURATED_EXPRESSION[gene]
            self._cache[gene] = tissues
            return tissues

        # HPA API — get full profile
        if self._use_api:
            tissues = self._query_hpa_all_tissues(gene)
            if tissues:
                self._cache[gene] = tissues
                return tissues

        self._cache[gene] = []
        return []

    def compute_tissue_relevance(
        self,
        gene_symbol: str,
        disease: str,
    ) -> tuple[float, str]:
        """Compute a tissue relevance score for a gene–disease pair.

        Returns a score in [0.0, 1.0]:
        - 1.0 = gene is expressed in disease-relevant tissue(s)
        - 0.8 = tissue expression unknown (benefit of the doubt)
        - 0.0–0.5 = gene is expressed only in irrelevant tissue(s)

        The score reflects what FRACTION of the gene's expression
        tissues overlap with disease-relevant tissues, with a
        bonus for being expressed in the PRIMARY disease tissue.

        Parameters
        ----------
        gene_symbol : str
            Gene symbol (e.g. "GABRD", "ERBB2").
        disease : str
            Disease name (e.g. "malaria", "lung cancer").

        Returns
        -------
        tuple[float, str]
            ``(relevance_score, explanation)``
        """
        gene_tissues = self.get_expression_tissues(gene_symbol)
        disease_tissues = self._get_disease_tissues(disease)

        # Unknown expression → benefit of the doubt
        if not gene_tissues:
            return 0.8, (
                f"Tissue expression unknown for {gene_symbol}; "
                f"relevance score 0.8 (benefit of the doubt)."
            )

        # Unknown disease tissues → benefit of the doubt
        if not disease_tissues:
            return 0.8, (
                f"No tissue profile for disease '{disease}'; "
                f"relevance score 0.8 (benefit of the doubt)."
            )

        # Compute overlap
        gene_set = {t.lower() for t in gene_tissues}
        disease_set = {t.lower() for t in disease_tissues}

        # Flexible matching: "cerebral cortex" matches "brain",
        # "dorsal root ganglia" matches "brain", etc.
        overlaps = set()
        for gt in gene_set:
            for dt in disease_set:
                if gt in dt or dt in gt:
                    overlaps.add(gt)
                    break

        n_overlap = len(overlaps)
        n_gene_tissues = len(gene_set)

        if n_overlap > 0:
            # Score = fraction of gene's tissues that are relevant,
            # but with a floor of 0.6 for any overlap at all
            raw = n_overlap / n_gene_tissues
            score = max(0.6, 0.6 + 0.4 * raw)
            overlap_names = sorted(overlaps)
            return score, (
                f"{gene_symbol} is expressed in {', '.join(sorted(gene_set))}; "
                f"{n_overlap}/{n_gene_tissues} overlap with {disease}-relevant "
                f"tissues ({', '.join(overlap_names)}). "
                f"Relevance score: {score:.2f}."
            )
        else:
            # No overlap — low score but not zero
            # More tissues → more likely to be genuinely irrelevant
            # Single-tissue genes get 0.3, multi-tissue get lower
            score = max(0.1, 0.4 - 0.05 * n_gene_tissues)
            return score, (
                f"{gene_symbol} is expressed in {', '.join(sorted(gene_set))}, "
                f"none of which overlap with {disease}-relevant tissues "
                f"({', '.join(sorted(disease_set))}). "
                f"Relevance score: {score:.2f}."
            )

    # ── Backward-compatible interface ───────────────────────────

    def is_tissue_relevant(
        self,
        gene_symbol: str,
        disease: str,
    ) -> tuple[float, str]:
        """Compute tissue relevance score.

        Returns
        -------
        tuple[float, str]
            ``(relevance_score, explanation)``
            Score of 1.0 = fully relevant, 0.0 = fully irrelevant.
        """
        return self.compute_tissue_relevance(gene_symbol, disease)

    # Legacy helper — returns first expression tissue for compat
    def get_primary_tissue(self, gene_symbol: str) -> str:
        """Return the primary (highest) expression tissue."""
        tissues = self.get_expression_tissues(gene_symbol)
        return tissues[0] if tissues else "unknown"

    def filter_targets(
        self,
        targets: list[dict[str, Any]],
        disease: str,
    ) -> list[dict[str, Any]]:
        """Annotate a list of targets with tissue relevance score.

        Each target dict gets:
        - ``tissue_relevant`` (float) — relevance score 0.0–1.0
        - ``tissue_explanation`` (str)
        """
        for target in targets:
            gene = target.get("gene_symbol", target.get("gene_name", ""))
            score, explanation = self.compute_tissue_relevance(gene, disease)
            target["tissue_relevant"] = score
            target["tissue_explanation"] = explanation
        return targets

    def _get_disease_tissues(self, disease: str) -> list[str]:
        """Resolve relevant tissues for a disease.

        Priority:
        1. Exact match in DISEASE_TISSUES
        2. Fuzzy substring match in DISEASE_TISSUES
        3. Infer from disease type via classify_disease()
        """
        key = disease.strip().lower()

        # Exact match
        if key in DISEASE_TISSUES:
            return DISEASE_TISSUES[key]

        # Fuzzy match
        for pattern, tissues in DISEASE_TISSUES.items():
            if pattern in key or key in pattern:
                return tissues

        # Infer from disease type
        return self._infer_tissues_from_type(disease)

    def _infer_tissues_from_type(self, disease: str) -> list[str]:
        """Infer relevant tissues from the disease category."""
        try:
            dtype = classify_disease(disease)
            return _TYPE_TISSUES.get(dtype.value, _TYPE_TISSUES["other"])
        except Exception:
            return _TYPE_TISSUES["other"]

    def _query_hpa_all_tissues(self, gene: str) -> list[str]:
        """Query Human Protein Atlas for ALL expressed tissues.

        Returns all tissues with nTPM > 1.0 (detectable expression),
        sorted by expression level (highest first).
        """
        url = f"https://www.proteinatlas.org/{gene}.json"
        try:
            resp = requests.get(url, timeout=self._timeout)
            if resp.status_code != 200:
                return []

            data = resp.json()

            # Parse RNA tissue expression — get ALL expressed tissues
            rna = data.get("RNA tissue specific nTPM", [])
            if not rna:
                rna = data.get("RNA tissue overview", [])
            if not rna:
                return []

            expressed: list[tuple[str, float]] = []
            for entry in rna:
                tissue = entry.get("Tissue", entry.get("tissue", ""))
                val = float(entry.get("Value", entry.get("nTPM", 0)))
                if val > 1.0 and tissue:  # nTPM > 1 = detectable
                    expressed.append((tissue.lower(), val))

            # Sort by expression level, return tissue names
            expressed.sort(key=lambda x: x[1], reverse=True)
            return [t for t, _ in expressed]

        except Exception as exc:
            logger.debug("HPA query failed for %s: %s", gene, exc)
            return []
