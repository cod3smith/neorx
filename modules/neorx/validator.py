"""
Known Target Validator
=======================

Quality-control layer that validates NeoRx's output
against established ground-truth drug targets.  For diseases
with well-characterised pharmacology, this checks whether
NeoRx identifies the known targets and correctly
excludes known false positives.

This is NOT used to bias the algorithm — it runs AFTER
identification as a retrospective sanity check.  The
validation results are included in the report so users
can assess pipeline quality.

Metrics
-------
- **Precision**: fraction of identified targets that are
  validated (among the subset with known ground truth).
- **Recall**: fraction of known targets that were identified.
- **F1**: harmonic mean of precision and recall.
- **Quality Grade**: A (F1>0.7), B (>0.4), C (>0.2), F (<0.2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Structured output from ground-truth validation."""
    validated: bool = False
    reason: str = ""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    true_positives: list[str] = field(default_factory=list)
    false_positives_known: list[str] = field(default_factory=list)
    missed_targets: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    quality_grade: str = "N/A"


# ── Ground Truth Database ──────────────────────────────────────────

# Curated from FDA-approved drugs, WHO Essential Medicines List,
# and published drug-target databases (ChEMBL, DrugBank, TTD).

GROUND_TRUTH: dict[str, dict[str, Any]] = {
    "malaria": {
        "known_targets": [
            {"gene": "DHFR", "drug": "pyrimethamine", "organism": "P. falciparum",
             "mechanism": "Dihydrofolate reductase inhibitor — blocks folate synthesis"},
            {"gene": "DHPS", "drug": "sulfadoxine", "organism": "P. falciparum",
             "mechanism": "Dihydropteroate synthase inhibitor — folate pathway"},
            {"gene": "GYPA", "drug": "invasion_blocker", "organism": "H. sapiens",
             "mechanism": "Glycophorin A — receptor for PfEBA-175 during RBC invasion"},
            {"gene": "GYPB", "drug": "invasion_blocker", "organism": "H. sapiens",
             "mechanism": "Glycophorin B — receptor for PfEBL-1 during merozoite invasion"},
            {"gene": "CR1", "drug": "invasion_blocker", "organism": "H. sapiens",
             "mechanism": "Complement receptor 1 — PfRh4 binding for invasion"},
            {"gene": "BSG", "drug": "invasion_blocker", "organism": "H. sapiens",
             "mechanism": "Basigin (CD147) — essential receptor for PfRH5 binding"},
            {"gene": "DARC", "drug": "invasion_blocker", "organism": "H. sapiens",
             "mechanism": "Duffy antigen — receptor for P. vivax DBP"},
        ],
        "known_false_targets": [
            {"gene": "GABRD", "reason": "GABA-A receptor delta — cerebral malaria seizure marker"},
            {"gene": "GABRA1", "reason": "GABA-A receptor alpha-1 — cerebral malaria seizure marker"},
            {"gene": "GABRA2", "reason": "GABA-A receptor alpha-2 — cerebral malaria seizure marker"},
            {"gene": "SCN2A", "reason": "Nav1.2 sodium channel — neuronal, not antimalarial"},
            {"gene": "SCN9A", "reason": "Nav1.7 sodium channel — pain signalling, not antimalarial"},
            {"gene": "SCN10A", "reason": "Nav1.8 sodium channel — pain signalling, not antimalarial"},
            {"gene": "TNF", "reason": "Inflammatory cytokine — consequence of immune activation"},
            {"gene": "IL6", "reason": "Inflammatory cytokine — acute phase response marker"},
        ],
    },
    "hiv": {
        "known_targets": [
            {"gene": "CCR5", "drug": "maraviroc", "organism": "H. sapiens",
             "mechanism": "HIV-1 co-receptor — blocks viral entry"},
            {"gene": "CXCR4", "drug": "AMD3100", "organism": "H. sapiens",
             "mechanism": "HIV-1 co-receptor (X4-tropic strains)"},
            {"gene": "CD4", "drug": "ibalizumab", "organism": "H. sapiens",
             "mechanism": "Primary HIV receptor — blocks gp120 binding"},
        ],
        "known_false_targets": [
            {"gene": "TNF", "reason": "Inflammatory marker — consequence of immune activation"},
            {"gene": "IL6", "reason": "Inflammatory marker — acute phase response"},
            {"gene": "CRP", "reason": "Acute phase protein — non-specific inflammation marker"},
        ],
    },
    "type 2 diabetes": {
        "known_targets": [
            {"gene": "GLP1R", "drug": "semaglutide/liraglutide", "organism": "H. sapiens",
             "mechanism": "GLP-1 receptor agonist — incretin pathway"},
            {"gene": "SLC5A2", "drug": "dapagliflozin/empagliflozin", "organism": "H. sapiens",
             "mechanism": "SGLT2 inhibitor — renal glucose reabsorption"},
            {"gene": "DPP4", "drug": "sitagliptin/saxagliptin", "organism": "H. sapiens",
             "mechanism": "DPP-4 inhibitor — incretin degradation"},
            {"gene": "PPARG", "drug": "pioglitazone", "organism": "H. sapiens",
             "mechanism": "PPARγ agonist — insulin sensitizer"},
            {"gene": "INSR", "drug": "insulin", "organism": "H. sapiens",
             "mechanism": "Insulin receptor — primary glucose regulation"},
            {"gene": "INS", "drug": "insulin", "organism": "H. sapiens",
             "mechanism": "Insulin itself — direct replacement therapy"},
        ],
        "known_false_targets": [
            {"gene": "CRP", "reason": "Inflammatory marker — consequence of metabolic syndrome"},
            {"gene": "TNF", "reason": "Inflammatory cytokine — associated but not causal for T2D"},
        ],
    },
    "lung cancer": {
        "known_targets": [
            {"gene": "EGFR", "drug": "erlotinib/osimertinib", "organism": "H. sapiens",
             "mechanism": "EGFR kinase — driver mutation in NSCLC"},
            {"gene": "ALK", "drug": "crizotinib/alectinib", "organism": "H. sapiens",
             "mechanism": "ALK fusion — driver rearrangement"},
            {"gene": "KRAS", "drug": "sotorasib/adagrasib", "organism": "H. sapiens",
             "mechanism": "KRAS G12C — oncogenic driver mutation"},
            {"gene": "PIK3CA", "drug": "alpelisib", "organism": "H. sapiens",
             "mechanism": "PI3K catalytic subunit — oncogenic pathway"},
            {"gene": "ERBB2", "drug": "trastuzumab", "organism": "H. sapiens",
             "mechanism": "HER2 amplification — growth signalling"},
            {"gene": "TP53", "drug": "research_stage", "organism": "H. sapiens",
             "mechanism": "Tumour suppressor — loss-of-function driver"},
            {"gene": "BRAF", "drug": "dabrafenib", "organism": "H. sapiens",
             "mechanism": "BRAF V600E — MAPK pathway driver"},
        ],
        "known_false_targets": [],
    },
    "breast cancer": {
        "known_targets": [
            {"gene": "ERBB2", "drug": "trastuzumab/pertuzumab", "organism": "H. sapiens",
             "mechanism": "HER2 amplification — growth signalling driver"},
            {"gene": "ESR1", "drug": "tamoxifen/fulvestrant", "organism": "H. sapiens",
             "mechanism": "Estrogen receptor — hormone-driven proliferation"},
            {"gene": "PIK3CA", "drug": "alpelisib", "organism": "H. sapiens",
             "mechanism": "PI3K mutation — oncogenic signalling"},
            {"gene": "BRCA1", "drug": "olaparib", "organism": "H. sapiens",
             "mechanism": "DNA repair deficiency — PARP inhibitor sensitivity"},
            {"gene": "BRCA2", "drug": "olaparib", "organism": "H. sapiens",
             "mechanism": "DNA repair deficiency — PARP inhibitor sensitivity"},
            {"gene": "CDK4", "drug": "palbociclib", "organism": "H. sapiens",
             "mechanism": "Cell cycle kinase — CDK4/6 inhibitor target"},
        ],
        "known_false_targets": [],
    },
    "alzheimer disease": {
        "known_targets": [
            {"gene": "BACE1", "drug": "research_stage", "organism": "H. sapiens",
             "mechanism": "Beta-secretase — amyloid precursor processing"},
            {"gene": "APP", "drug": "aducanumab_target", "organism": "H. sapiens",
             "mechanism": "Amyloid precursor protein — amyloid plaque source"},
            {"gene": "PSEN1", "drug": "research_stage", "organism": "H. sapiens",
             "mechanism": "Presenilin-1 — gamma-secretase component"},
            {"gene": "MAPT", "drug": "research_stage", "organism": "H. sapiens",
             "mechanism": "Tau protein — neurofibrillary tangle formation"},
            {"gene": "ACHE", "drug": "donepezil/rivastigmine", "organism": "H. sapiens",
             "mechanism": "Acetylcholinesterase — cholinergic deficit treatment"},
        ],
        "known_false_targets": [],
    },
    "ebola": {
        "known_targets": [
            {"gene": "NPC1", "drug": "U18666A", "organism": "H. sapiens",
             "mechanism": "Niemann-Pick C1 — required for Ebola viral entry"},
        ],
        "known_false_targets": [
            {"gene": "TNF", "reason": "Cytokine storm marker — consequence of infection"},
            {"gene": "IL6", "reason": "Inflammatory marker — not directly antiviral"},
        ],
    },
}


class KnownTargetValidator:
    """Validate identified targets against established ground truth.

    This validator is a quality-control tool, not a training
    signal.  It runs post-identification to flag potential
    issues and compute quality metrics for the report.
    """

    def validate(
        self,
        disease: str,
        identified_targets: list[dict[str, Any]],
    ) -> ValidationResult:
        """Compare identified targets against ground truth.

        Parameters
        ----------
        disease : str
            Disease name (case-insensitive).
        identified_targets : list[dict]
            Each dict must have ``"gene_symbol"`` and ``"is_causal"``
            keys.

        Returns
        -------
        ValidationResult
        """
        key = disease.strip().lower()
        truth = GROUND_TRUTH.get(key)

        if truth is None:
            # Try fuzzy matching
            for gt_key in GROUND_TRUTH:
                if gt_key in key or key in gt_key:
                    truth = GROUND_TRUTH[gt_key]
                    break

        if truth is None:
            return ValidationResult(
                validated=False,
                reason=f"No ground truth available for '{disease}'",
            )

        # Collect identified causal genes
        identified_causal = {
            t["gene_symbol"].upper()
            for t in identified_targets
            if t.get("is_causal", False)
        }
        all_identified = {
            t["gene_symbol"].upper()
            for t in identified_targets
        }

        true_target_genes = {
            t["gene"].upper() for t in truth["known_targets"]
        }
        false_target_genes = {
            t["gene"].upper() for t in truth.get("known_false_targets", [])
        }

        # Metrics against known targets only
        tp = identified_causal & true_target_genes
        fp_known = identified_causal & false_target_genes
        fn = true_target_genes - all_identified  # Not identified at all

        n_identified = len(identified_causal)
        precision = len(tp) / n_identified if n_identified > 0 else 0.0
        recall = len(tp) / len(true_target_genes) if true_target_genes else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Determine quality grade
        if f1 > 0.7:
            grade = "A"
        elif f1 > 0.4:
            grade = "B"
        elif f1 > 0.2:
            grade = "C"
        else:
            grade = "F"

        # Generate warnings
        warnings: list[str] = []
        if fp_known:
            false_reasons = {
                t["gene"].upper(): t["reason"]
                for t in truth.get("known_false_targets", [])
            }
            for fp_gene in sorted(fp_known):
                reason = false_reasons.get(fp_gene, "known non-target")
                warnings.append(
                    f"FALSE POSITIVE: {fp_gene} is a known non-target "
                    f"({reason})"
                )

        if fn:
            missed_info = {
                t["gene"].upper(): t.get("drug", "")
                for t in truth["known_targets"]
            }
            for m in sorted(fn):
                drug = missed_info.get(m, "")
                drug_note = f" (targeted by {drug})" if drug else ""
                warnings.append(
                    f"MISSED: {m}{drug_note} is an established target "
                    f"not identified"
                )

        if n_identified > 0 and precision < 0.5:
            warnings.append(
                f"LOW PRECISION: Only {precision:.0%} of identified "
                f"causal targets are validated"
            )

        return ValidationResult(
            validated=True,
            precision=round(precision, 3),
            recall=round(recall, 3),
            f1=round(f1, 3),
            true_positives=sorted(tp),
            false_positives_known=sorted(fp_known),
            missed_targets=sorted(fn),
            warnings=warnings,
            quality_grade=grade,
        )

    def get_ground_truth_genes(self, disease: str) -> set[str]:
        """Return known target genes for a disease (for reporting)."""
        key = disease.strip().lower()
        truth = GROUND_TRUTH.get(key)
        if truth is None:
            for gt_key in GROUND_TRUTH:
                if gt_key in key or key in gt_key:
                    truth = GROUND_TRUTH[gt_key]
                    break
        if truth is None:
            return set()
        return {t["gene"].upper() for t in truth["known_targets"]}

    def get_known_false_targets(self, disease: str) -> set[str]:
        """Return known false-positive genes for a disease."""
        key = disease.strip().lower()
        truth = GROUND_TRUTH.get(key)
        if truth is None:
            for gt_key in GROUND_TRUTH:
                if gt_key in key or key in gt_key:
                    truth = GROUND_TRUTH[gt_key]
                    break
        if truth is None:
            return set()
        return {
            t["gene"].upper()
            for t in truth.get("known_false_targets", [])
        }
