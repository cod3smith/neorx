"""
Literature Validation
======================

Validates identified drug targets against published literature
using PubMed E-utilities and — optionally — an LLM for
structured evidence assessment.

Two modes
---------
1. **PubMed-only** (default): searches PubMed for
   ``"<gene> <disease> drug target"`` and counts the number of
   relevant publications.  Genes with zero hits receive a
   ``literature_score`` of 0.0; those with ≥5 hits receive 1.0.

2. **LLM-enhanced** (opt-in): sends the top PubMed abstracts to
   an LLM (Anthropic Claude) for structured assessment of target
   validity, evidence strength, and concerns.  Requires an
   ``ANTHROPIC_API_KEY`` environment variable.

This module is designed to fail gracefully — if PubMed is
unreachable or the API key is missing, a
``LiteratureResult(validated=False)`` is returned.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

_PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@dataclass
class LiteratureResult:
    """Result of literature validation for one target."""
    gene_symbol: str = ""
    disease: str = ""
    validated: bool = False
    n_publications: int = 0
    literature_score: float = 0.0  # 0–1
    top_pmids: list[str] = field(default_factory=list)
    llm_assessment: dict[str, Any] | None = None
    reasoning: str = ""


class LiteratureValidator:
    """Validate drug targets against PubMed literature.

    Parameters
    ----------
    use_llm : bool
        If *True* and ``ANTHROPIC_API_KEY`` is set, use an LLM
        to assess the abstracts.  Default *False*.
    timeout : int
        HTTP timeout for PubMed queries.
    max_abstracts : int
        Maximum number of abstracts to retrieve per gene.
    """

    def __init__(
        self,
        use_llm: bool = False,
        timeout: int = 15,
        max_abstracts: int = 5,
    ) -> None:
        self._use_llm = use_llm
        self._timeout = timeout
        self._max_abstracts = max_abstracts
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    def validate(
        self,
        gene_symbol: str,
        disease: str,
    ) -> LiteratureResult:
        """Validate a single target against PubMed.

        Parameters
        ----------
        gene_symbol : str
            Gene symbol (e.g. "CCR5").
        disease : str
            Disease name (e.g. "HIV").

        Returns
        -------
        LiteratureResult
        """
        gene = gene_symbol.upper().strip()

        # ── Step 1: PubMed search ───────────────────────────
        pmids = self._search_pubmed(gene, disease)
        n_pubs = len(pmids)

        # Score based on publication count
        if n_pubs == 0:
            lit_score = 0.0
        elif n_pubs < 3:
            lit_score = 0.3
        elif n_pubs < 10:
            lit_score = 0.6
        elif n_pubs < 50:
            lit_score = 0.8
        else:
            lit_score = 1.0

        result = LiteratureResult(
            gene_symbol=gene,
            disease=disease,
            validated=n_pubs > 0,
            n_publications=n_pubs,
            literature_score=lit_score,
            top_pmids=pmids[:self._max_abstracts],
        )

        # ── Step 2: Optional LLM assessment ─────────────────
        if self._use_llm and self._api_key and pmids:
            abstracts = self._fetch_abstracts(pmids[:self._max_abstracts])
            if abstracts:
                assessment = self._llm_assess(gene, disease, abstracts)
                result.llm_assessment = assessment

        # ── Reasoning ───────────────────────────────────────
        if n_pubs == 0:
            result.reasoning = (
                f"No PubMed publications found for '{gene} {disease} "
                f"drug target'.  This target lacks literature support."
            )
        else:
            result.reasoning = (
                f"Found {n_pubs} publication(s) for '{gene}' as a "
                f"drug target in {disease} (literature score: "
                f"{lit_score:.1f}).  Top PMIDs: "
                f"{', '.join(pmids[:3])}."
            )

        return result

    def validate_batch(
        self,
        targets: list[str],
        disease: str,
    ) -> list[LiteratureResult]:
        """Validate multiple targets."""
        return [self.validate(gene, disease) for gene in targets]

    # ── PubMed Search ─────────────────────────────────────────

    def _search_pubmed(
        self,
        gene: str,
        disease: str,
    ) -> list[str]:
        """Search PubMed for gene-disease drug target publications.

        Returns a list of PubMed IDs (PMIDs).
        """
        query = f"{gene} {disease} drug target therapeutic"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": 100,
            "retmode": "json",
            "sort": "relevance",
        }

        try:
            resp = requests.get(
                _PUBMED_SEARCH,
                params=params,
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                logger.debug("PubMed search failed: HTTP %d", resp.status_code)
                return []

            data = resp.json()
            result = data.get("esearchresult", {})
            pmids = result.get("idlist", [])
            return pmids

        except Exception as exc:
            logger.debug("PubMed search failed for %s/%s: %s", gene, disease, exc)
            return []

    def _fetch_abstracts(self, pmids: list[str]) -> list[str]:
        """Fetch abstracts for a list of PMIDs."""
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "text",
        }

        try:
            resp = requests.get(
                _PUBMED_FETCH,
                params=params,
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                return []

            # Split by double newlines to get individual abstracts
            text = resp.text.strip()
            abstracts = [a.strip() for a in re.split(r"\n{3,}", text) if a.strip()]
            return abstracts[:self._max_abstracts]

        except Exception as exc:
            logger.debug("PubMed fetch failed: %s", exc)
            return []

    # ── LLM Assessment ────────────────────────────────────────

    def _llm_assess(
        self,
        gene: str,
        disease: str,
        abstracts: list[str],
    ) -> dict[str, Any] | None:
        """Use Anthropic Claude to assess target validity."""
        if not self._api_key:
            return None

        try:
            import anthropic
        except ImportError:
            logger.debug("anthropic package not installed — skipping LLM assessment")
            return None

        abstracts_text = "\n\n---\n\n".join(abstracts[:3])

        prompt = (
            f"You are a medicinal chemist reviewing potential drug targets.\n\n"
            f"Target: {gene}\n"
            f"Disease: {disease}\n\n"
            f"Here are relevant PubMed abstracts:\n{abstracts_text}\n\n"
            f"Questions:\n"
            f"1. Is {gene} a validated drug target for {disease}?\n"
            f"2. Are there existing drugs targeting this protein for this disease?\n"
            f"3. Is this a direct disease mechanism target, or a symptom/biomarker?\n"
            f"4. What is the evidence strength (strong/moderate/weak/none)?\n"
            f"5. Any concerns about targeting this protein?\n\n"
            f"Respond as JSON with keys: is_validated (bool), "
            f"existing_drugs (list of strings), target_type (string), "
            f"evidence_strength (string), concerns (string), "
            f"reasoning (string)."
        )

        try:
            client = anthropic.Anthropic(api_key=self._api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text

            # Extract JSON from response
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            return {"raw_response": text}

        except Exception as exc:
            logger.debug("LLM assessment failed: %s", exc)
            return None
