"""
ChEMBL Local SQLite Data Source
================================

ChEMBL (https://www.ebi.ac.uk/chembl/) is the world's largest
open-access database of bioactive drug-like small molecules.
It contains ~2.9 M compounds, ~1.9 M assays, and ~24 M activities
with curated clinical development data.

This module queries a **local ChEMBL SQLite dump** (chembl_36.db,
~28 GB) to extract validated drug targets for a disease.  This is
the 8th data source in the NeoRx pipeline and the only one that
can surface *pathogen* targets (e.g. PfDHFR-TS for malaria,
HIV-1 protease for AIDS).

Query chain
-----------
1. ``drug_indication`` — find drugs indicated for the disease
   (LIKE match on MeSH heading and EFO term)
2. ``drug_mechanism`` — find the mechanism of action for each drug
3. ``target_dictionary`` — resolve the molecular target
4. ``target_components`` → ``component_sequences`` — get UniProt
   accessions and organism classification
5. ``component_synonyms`` — resolve gene symbols for human targets

Target classification
---------------------
- **Homo sapiens** targets → ``NodeType.GENE`` nodes that merge
  with existing graph nodes (additional drug evidence)
- **Non-human** targets → ``NodeType.PATHOGEN_GENE`` nodes that
  are new to the graph and represent validated pathogen targets

Drug evidence score
-------------------
Each target receives a score based on:
- 60% — Maximum clinical phase (Phase 4=1.0, 3=0.8, 2=0.6, 1=0.4)
- 20% — Number of distinct drugs targeting it (log-scaled diversity)
- 20% — Mechanism-of-action diversity (multiple MOAs = well-characterised)

ChEMBL is free, no API key.  The SQLite dump is available via
``chembl_downloader`` or from https://ftp.ebi.ac.uk/pub/databases/chembl/.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from ..models import GraphNode, GraphEdge, NodeType, EdgeType

logger = logging.getLogger(__name__)

# ── Default DB paths (checked in order) ────────────────────────────

_DB_SEARCH_PATHS = [
    # 1. Project root
    Path(__file__).resolve().parents[3] / "chembl_36.db",
    # 2. pystow default location
    Path.home() / ".data" / "chembl" / "36" / "chembl_36.db",
    # 3. Environment variable
]

TIMEOUT = 60  # SQLite query timeout (seconds)


def _find_db(db_path: str | None = None) -> Path | None:
    """Locate the ChEMBL SQLite database.

    Search order:
    1. Explicit ``db_path`` argument
    2. ``CHEMBL_DB`` environment variable
    3. Project root ``chembl_36.db``
    4. ``~/.data/chembl/36/chembl_36.db`` (pystow default)
    """
    if db_path:
        p = Path(db_path)
        if p.is_file():
            return p

    env_path = os.environ.get("CHEMBL_DB")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p

    for candidate in _DB_SEARCH_PATHS:
        if candidate.is_file():
            return candidate

    return None


# ── Main query ──────────────────────────────────────────────────────

_MAIN_QUERY = """
SELECT
    td.tid,
    td.pref_name       AS target_name,
    td.chembl_id        AS target_chembl_id,
    td.organism,
    td.target_type,
    cs.accession        AS uniprot_accession,
    cs.component_id,
    dm.mechanism_of_action,
    dm.action_type,
    di.max_phase_for_ind,
    md.pref_name        AS drug_name,
    md.chembl_id        AS drug_chembl_id,
    md.max_phase        AS drug_max_phase
FROM drug_indication di
JOIN molecule_dictionary md ON di.molregno = md.molregno
JOIN drug_mechanism dm     ON di.molregno = dm.molregno
JOIN target_dictionary td  ON dm.tid      = td.tid
LEFT JOIN target_components  tc ON td.tid            = tc.tid
LEFT JOIN component_sequences cs ON tc.component_id  = cs.component_id
WHERE LOWER(di.mesh_heading) LIKE :pattern
   OR LOWER(di.efo_term)     LIKE :pattern
"""

_GENE_SYMBOL_QUERY = """
SELECT component_synonym
FROM component_synonyms
WHERE component_id = :comp_id
  AND syn_type = 'GENE_SYMBOL'
LIMIT 1
"""

_ALL_SYNONYMS_QUERY = """
SELECT syn_type, component_synonym
FROM component_synonyms
WHERE component_id = :comp_id
ORDER BY syn_type
"""


def query_chembl(
    disease_name: str,
    *,
    db_path: str | None = None,
    max_results: int = 100,
    allow_mocks: bool = False,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Query ChEMBL for validated drug targets of a disease.

    Uses the local SQLite dump to find drugs indicated for the
    disease, then traces back through mechanisms to identify the
    molecular targets — both human and pathogen.

    Parameters
    ----------
    disease_name : str
        Disease name (e.g. "malaria", "HIV infection", "breast cancer").
    db_path : str or None
        Explicit path to ``chembl_36.db``.  If *None*, auto-detected.
    max_results : int
        Maximum number of unique targets to return.
    allow_mocks : bool
        Ignored (ChEMBL has no mock fallback — returns empty).

    Returns
    -------
    tuple[list[GraphNode], list[GraphEdge]]
        Discovered nodes and edges.
    """
    _empty: tuple[list[GraphNode], list[GraphEdge]] = ([], [])

    db_file = _find_db(db_path)
    if db_file is None:
        logger.warning(
            "ChEMBL SQLite not found.  Set CHEMBL_DB env var or "
            "place chembl_36.db in the project root."
        )
        return _empty

    try:
        return _query_chembl_db(disease_name, db_file, max_results)
    except Exception as e:
        logger.warning("ChEMBL query failed: %s", e)
        return _empty


def _query_chembl_db(
    disease_name: str,
    db_path: Path,
    max_results: int,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Execute the ChEMBL SQLite query and build graph nodes."""

    pattern = f"%{disease_name.lower()}%"

    conn = sqlite3.connect(str(db_path), timeout=TIMEOUT)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        # ── Step 1: Run main query ──────────────────────────────
        cur.execute(_MAIN_QUERY, {"pattern": pattern})
        rows = cur.fetchall()

        if not rows:
            logger.info("ChEMBL: no drug indications for '%s'.", disease_name)
            return ([], [])

        # ── Step 2: Group by (tid, organism) to deduplicate ─────
        targets: dict[int, dict[str, Any]] = {}

        for row in rows:
            tid = row["tid"]
            if tid not in targets:
                targets[tid] = {
                    "target_name": row["target_name"],
                    "target_chembl_id": row["target_chembl_id"],
                    "organism": row["organism"] or "Unknown",
                    "target_type": row["target_type"] or "",
                    "uniprots": set(),
                    "component_ids": set(),
                    "drugs": set(),
                    "drug_chembl_ids": set(),
                    "mechanisms": set(),
                    "action_types": set(),
                    "max_phase": 0,
                }

            info = targets[tid]
            if row["uniprot_accession"]:
                info["uniprots"].add(row["uniprot_accession"])
            if row["component_id"]:
                info["component_ids"].add(row["component_id"])
            if row["drug_name"]:
                info["drugs"].add(row["drug_name"])
            if row["drug_chembl_id"]:
                info["drug_chembl_ids"].add(row["drug_chembl_id"])
            if row["mechanism_of_action"]:
                info["mechanisms"].add(row["mechanism_of_action"])
            if row["action_type"]:
                info["action_types"].add(row["action_type"])
            phase = row["max_phase_for_ind"] or 0
            if phase > info["max_phase"]:
                info["max_phase"] = phase

        # ── Step 3: Resolve gene symbols for ALL targets ─────
        gene_symbol_cache: dict[int, str] = {}
        for tid, info in targets.items():
            is_human = _is_human(info["organism"])
            for comp_id in info["component_ids"]:
                if comp_id in gene_symbol_cache:
                    continue
                if is_human:
                    # Human targets: GENE_SYMBOL is reliable
                    cur.execute(
                        _GENE_SYMBOL_QUERY, {"comp_id": comp_id}
                    )
                    row = cur.fetchone()
                    if row:
                        gene_symbol_cache[comp_id] = row["component_synonym"]
                else:
                    # Pathogen targets: try GENE_SYMBOL, then
                    # shortest UNIPROT synonym as fallback
                    symbol = _resolve_pathogen_symbol(cur, comp_id)
                    if symbol:
                        gene_symbol_cache[comp_id] = symbol

        logger.info(
            "ChEMBL: %d unique targets for '%s' (%d rows).",
            len(targets), disease_name, len(rows),
        )

    finally:
        conn.close()

    # ── Step 4: Build graph nodes and edges ─────────────────────
    # Use a dict to deduplicate nodes by node_id (different ChEMBL
    # targets can resolve to the same gene symbol, e.g. HIV protease,
    # RT, and integrase all map to POL).
    node_map: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    disease_node_id = f"disease:{disease_name.lower().replace(' ', '_')}"

    # Sort by max_phase descending, then by drug count
    sorted_targets = sorted(
        targets.values(),
        key=lambda t: (t["max_phase"], len(t["drugs"])),
        reverse=True,
    )[:max_results]

    for info in sorted_targets:
        # Skip targets that are not druggable protein targets.
        # NUCLEIC-ACID (e.g. "DNA" for HIV), SUBCELLULAR (e.g.
        # "Cell membrane"), and SMALL MOLECULE (e.g. heme) targets
        # are not actionable for drug target identification.
        if info["target_type"] in (
            "NUCLEIC-ACID", "SUBCELLULAR", "SMALL MOLECULE", "UNKNOWN",
        ):
            continue

        is_human = _is_human(info["organism"])

        # Compute drug evidence score
        drug_score = _compute_drug_evidence_score(info)

        if is_human:
            # Resolve gene symbol — use first available from any component
            gene_symbol = None
            for comp_id in info["component_ids"]:
                gs = gene_symbol_cache.get(comp_id)
                if gs:
                    gene_symbol = gs.upper()
                    break

            if not gene_symbol:
                # Fall back to target name for protein families
                gene_symbol = _target_name_to_symbol(info["target_name"])
                if not gene_symbol:
                    continue

            node_id = f"gene:{gene_symbol}"
            node_type = NodeType.GENE

            # Pick primary UniProt (first one)
            primary_uniprot = next(iter(info["uniprots"]), None)

            node = GraphNode(
                node_id=node_id,
                name=gene_symbol,
                node_type=node_type,
                source="ChEMBL",
                score=drug_score,
                uniprot_id=primary_uniprot,
                description=info["target_name"],
                metadata={
                    "chembl_target_id": info["target_chembl_id"],
                    "chembl_drug_evidence_score": drug_score,
                    "clinical_phase": info["max_phase"],
                    "n_drugs": len(info["drugs"]),
                    "drugs": sorted(info["drugs"]),
                    "mechanisms_of_action": sorted(info["mechanisms"]),
                    "action_types": sorted(info["action_types"]),
                    "has_known_drug": True,
                    "is_druggable": True,
                    "is_pathogen_target": False,
                    "organism": info["organism"],
                },
            )
        else:
            # Non-human target → pathogen gene
            organism_short = _sanitize_organism(info["organism"])

            # Resolve symbol from component_synonyms first
            target_symbol = None
            for comp_id in info["component_ids"]:
                gs = gene_symbol_cache.get(comp_id)
                if gs:
                    target_symbol = gs.upper()
                    break

            if not target_symbol:
                target_symbol = _target_name_to_symbol(info["target_name"])
            if not target_symbol:
                # No usable gene symbol — skip this target entirely.
                # (e.g. "DNA polymerase catalytic subunit" from HSV,
                #  "Reverse transcriptase" from HIV-2 with bad synonyms)
                logger.debug(
                    "ChEMBL: skipping pathogen target '%s' (%s) — "
                    "no usable gene symbol.",
                    info["target_name"], info["organism"],
                )
                continue

            node_id = f"pathogen:{organism_short}:{target_symbol}"
            node_type = NodeType.PATHOGEN_GENE

            primary_uniprot = next(iter(info["uniprots"]), None)

            node = GraphNode(
                node_id=node_id,
                name=target_symbol,
                node_type=node_type,
                source="ChEMBL",
                score=drug_score,
                uniprot_id=primary_uniprot,
                description=f"{info['target_name']} ({info['organism']})",
                metadata={
                    "chembl_target_id": info["target_chembl_id"],
                    "chembl_drug_evidence_score": drug_score,
                    "clinical_phase": info["max_phase"],
                    "n_drugs": len(info["drugs"]),
                    "drugs": sorted(info["drugs"]),
                    "mechanisms_of_action": sorted(info["mechanisms"]),
                    "action_types": sorted(info["action_types"]),
                    "has_known_drug": True,
                    "is_druggable": True,
                    "is_pathogen_target": True,
                    "pathogen_organism": info["organism"],
                    "organism": info["organism"],
                    "all_uniprots": sorted(info["uniprots"]),
                },
            )

        # Deduplicate: merge drug/mechanism info for same node_id
        if node_id in node_map:
            existing = node_map[node_id]
            # Keep highest score
            if node.score > existing.score:
                existing.score = node.score
            # Merge drug lists and metadata
            ex_drugs = set(existing.metadata.get("drugs", []))
            ex_drugs.update(node.metadata.get("drugs", []))
            existing.metadata["drugs"] = sorted(ex_drugs)
            existing.metadata["n_drugs"] = len(ex_drugs)
            ex_moas = set(existing.metadata.get("mechanisms_of_action", []))
            ex_moas.update(node.metadata.get("mechanisms_of_action", []))
            existing.metadata["mechanisms_of_action"] = sorted(ex_moas)
            ex_phase = existing.metadata.get("clinical_phase", 0)
            new_phase = node.metadata.get("clinical_phase", 0)
            if new_phase > ex_phase:
                existing.metadata["clinical_phase"] = new_phase
            # Recalculate drug evidence score with merged info
            existing.metadata["chembl_drug_evidence_score"] = _compute_drug_evidence_score({
                "max_phase": existing.metadata["clinical_phase"],
                "drugs": ex_drugs,
                "mechanisms": ex_moas,
            })
            existing.score = existing.metadata["chembl_drug_evidence_score"]
        else:
            node_map[node_id] = node
            edges.append(GraphEdge(
                source_id=node_id,
                target_id=disease_node_id,
                edge_type=EdgeType.ASSOCIATED_WITH,
                weight=drug_score,
                source_db="ChEMBL",
                evidence=(
                    f"Phase {info['max_phase']} | "
                    f"{len(info['drugs'])} drug(s) | "
                    f"{', '.join(sorted(info['mechanisms']))}"
                ),
            ))

    nodes = list(node_map.values())
    n_human = sum(1 for n in nodes if not n.metadata.get("is_pathogen_target"))
    n_pathogen = sum(1 for n in nodes if n.metadata.get("is_pathogen_target"))
    logger.info(
        "ChEMBL: returning %d nodes (%d human, %d pathogen) for '%s'.",
        len(nodes), n_human, n_pathogen, disease_name,
    )

    return nodes, edges


# ── Scoring ─────────────────────────────────────────────────────────

def _compute_drug_evidence_score(info: dict[str, Any]) -> float:
    """Compute a drug evidence score for a target.

    Weighted composite:
    - 60% — Clinical phase advancement
    - 20% — Drug diversity (more drugs = more validated)
    - 20% — Mechanism diversity (multiple MOAs = well-studied)

    Returns
    -------
    float
        Score between 0.0 and 1.0.
    """
    # Clinical phase (60%)
    phase = info.get("max_phase", 0)
    phase_scores = {4: 1.0, 3: 0.8, 2: 0.6, 1: 0.4, 0: 0.1}
    phase_score = phase_scores.get(phase, 0.1)

    # Drug diversity (20%) — log-scaled, cap at 10 drugs
    n_drugs = len(info.get("drugs", set()))
    drug_diversity = min(1.0, math.log2(n_drugs + 1) / math.log2(11))

    # Mechanism diversity (20%) — cap at 4 mechanisms
    n_mech = len(info.get("mechanisms", set()))
    mech_diversity = min(1.0, n_mech / 4.0)

    score = 0.60 * phase_score + 0.20 * drug_diversity + 0.20 * mech_diversity
    return round(min(1.0, max(0.0, score)), 4)


# ── Helpers ─────────────────────────────────────────────────────────

def _is_human(organism: str | None) -> bool:
    """Check if an organism is Homo sapiens."""
    if not organism:
        return False
    return "homo sapiens" in organism.lower()


def _sanitize_organism(organism: str) -> str:
    """Create a short, filesystem-safe organism identifier.

    Examples:
    - "Plasmodium falciparum K1" → "p_falciparum"
    - "Human immunodeficiency virus 1" → "hiv1"
    - "Mycobacterium tuberculosis" → "m_tuberculosis"
    """
    org = organism.strip().lower()

    # Common abbreviations
    abbreviations = {
        "human immunodeficiency virus 1": "hiv1",
        "human immunodeficiency virus type 1": "hiv1",
        "human immunodeficiency virus 2": "hiv2",
        "human immunodeficiency virus": "hiv1",
        "hiv-1": "hiv1",
        "hepatitis c virus": "hcv",
        "hepacivirus hominis": "hcv",
        "hepatitis b virus": "hbv",
        "herpes simplex virus": "hsv",
        "human cytomegalovirus": "hcmv",
        "homo sapiens": "human",
    }
    for full_name, abbrev in abbreviations.items():
        if full_name in org:
            return abbrev

    # Generic: genus_species
    parts = org.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}_{parts[1]}".replace("(", "").replace(")", "")
    return parts[0] if parts else "unknown"


def _resolve_pathogen_symbol(
    cur: sqlite3.Cursor, comp_id: int,
) -> str | None:
    """Resolve the best gene symbol for a pathogen component.

    Priority:
    1. GENE_SYMBOL  (e.g. "GP", "NP", "pol")
    2. Shortest UNIPROT synonym that looks like an abbreviation
       (e.g. "DHFR-TS" from the set of UNIPROT synonyms)
    3. None (fall back to target name)

    Quality gates:
    - Minimum 2 characters (rejects "P")
    - Must not be a common English word mistakenly entered as
      a gene symbol (rejects "reverse", "transcriptase", "protease")
    """
    # Common English words that appear as bogus GENE_SYMBOL entries
    # in ChEMBL component_synonyms.  True gene symbols (e.g. POL,
    # ENV, GP) are short capitalised abbreviations, not words.
    _BAD_SYMBOLS = frozenset({
        "reverse", "transcriptase", "protease", "integrase",
        "polymerase", "reductase", "synthase", "synthetase",
        "kinase", "transferase", "transporter", "receptor",
        "inhibitor", "activator", "regulator", "protein",
        "antigen", "toxin", "capsid", "envelope",
    })

    cur.execute(_ALL_SYNONYMS_QUERY, {"comp_id": comp_id})
    all_syns = cur.fetchall()
    if not all_syns:
        return None

    gene_symbols = []
    uniprot_syns = []
    for row in all_syns:
        stype = row["syn_type"]
        sval = row["component_synonym"].strip()
        if stype == "GENE_SYMBOL":
            # Quality gate: ≥2 chars and not a common English word
            if len(sval) >= 2 and sval.lower() not in _BAD_SYMBOLS:
                gene_symbols.append(sval)
        elif stype == "UNIPROT":
            uniprot_syns.append(sval)

    # Prefer GENE_SYMBOL
    if gene_symbols:
        # Pick shortest meaningful gene symbol
        gene_symbols.sort(key=len)
        return gene_symbols[0]

    # Fall back to shortest UNIPROT synonym that looks like
    # an abbreviation (≤15 chars, no spaces, contains uppercase)
    abbreviations = [
        s for s in uniprot_syns
        if len(s) <= 15 and " " not in s and any(c.isupper() for c in s)
    ]
    if abbreviations:
        abbreviations.sort(key=len)
        return abbreviations[0]

    return None


def _target_name_to_symbol(target_name: str) -> str | None:
    """Convert a ChEMBL target name to a usable symbol.

    Only returns single-word names that look like gene symbols.
    Multi-word target names (e.g. "DNA polymerase catalytic subunit")
    are rejected — they aren't useful identifiers and the target
    should have been resolved via component_synonyms instead.

    Returns None if the name is too generic or multi-word.
    """
    if not target_name:
        return None

    symbol = target_name.strip()

    # Reject multi-word names — not gene symbols
    if " " in symbol:
        return None

    # Skip overly generic single-word targets
    generic = {"DNA", "RNA", "Protein", "Lipid", "Ergosterol",
               "Tubulin", "Polypeptides", "Ribosome"}
    if symbol in generic:
        return None

    return symbol.upper()


def get_chembl_version(db_path: str | None = None) -> str | None:
    """Return the ChEMBL version from the database, or None."""
    db_file = _find_db(db_path)
    if db_file is None:
        return None

    try:
        conn = sqlite3.connect(str(db_file), timeout=10)
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='version'"
        )
        if cur.fetchone():
            cur.execute("SELECT name, creation_date FROM version LIMIT 1")
            row = cur.fetchone()
            conn.close()
            if row:
                return f"{row[0]} ({row[1]})"
        conn.close()
    except Exception:
        pass
    return None
