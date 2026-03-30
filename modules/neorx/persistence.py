"""
Graph Persistence & Export
===========================

Save and load ``DiseaseGraph`` objects for reproducibility
and external analysis.

Formats
-------
- **JSON** (native): Full-fidelity serialisation of DiseaseGraph
- **GraphML**: For Cytoscape, yEd, and NetworkX interop
- **GEXF**: For Gephi
- **Cytoscape JSON** (.cyjs): For Cytoscape web/desktop

Database
--------
When PostgreSQL is available (Docker deployment), graphs are
also persisted to the ``disease_graphs`` table for querying
and sharing.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from .models import DiseaseGraph

logger = logging.getLogger(__name__)

GRAPH_DIR = Path.home() / ".neorx" / "graphs"


# ── File-Based Persistence ──────────────────────────────────────────

def save_graph(
    graph: DiseaseGraph,
    output_dir: Path | str | None = None,
    fmt: str = "json",
) -> str:
    """Save a DiseaseGraph to disk.

    Parameters
    ----------
    graph : DiseaseGraph
        The graph to save.
    output_dir : Path, optional
        Defaults to ``~/.neorx/graphs/``.
    fmt : str
        ``json``, ``graphml``, ``gexf``, or ``cytoscape``.

    Returns
    -------
    str
        Path to the saved file.
    """
    if output_dir is None:
        output_dir = GRAPH_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = graph.disease_name.lower().replace(" ", "_").replace("/", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if fmt == "json":
        path = output_dir / f"{safe_name}_{ts}.json"
        data = graph.model_dump(mode="json")
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    elif fmt == "graphml":
        path = output_dir / f"{safe_name}_{ts}.graphml"
        G = _to_export_networkx(graph)
        nx.write_graphml(G, str(path))

    elif fmt == "gexf":
        path = output_dir / f"{safe_name}_{ts}.gexf"
        G = _to_export_networkx(graph)
        nx.write_gexf(G, str(path))

    elif fmt == "cytoscape":
        path = output_dir / f"{safe_name}_{ts}.cyjs"
        G = _to_export_networkx(graph)
        cy_data = nx.cytoscape_data(G)
        path.write_text(json.dumps(cy_data, indent=2), encoding="utf-8")

    else:
        raise ValueError(f"Unknown format: {fmt!r}. Use json/graphml/gexf/cytoscape.")

    logger.info("Graph saved to %s (%s format).", path, fmt)
    return str(path)


def load_graph(path: str | Path) -> DiseaseGraph:
    """Load a DiseaseGraph from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to a ``.json`` file saved by :func:`save_graph`.

    Returns
    -------
    DiseaseGraph
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    graph = DiseaseGraph.model_validate(data)

    logger.info(
        "Graph loaded from %s (%d nodes, %d edges).",
        path, len(graph.nodes), len(graph.edges),
    )
    return graph


def list_saved_graphs(directory: Path | str | None = None) -> list[dict[str, Any]]:
    """List saved graph files with metadata.

    Returns
    -------
    list[dict]
        Each dict has ``path``, ``disease``, ``n_nodes``, ``n_edges``,
        ``created``, ``size_kb``.
    """
    d = Path(directory) if directory else GRAPH_DIR
    if not d.exists():
        return []

    results: list[dict[str, Any]] = []
    for f in sorted(d.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append({
                "path": str(f),
                "disease": data.get("disease_name", "unknown"),
                "n_nodes": len(data.get("nodes", [])),
                "n_edges": len(data.get("edges", [])),
                "created": f.stat().st_mtime,
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return results


# ── Database Persistence ────────────────────────────────────────────

def save_graph_to_db(
    graph: DiseaseGraph,
    params: dict[str, Any] | None = None,
) -> int | None:
    """Persist a graph to PostgreSQL (when available).

    Returns
    -------
    int or None
        Database row ID, or ``None`` if DB is unavailable.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return None

    try:
        import psycopg2  # type: ignore[import-untyped]

        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        graph_json = graph.model_dump(mode="json")

        cur.execute(
            """
            INSERT INTO disease_graphs
                (disease_name, disease_id, graph_json, parameters,
                 n_nodes, n_edges, sources_queried)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (disease_name, parameters)
            DO UPDATE SET
                graph_json = EXCLUDED.graph_json,
                n_nodes    = EXCLUDED.n_nodes,
                n_edges    = EXCLUDED.n_edges,
                created_at = NOW()
            RETURNING id
            """,
            (
                graph.disease_name,
                graph.disease_id,
                json.dumps(graph_json, default=str),
                json.dumps(params or {}),
                len(graph.nodes),
                len(graph.edges),
                graph.sources_queried,
            ),
        )
        row_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        logger.info("Graph persisted to DB (id=%d).", row_id)
        return row_id

    except Exception as e:
        logger.debug("DB persistence skipped: %s", e)
        return None


def load_graph_from_db(disease: str) -> DiseaseGraph | None:
    """Load the most recent graph for a disease from PostgreSQL.

    Returns
    -------
    DiseaseGraph or None
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return None

    try:
        import psycopg2  # type: ignore[import-untyped]

        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        cur.execute(
            """
            SELECT graph_json FROM disease_graphs
            WHERE LOWER(disease_name) = LOWER(%s)
            ORDER BY created_at DESC LIMIT 1
            """,
            (disease,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        if row:
            graph = DiseaseGraph.model_validate(row[0])
            logger.info("Graph loaded from DB for '%s'.", disease)
            return graph
        return None

    except Exception as e:
        logger.debug("DB load skipped: %s", e)
        return None


def save_job_to_db(job_data: dict[str, Any]) -> bool:
    """Persist a pipeline job result to PostgreSQL.

    Parameters
    ----------
    job_data : dict
        Must contain ``job_id``, ``disease``, ``status``.

    Returns
    -------
    bool
        ``True`` if persisted, ``False`` otherwise.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return False

    try:
        import psycopg2  # type: ignore[import-untyped]

        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO pipeline_jobs
                (job_id, disease, status, top_n_targets,
                 candidates_per_target, result_json, report_path,
                 completed_at, error)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (job_id) DO UPDATE SET
                status       = EXCLUDED.status,
                result_json  = EXCLUDED.result_json,
                completed_at = EXCLUDED.completed_at,
                error        = EXCLUDED.error
            """,
            (
                job_data.get("job_id"),
                job_data.get("disease"),
                job_data.get("status", "pending"),
                job_data.get("top_n_targets", 5),
                job_data.get("candidates_per_target", 100),
                json.dumps(job_data.get("result_json"), default=str)
                if job_data.get("result_json") else None,
                job_data.get("report_path"),
                job_data.get("completed_at"),
                job_data.get("error"),
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
        return True

    except Exception as e:
        logger.debug("Job DB persistence skipped: %s", e)
        return False


# ── Internal Helpers ────────────────────────────────────────────────

def _to_export_networkx(graph: DiseaseGraph) -> nx.DiGraph:
    """Convert DiseaseGraph to NetworkX DiGraph for file export.

    Serialises all attributes as strings for GraphML/GEXF
    compatibility.
    """
    G = nx.DiGraph()

    for node in graph.nodes:
        G.add_node(
            node.node_id,
            label=node.name,
            node_type=node.node_type.value,
            score=str(node.score),
            uniprot_id=node.uniprot_id or "",
            description=node.description or "",
            pdb_ids=",".join(node.pdb_ids) if node.pdb_ids else "",
        )

    for edge in graph.edges:
        G.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type.value,
            weight=str(edge.weight),
            source_db=edge.source_db,
            evidence=edge.evidence or "",
            pmids=",".join(edge.pmids) if edge.pmids else "",
        )

    return G
