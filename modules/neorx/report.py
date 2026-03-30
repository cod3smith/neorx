"""
NeoRx Report Generator
===============================

Generates a self-contained HTML report summarising the pipeline
output.  The report includes:

1. **Executive Summary** — disease, targets found, top candidates
2. **Causal Graph Visualisation** — interactive node-link diagram
3. **Target Deep-Dive** — per-target causal reasoning
4. **Candidate Table** — ranked candidates with score breakdown
5. **Methodology** — explanation of causal reasoning approach

The report uses Jinja2 templating with inline CSS (NeoForge
brand: Deep Navy #0D1B2A, Neon Teal #00D4AA).  No external
dependencies are required to view it.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import (
    DiseaseGraph,
    NeoRxResult,
    ScoredCandidate,
    NodeType,
)

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


def generate_report(
    disease: str,
    graph: DiseaseGraph,
    causal_targets: list[NeoRxResult],
    candidates: list[ScoredCandidate],
    output_dir: str | Path | None = None,
    validation: dict | None = None,
) -> tuple[str, str]:
    """Generate an HTML report for a pipeline run.

    Parameters
    ----------
    disease : str
        Disease name.
    graph : DiseaseGraph
        The assembled causal knowledge graph.
    causal_targets : list[NeoRxResult]
        Identified causal targets.
    candidates : list[ScoredCandidate]
        Scored and ranked candidates.
    output_dir : str or Path, optional
        Directory to save the report.  Defaults to ``./reports/``.

    Returns
    -------
    tuple[str, str]
        (report_html, report_path)
    """
    if output_dir is None:
        output_dir = Path("reports")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build context for template
    context = _build_context(disease, graph, causal_targets, candidates, validation)

    # Render template
    html = _render_template(context)

    # Save to file
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_disease = disease.lower().replace(" ", "_").replace("/", "_")
    filename = f"neorx_{safe_disease}_{timestamp}.html"
    filepath = output_dir / filename
    filepath.write_text(html, encoding="utf-8")

    logger.info("Report saved to %s.", filepath)
    return html, str(filepath)


def _build_context(
    disease: str,
    graph: DiseaseGraph,
    causal_targets: list[NeoRxResult],
    candidates: list[ScoredCandidate],
    validation: dict | None = None,
) -> dict[str, Any]:
    """Build the template context dictionary."""
    n_causal = sum(1 for t in causal_targets if t.is_causal_target)
    n_correlational = sum(
        1 for t in causal_targets if not t.is_causal_target
    )
    top_candidates = candidates[:20]

    # Build graph data for visualisation
    graph_nodes = []
    for node in graph.nodes[:100]:  # Limit for performance
        graph_nodes.append({
            "id": node.node_id,
            "label": node.name,
            "type": node.node_type.value,
            "score": node.score,
        })

    graph_edges = []
    for edge in graph.edges[:200]:
        graph_edges.append({
            "from": edge.source_id,
            "to": edge.target_id,
            "type": edge.edge_type.value,
            "weight": edge.weight,
        })

    # Collect PMIDs from graph edges for evidence tracking
    edge_pmids: dict[str, list[str]] = {}
    for edge in graph.edges:
        key = f"{edge.source_id}→{edge.target_id}"
        pmids = edge.pmids if edge.pmids else []
        if pmids:
            edge_pmids[key] = pmids[:5]  # Cap at 5 per edge

    return {
        "disease": disease,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "graph": {
            "n_nodes": len(graph.nodes),
            "n_edges": len(graph.edges),
            "n_genes": graph.n_genes,
            "n_proteins": graph.n_proteins,
            "n_pathways": graph.n_pathways,
            "sources": graph.sources_queried,
            "nodes_json": json.dumps(graph_nodes),
            "edges_json": json.dumps(graph_edges),
            "edge_pmids": edge_pmids,
        },
        "targets": {
            "total": len(causal_targets),
            "n_causal": n_causal,
            "n_correlational": n_correlational,
            "entries": [
                {
                    "gene_name": t.gene_name,
                    "protein_name": t.protein_name,
                    "classification": t.classification.value,
                    "is_causal": t.is_causal_target,
                    "causal_confidence": t.causal_confidence,
                    "causal_effect": t.causal_effect,
                    "robustness": t.robustness_score,
                    "druggability": t.druggability_score,
                    "reasoning": t.reasoning,
                    "n_pathways": t.n_supporting_pathways,
                    "n_interactions": t.n_protein_interactions,
                    "pdb_ids": t.pdb_ids[:3],
                    "confidence_interval": t.confidence_interval,
                    "target_type": getattr(t, "target_type", ""),
                    "tissue_relevant": getattr(t, "tissue_relevant", True),
                    "tissue_explanation": getattr(t, "tissue_explanation", ""),
                    "evidence_streams": getattr(t, "evidence_streams", 0),
                }
                for t in causal_targets
            ],
        },
        "candidates": {
            "total": len(candidates),
            "n_drug_like": sum(1 for c in candidates if c.is_drug_like),
            "n_novel": sum(1 for c in candidates if c.is_novel),
            "entries": [
                {
                    "rank": c.rank,
                    "smiles": c.smiles,
                    "target": c.target_protein_name,
                    "composite": c.composite_score,
                    "causal_conf": c.causal_confidence,
                    "binding": c.binding_affinity,
                    "qed": c.qed_score,
                    "sa": c.sa_score,
                    "admet": c.admet_score,
                    "novelty": c.novelty_score,
                    "mw": c.molecular_weight,
                    "is_drug_like": c.is_drug_like,
                    "is_novel": c.is_novel,
                }
                for c in top_candidates
            ],
        },
        "validation": validation,
    }


def _render_template(context: dict[str, Any]) -> str:
    """Render the HTML report template.

    Uses Jinja2 if available, otherwise falls back to a simple
    string-based template.
    """
    try:
        from jinja2 import Environment, FileSystemLoader
        template_path = TEMPLATE_DIR / "report.html"
        if template_path.exists():
            env = Environment(
                loader=FileSystemLoader(str(TEMPLATE_DIR)),
                autoescape=True,
            )
            template = env.get_template("report.html")
            return template.render(**context)
    except ImportError:
        logger.info("Jinja2 not installed. Using built-in template.")
    except Exception as e:
        logger.warning("Jinja2 rendering failed: %s. Using built-in.", e)

    # Fallback: built-in template
    return _builtin_template(context)


def _builtin_template(ctx: dict[str, Any]) -> str:
    """A self-contained HTML report without Jinja2."""
    targets_html = ""
    for t in ctx["targets"]["entries"]:
        badge = "causal" if t["is_causal"] else t["classification"]
        badge_color = "#00D4AA" if t["is_causal"] else "#ff6b6b"

        # Target type badge
        tt = t.get("target_type", "")
        type_badges = {
            "PATHOGEN_DIRECT": ("🟢", "#00D4AA", "Direct Target"),
            "HOST_INVASION": ("🟢", "#1B98E0", "Host Invasion"),
            "HOST_IMMUNE": ("🟡", "#FFD93D", "Immune"),
            "HOST_SYMPTOM": ("🔴", "#ff6b6b", "Symptom Marker"),
            "CORRELATIONAL": ("⚪", "#999", "Correlational"),
        }
        tt_emoji, _, tt_label = type_badges.get(tt, ("⚪", "#999", tt or "—"))

        tissue_icon = "✓" if t.get("tissue_relevant", True) else "✗"
        tissue_title = t.get("tissue_explanation", "")

        ci = t.get("confidence_interval", (0.0, 1.0))
        ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]"
        targets_html += f"""
        <tr>
            <td><strong>{t["gene_name"]}</strong></td>
            <td>{t["protein_name"]}</td>
            <td><span style="background:{badge_color};color:#fff;
                padding:2px 8px;border-radius:4px;font-size:0.85em">
                {badge}</span></td>
            <td>{tt_emoji} {tt_label}</td>
            <td>{t["causal_confidence"]:.3f}</td>
            <td style="font-size:0.85em">{ci_str}</td>
            <td>{t["robustness"]:.3f}</td>
            <td title="{tissue_title}">{tissue_icon}</td>
            <td>{t.get("evidence_streams", 0)}</td>
        </tr>"""

    candidates_html = ""
    for c in ctx["candidates"]["entries"]:
        candidates_html += f"""
        <tr>
            <td>{c["rank"]}</td>
            <td style="font-family:monospace;font-size:0.85em">{c["smiles"][:40]}</td>
            <td>{c["target"]}</td>
            <td><strong>{c["composite"]:.4f}</strong></td>
            <td>{c["causal_conf"]:.3f}</td>
            <td>{c["binding"] if c["binding"] else "N/A"}</td>
            <td>{c["qed"]:.2f if c["qed"] else "N/A"}</td>
            <td>{"✓" if c["is_drug_like"] else "✗"}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NeoRx Report — {ctx["disease"]}</title>
<style>
  :root {{ --navy: #0D1B2A; --teal: #00D4AA; --bg: #f8f9fa; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif;
         background: var(--bg); color: #333; }}
  .header {{ background: var(--navy); color: #fff; padding: 2rem;
             text-align: center; }}
  .header h1 {{ color: var(--teal); font-size: 2rem; }}
  .header p {{ opacity: 0.8; margin-top: 0.5rem; }}
  .container {{ max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }}
  .card {{ background: #fff; border-radius: 12px;
           box-shadow: 0 2px 12px rgba(0,0,0,0.08);
           padding: 1.5rem; margin-bottom: 1.5rem; }}
  .card h2 {{ color: var(--navy); margin-bottom: 1rem;
              border-bottom: 2px solid var(--teal);
              padding-bottom: 0.5rem; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem; margin-bottom: 1rem; }}
  .stat {{ background: var(--navy); color: #fff; padding: 1rem;
           border-radius: 8px; text-align: center; }}
  .stat .value {{ font-size: 2rem; color: var(--teal); font-weight: bold; }}
  .stat .label {{ font-size: 0.85rem; opacity: 0.8; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th {{ background: var(--navy); color: #fff; padding: 0.75rem;
       text-align: left; }}
  td {{ padding: 0.75rem; border-bottom: 1px solid #e9ecef; }}
  tr:hover {{ background: #f1f3f5; }}
  .footer {{ text-align: center; padding: 2rem; color: #999;
             font-size: 0.85rem; }}
</style>
</head>
<body>
<div class="header">
  <h1>🧬 NeoRx Report</h1>
  <p>{ctx["disease"]} — {ctx["timestamp"]}</p>
</div>
<div class="container">
  <div class="card">
    <h2>Executive Summary</h2>
    <div class="stats">
      <div class="stat">
        <div class="value">{ctx["graph"]["n_nodes"]}</div>
        <div class="label">Graph Nodes</div>
      </div>
      <div class="stat">
        <div class="value">{ctx["graph"]["n_edges"]}</div>
        <div class="label">Graph Edges</div>
      </div>
      <div class="stat">
        <div class="value">{ctx["targets"]["n_causal"]}</div>
        <div class="label">Causal Targets</div>
      </div>
      <div class="stat">
        <div class="value">{ctx["targets"]["n_correlational"]}</div>
        <div class="label">Correlational</div>
      </div>
      <div class="stat">
        <div class="value">{ctx["candidates"]["total"]}</div>
        <div class="label">Candidates Scored</div>
      </div>
      <div class="stat">
        <div class="value">{ctx["candidates"]["n_drug_like"]}</div>
        <div class="label">Drug-Like</div>
      </div>
    </div>
    <p>Data sources: {", ".join(ctx["graph"]["sources"])}</p>
  </div>

  <div class="card">
    <h2>Causal Knowledge Graph</h2>
    <div id="network-graph" style="width:100%;height:450px;border:1px solid #e9ecef;border-radius:8px;"></div>
    <p style="margin-top:0.5rem;font-size:0.85em;color:#666;">Interactive graph — drag to pan, scroll to zoom, click nodes for details.</p>
  </div>

  <div class="card">
    <h2>Causal Target Analysis</h2>
    <table>
      <thead>
        <tr><th>Gene</th><th>Protein</th><th>Classification</th>
            <th>Target Type</th><th>Causal Conf.</th><th>95% CI</th>
            <th>Robustness</th><th>Tissue</th><th>Evidence</th></tr>
      </thead>
      <tbody>{targets_html}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Top Drug Candidates</h2>
    <table>
      <thead>
        <tr><th>#</th><th>SMILES</th><th>Target</th>
            <th>Composite</th><th>Causal</th><th>Binding</th>
            <th>QED</th><th>Drug-like</th></tr>
      </thead>
      <tbody>{candidates_html}</tbody>
    </table>
  </div>
{"" if not ctx.get("validation") else f"""
  <div class="card">
    <h2>Known Target Validation</h2>
    <div class="stats">
      <div class="stat">
        <div class="value">{ctx["validation"]["quality_grade"]}</div>
        <div class="label">Quality Grade</div>
      </div>
      <div class="stat">
        <div class="value">{ctx["validation"]["precision"]:.2f}</div>
        <div class="label">Precision</div>
      </div>
      <div class="stat">
        <div class="value">{ctx["validation"]["recall"]:.2f}</div>
        <div class="label">Recall</div>
      </div>
      <div class="stat">
        <div class="value">{ctx["validation"]["f1"]:.2f}</div>
        <div class="label">F1 Score</div>
      </div>
    </div>
    {"".join(f'<p style="color:#ff6b6b">⚠ {w}</p>' for w in ctx["validation"].get("warnings", []))}
  </div>
"""}
  <div class="card">
    <h2>Methodology</h2>
    <p>NeoRx applies Pearl's do-calculus to distinguish genuine
    causal drug targets from correlational bystanders. The pipeline:</p>
    <ol style="margin:1rem 0 0 1.5rem">
      <li>Builds a causal knowledge graph from 7 biomedical databases</li>
      <li>Tests identifiability via the backdoor criterion (d-separation)</li>
      <li>Estimates causal effects via multi-source evidence triangulation</li>
      <li>Validates robustness through leave-one-source-out sensitivity analysis</li>
      <li>Computes bootstrap 95% confidence intervals (200 resamples)</li>
      <li>Generates novel molecules with GenMol (VAE)</li>
      <li>Screens for drug-likeness (MolScreen) + multi-rule ADMET + binding (DockBot)</li>
      <li>Ranks by composite score with causal confidence weighted highest (0.30)</li>
    </ol>
  </div>
</div>
<div class="footer">
  Generated by NeoRx &middot; NeoForge Bio-AI Platform
</div>
<script src="https://unpkg.com/vis-network@9/standalone/umd/vis-network.min.js"></script>
<script>
(function() {{
  var nodesData = {json.dumps(ctx['graph']['nodes_json']) if isinstance(ctx['graph']['nodes_json'], list) else ctx['graph']['nodes_json']};
  var edgesData = {json.dumps(ctx['graph']['edges_json']) if isinstance(ctx['graph']['edges_json'], list) else ctx['graph']['edges_json']};
  var colorMap = {{gene:'#00D4AA',protein:'#1B98E0',pathway:'#FF6B6B',disease:'#FFD93D',drug:'#A78BFA'}};
  var nodes = nodesData.map(function(n) {{
    return {{id:n.id, label:n.label, color:colorMap[n.type]||'#888',
             shape:'dot', size:8+Math.round((n.score||0)*12),
             title:n.type+' (score: '+(n.score||0).toFixed(2)+')'}};
  }});
  var edges = edgesData.map(function(e) {{
    return {{from:e.from, to:e.to, arrows:'to',
             title:e.type+' (w: '+(e.weight||0).toFixed(2)+')',
             width:1+Math.round((e.weight||0)*3),
             color:{{color:'#0D1B2A', opacity:0.5}}}};
  }});
  var container = document.getElementById('network-graph');
  if (container && nodes.length) {{
    new vis.Network(container,
      {{nodes:new vis.DataSet(nodes), edges:new vis.DataSet(edges)}},
      {{physics:{{solver:'forceAtlas2Based',stabilization:{{iterations:80}}}},
        interaction:{{hover:true,tooltipDelay:100}}}});
  }} else if (container) {{
    container.innerHTML = '<p style="padding:2rem;text-align:center;color:#999">No graph data available.</p>';
  }}
}})()
</script>
</body>
</html>"""
