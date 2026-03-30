"""
NeoRx CLI
=================

Typer-based command-line interface.

Commands
--------
- ``neorx run <disease>``     — Full pipeline
- ``neorx graph <disease>``   — Build graph only
- ``neorx identify <disease>``— Identify causal targets
- ``neorx report <disease>``  — Generate report from cache
- ``neorx serve``             — Start FastAPI server

Examples::

    $ neorx run HIV --top-n 5
    $ neorx graph "Type 2 Diabetes"
    $ neorx identify HIV --top-n 10
    $ neorx serve --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="neorx",
    help="🧬 NeoRx — Causal Drug Target Discovery Pipeline",
    add_completion=False,
)


def _setup_logging(
    verbose: bool = False,
    log_file: str | None = None,
) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s │ %(name)-25s │ %(levelname)-7s │ %(message)s"
    datefmt = "%H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
    ]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(log_file, encoding="utf-8"),
        )

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )


@app.command()
def run(
    disease: str = typer.Argument(..., help="Disease name (e.g. HIV)"),
    top_n: int = typer.Option(5, "--top-n", "-n", help="Number of top targets"),
    candidates: int = typer.Option(100, "--candidates", "-c", help="Candidates per target"),
    no_docking: bool = typer.Option(False, "--no-docking", help="Skip molecular docking"),
    no_generation: bool = typer.Option(False, "--no-generation", help="Skip molecule generation"),
    no_report: bool = typer.Option(False, "--no-report", help="Skip report generation"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass API response cache"),
    allow_mocks: bool = typer.Option(
        False, "--allow-mocks",
        help="Allow fallback to curated mock data when APIs fail",
    ),
    seed: int | None = typer.Option(None, "--seed", help="Random seed for reproducibility"),
    log_file: str | None = typer.Option(None, "--log-file", help="Write logs to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Run the full NeoRx pipeline."""
    _setup_logging(verbose, log_file)

    if seed is not None:
        os.environ["NEORX_SEED"] = str(seed)

    from .pipeline import run_pipeline

    # ── Rich progress bar (optional) ───────────────────────
    try:
        from rich.progress import (
            Progress, SpinnerColumn, TextColumn,
            BarColumn, TimeElapsedColumn,
        )
        from rich.console import Console
        console = Console()
        console.print(f"\n[bold cyan]🧬 NeoRx Pipeline — {disease}[/]")
        console.print("═" * 50)

        steps = [
            "Building disease graph",
            "Identifying causal targets",
            "Generating molecules",
            "Screening candidates",
            "Scoring & ranking",
            "Generating report",
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(steps[0], total=len(steps))

            # We can't hook into pipeline steps directly,
            # so just advance around the single call.
            for i, step in enumerate(steps):
                progress.update(task, description=step, completed=i)

            result = run_pipeline(
                disease=disease,
                top_n_targets=top_n,
                candidates_per_target=candidates,
                generate_molecules=not no_generation,
                run_docking=not no_docking,
                generate_report=not no_report,
                allow_mocks=allow_mocks,
            )
            progress.update(task, completed=len(steps))

        console.print(f"\n[bold green]✅ Pipeline complete: {result.job.status.value}[/]")
        console.print(f"   Causal targets: {result.n_causal_targets}")
        console.print(f"   Candidates scored: {len(result.scored_candidates)}")
        if result.report_path:
            console.print(f"   Report: {result.report_path}")

    except ImportError:
        # Fallback: no rich
        typer.echo(f"\n🧬 NeoRx Pipeline — {disease}")
        typer.echo("═" * 50)

        result = run_pipeline(
            disease=disease,
            top_n_targets=top_n,
            candidates_per_target=candidates,
            generate_molecules=not no_generation,
            run_docking=not no_docking,
            generate_report=not no_report,
            allow_mocks=allow_mocks,
        )

        typer.echo(f"\n✅ Pipeline complete: {result.job.status.value}")
        typer.echo(f"   Causal targets: {result.n_causal_targets}")
        typer.echo(f"   Candidates scored: {len(result.scored_candidates)}")
        if result.report_path:
            typer.echo(f"   Report: {result.report_path}")

    if result.scored_candidates:
        typer.echo("\n🏆 Top 5 Candidates:")
        for c in result.top_candidates[:5]:
            typer.echo(
                f"   #{c.rank}  {c.composite_score:.4f}  "
                f"{c.smiles[:40]}  ({c.target_protein_name})"
            )


@app.command()
def graph(
    disease: str = typer.Argument(..., help="Disease name"),
    export: str | None = typer.Option(
        None, "--export", "-e",
        help="Export format: json, graphml, gexf, cytoscape",
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass cache"),
    allow_mocks: bool = typer.Option(
        False, "--allow-mocks",
        help="Allow fallback to curated mock data when APIs fail",
    ),
    seed: int | None = typer.Option(None, "--seed", help="Random seed"),
    log_file: str | None = typer.Option(None, "--log-file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build a disease causal knowledge graph."""
    _setup_logging(verbose, log_file)

    if seed is not None:
        os.environ["NEORX_SEED"] = str(seed)

    from .graph_builder import build_disease_graph

    typer.echo(f"\n🧬 Building causal graph for '{disease}'…")
    g = build_disease_graph(disease, use_cache=not no_cache, allow_mocks=allow_mocks)

    typer.echo(f"\n✅ Graph built:")
    typer.echo(f"   Nodes: {len(g.nodes)} ({g.n_genes} genes, {g.n_proteins} proteins, {g.n_pathways} pathways)")
    typer.echo(f"   Edges: {len(g.edges)}")
    typer.echo(f"   Sources: {', '.join(g.sources_queried)}")

    if export:
        from .persistence import save_graph
        path = save_graph(g, fmt=export)
        typer.echo(f"   Exported: {path}")


@app.command()
def identify(
    disease: str = typer.Argument(..., help="Disease name"),
    top_n: int = typer.Option(10, "--top-n", "-n"),
    allow_mocks: bool = typer.Option(
        False, "--allow-mocks",
        help="Allow fallback to curated mock data when APIs fail",
    ),
    seed: int | None = typer.Option(None, "--seed", help="Random seed"),
    log_file: str | None = typer.Option(None, "--log-file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Identify causal drug targets for a disease."""
    _setup_logging(verbose, log_file)

    if seed is not None:
        os.environ["NEORX_SEED"] = str(seed)

    from .graph_builder import build_disease_graph
    from .identifier import identify_causal_targets

    typer.echo(f"\n🧬 Identifying causal targets for '{disease}'…")
    g = build_disease_graph(disease, allow_mocks=allow_mocks)
    targets = identify_causal_targets(g, top_n=top_n)

    n_causal = sum(1 for t in targets if t.is_causal_target)
    typer.echo(f"\n✅ Found {n_causal} causal targets (of {len(targets)} evaluated):\n")

    for t in targets:
        icon = "✓" if t.is_causal_target else "✗"
        ci = t.confidence_interval
        typer.echo(
            f"  {icon} {t.gene_name:12s} "
            f"conf={t.causal_confidence:.3f} "
            f"CI=[{ci[0]:.2f},{ci[1]:.2f}]  "
            f"robust={t.robustness_score:.3f}  "
            f"drug={t.druggability_score:.3f}  "
            f"[{t.classification.value}]"
        )


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
    log_file: str | None = typer.Option(None, "--log-file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Start the NeoRx FastAPI server."""
    _setup_logging(verbose, log_file)

    try:
        import uvicorn
        from .api import app as fastapi_app  # noqa: F811
        typer.echo(f"\n🧬 NeoRx API server starting on {host}:{port}")
        uvicorn.run(fastapi_app, host=host, port=port)
    except ImportError:
        typer.echo("Error: uvicorn not installed. Run: pip install uvicorn", err=True)
        raise typer.Exit(1)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
