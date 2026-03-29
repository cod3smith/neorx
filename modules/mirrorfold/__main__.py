"""
MirrorFold CLI
===============

Command-line interface for the MirrorFold module.

Commands
--------
* ``mirrorfold predict``  — Predict structure for a protein sequence
* ``mirrorfold compare``  — Compare L- and D-protein structures
* ``mirrorfold assess``   — Full therapeutic D-protein assessment
* ``mirrorfold info``     — Show amino acid chirality info

Examples
--------
.. code-block:: bash

    # Predict L-protein structure
    mirrorfold predict --sequence MKFLILF --chirality L

    # Compare L and D structures
    mirrorfold compare --sequence YGGFL

    # Full therapeutic assessment
    mirrorfold assess --sequence FVNQHLCGSHLVEALYLVCGERG \\
                      --name "Insulin B1-22" --allow-mock

    # Show mirror amino acid info
    mirrorfold info --residue A
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="mirrorfold",
    help=(
        "🪞 MirrorFold — Predict and compare structures of natural "
        "(L-amino acid) proteins versus their mirror-image "
        "(D-amino acid) counterparts."
    ),
    no_args_is_help=True,
)

logger = logging.getLogger("mirrorfold")


def _setup_logging(verbose: bool) -> None:
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(name)s | %(message)s",
    )


# ── Predict Command ────────────────────────────────────────────────

@app.command()
def predict(
    sequence: str = typer.Option(..., "--sequence", "-s", help="Amino acid sequence"),
    chirality: str = typer.Option("L", "--chirality", "-c", help="L or D"),
    allow_mock: bool = typer.Option(False, "--allow-mock", help="Allow mock PDB if ESMFold unavailable"),
    allow_api: bool = typer.Option(True, "--allow-api", help="Allow ESMFold API fallback"),
    output: str = typer.Option(None, "--output", "-o", help="Save PDB to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Predict the 3D structure of a protein sequence."""
    _setup_logging(verbose)

    from .mirror import is_valid_sequence
    from .predictor import predict_structure

    if not is_valid_sequence(sequence):
        typer.echo("❌ Invalid amino acid sequence.", err=True)
        raise typer.Exit(1)

    chirality = chirality.upper()
    if chirality not in ("L", "D"):
        typer.echo("❌ Chirality must be 'L' or 'D'.", err=True)
        raise typer.Exit(1)

    typer.echo(f"🔬 Predicting {chirality}-protein structure for {len(sequence)} residues...")

    prediction = predict_structure(
        sequence=sequence,
        chirality=chirality,
        allow_api_fallback=allow_api,
        allow_mock=allow_mock,
    )

    typer.echo(f"✅ Prediction complete!")
    typer.echo(f"   Method:     {prediction.prediction_method}")
    typer.echo(f"   Mean pLDDT: {prediction.mean_plddt:.1f}")
    typer.echo(f"   Residues:   {len(prediction.sequence)}")

    if output:
        Path(output).write_text(prediction.pdb_string)
        typer.echo(f"   PDB saved:  {output}")


# ── Compare Command ────────────────────────────────────────────────

@app.command()
def compare(
    sequence: str = typer.Option(..., "--sequence", "-s", help="Amino acid sequence"),
    threshold: float = typer.Option(2.0, "--threshold", "-t", help="Distance threshold (Å)"),
    allow_mock: bool = typer.Option(False, "--allow-mock", help="Allow mock PDB"),
    allow_api: bool = typer.Option(True, "--allow-api", help="Allow API fallback"),
    output_html: str = typer.Option(None, "--html", help="Save HTML comparison report"),
    output_json: str = typer.Option(None, "--json", "-j", help="Save JSON results"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Compare L- and D-protein structures for a given sequence."""
    _setup_logging(verbose)

    from .mirror import is_valid_sequence
    from .predictor import predict_pair
    from .compare import compare_structures
    from .viz import generate_comparison_html

    if not is_valid_sequence(sequence):
        typer.echo("❌ Invalid amino acid sequence.", err=True)
        raise typer.Exit(1)

    typer.echo(f"🪞 Comparing L vs D structures for {len(sequence)}-residue protein...")

    # Predict both forms
    typer.echo("   Predicting L-protein...")
    l_pred, d_pred = predict_pair(
        sequence=sequence,
        allow_api_fallback=allow_api,
        allow_mock=allow_mock,
    )

    typer.echo("   Aligning and comparing structures...")
    report = compare_structures(l_pred, d_pred, distance_threshold=threshold)

    typer.echo(f"\n📊 Comparison Results:")
    typer.echo(f"   RMSD:              {report.rmsd:.2f} Å")
    typer.echo(f"   TM-score:          {report.tm_score:.3f}")
    typer.echo(f"   SS match:          {(report.secondary_structure_match or 0) * 100:.0f}%")
    typer.echo(f"   Conserved regions: {len(report.conserved_regions)}")
    typer.echo(f"   Divergent regions: {len(report.divergent_regions)}")
    typer.echo(f"   Aligned length:    {report.aligned_length}")

    if report.tm_score and report.tm_score > 0.5:
        typer.echo(f"\n   ✅ Same fold predicted (TM > 0.5)")
    else:
        typer.echo(f"\n   ⚠️  Fold may differ (TM ≤ 0.5)")

    if output_html:
        generate_comparison_html(report, output_path=output_html)
        typer.echo(f"\n   HTML report: {output_html}")

    if output_json:
        data = {
            "rmsd": report.rmsd,
            "tm_score": report.tm_score,
            "secondary_structure_match": report.secondary_structure_match,
            "conserved_regions": report.conserved_regions,
            "divergent_regions": report.divergent_regions,
            "aligned_length": report.aligned_length,
            "l_plddt": l_pred.mean_plddt,
            "d_plddt": d_pred.mean_plddt,
        }
        Path(output_json).write_text(json.dumps(data, indent=2))
        typer.echo(f"   JSON results: {output_json}")


# ── Assess Command ─────────────────────────────────────────────────

@app.command()
def assess(
    sequence: str = typer.Option(..., "--sequence", "-s", help="Amino acid sequence"),
    name: str = typer.Option("Unnamed protein", "--name", "-n", help="Protein name"),
    pdb_id: str = typer.Option(None, "--pdb-id", help="PDB identifier"),
    allow_mock: bool = typer.Option(False, "--allow-mock", help="Allow mock PDB"),
    allow_api: bool = typer.Option(True, "--allow-api", help="Allow API fallback"),
    output: str = typer.Option(None, "--output", "-o", help="Save JSON assessment"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a full D-protein therapeutic assessment."""
    _setup_logging(verbose)

    from .mirror import is_valid_sequence
    from .predictor import predict_pair
    from .compare import compare_structures
    from .analysis import compute_property_profile
    from .therapeutic import assess_therapeutic_potential

    if not is_valid_sequence(sequence):
        typer.echo("❌ Invalid amino acid sequence.", err=True)
        raise typer.Exit(1)

    typer.echo(f"🧬 Therapeutic assessment for: {name}")
    typer.echo(f"   Sequence length: {len(sequence)} residues\n")

    # Step 1: Predict structures
    typer.echo("   Step 1/4: Predicting L and D structures...")
    l_pred, d_pred = predict_pair(
        sequence=sequence,
        allow_api_fallback=allow_api,
        allow_mock=allow_mock,
    )

    # Step 2: Compare structures
    typer.echo("   Step 2/4: Comparing structures...")
    comparison = compare_structures(l_pred, d_pred)

    # Step 3: Analyse properties
    typer.echo("   Step 3/4: Analysing biophysical properties...")
    l_props = compute_property_profile(l_pred)
    d_props = compute_property_profile(d_pred)

    # Step 4: Assess therapeutic potential
    typer.echo("   Step 4/4: Assessing therapeutic potential...")
    assessment = assess_therapeutic_potential(
        protein_name=name,
        comparison=comparison,
        l_properties=l_props,
        d_properties=d_props,
        pdb_id=pdb_id,
    )

    typer.echo(f"\n{'═' * 60}")
    typer.echo(f"  🪞 THERAPEUTIC ASSESSMENT — {name}")
    typer.echo(f"{'═' * 60}")
    typer.echo(f"\n  Viability:          {assessment.therapeutic_viability}")
    typer.echo(f"  Protease resistance: {assessment.estimated_protease_resistance}")
    typer.echo(f"  Immunogenicity:      {assessment.estimated_immunogenicity}")
    typer.echo(f"  Pocket conserved:    {assessment.binding_pocket_conserved}")
    typer.echo(f"\n  {'─' * 56}")
    typer.echo(f"\n{assessment.rationale}")

    if output:
        data = assessment.model_dump()
        # Convert non-serialisable fields
        data["comparison"] = {
            "rmsd": comparison.rmsd,
            "tm_score": comparison.tm_score,
            "aligned_length": comparison.aligned_length,
            "secondary_structure_match": comparison.secondary_structure_match,
        }
        data["l_properties"] = l_props.model_dump()
        data["d_properties"] = d_props.model_dump()
        Path(output).write_text(json.dumps(data, indent=2, default=str))
        typer.echo(f"\n  📄 Assessment saved: {output}")


# ── Info Command ───────────────────────────────────────────────────

@app.command()
def info(
    residue: str = typer.Option(None, "--residue", "-r", help="Single-letter amino acid code"),
    all_residues: bool = typer.Option(False, "--all", "-a", help="Show all amino acids"),
) -> None:
    """Display chirality information for amino acids."""
    from .mirror import (
        L_AMINO_ACID_SMILES,
        D_AMINO_ACID_SMILES,
        AA_1TO3,
        verify_mirror_smiles,
    )

    if all_residues:
        residues = sorted(L_AMINO_ACID_SMILES.keys())
    elif residue:
        residues = [residue.upper()]
    else:
        typer.echo("Specify --residue or --all. Use --help for details.")
        raise typer.Exit(1)

    typer.echo(f"\n{'Residue':<10} {'3-letter':<10} {'L-SMILES':<40} {'D-SMILES':<40}")
    typer.echo("─" * 100)

    for aa in residues:
        if aa not in L_AMINO_ACID_SMILES:
            typer.echo(f"  Unknown residue: {aa}")
            continue

        three = AA_1TO3.get(aa, "???")
        l_smiles = L_AMINO_ACID_SMILES[aa]
        d_smiles = D_AMINO_ACID_SMILES[aa]

        result = verify_mirror_smiles(aa)
        status = "✅" if result.get("valid") else "❌"

        typer.echo(f"{aa:<10} {three:<10} {l_smiles:<40} {d_smiles:<40} {status}")


# ── Entry Point ────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the MirrorFold CLI."""
    app()


if __name__ == "__main__":
    main()
