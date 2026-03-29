"""
3-D Structure Visualisation
============================

Interactive visualisation of L- and D-protein structures using
py3Dmol, with matplotlib plots for Ramachandran comparisons.

Colour palette — NeoForge brand
---------------------------------
* **Deep Navy** ``#0D1B2A`` — backgrounds
* **Neon Teal** ``#00D4AA`` — L-protein / highlights
* **Coral**     ``#FF6B6B`` — D-protein
* **Gold**      ``#FFD93D`` — divergent regions

py3Dmol primer
---------------
py3Dmol is a Jupyter-friendly wrapper around 3Dmol.js.  Each
``py3Dmol.view()`` creates an interactive WebGL canvas that
renders PDB structures with ribbon, cartoon, stick, or surface
representations.  The viewer returns an HTML object that can be
embedded in notebooks.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from .models import ComparisonReport, StructurePrediction

logger = logging.getLogger(__name__)

# ── Colour Palette ──────────────────────────────────────────────────

DEEP_NAVY = "#0D1B2A"
NEON_TEAL = "#00D4AA"
CORAL = "#FF6B6B"
GOLD = "#FFD93D"
WHITE = "#FFFFFF"
GREY = "#8B95A2"


# ── 3D Structure Views ─────────────────────────────────────────────

def view_structure(
    prediction: StructurePrediction,
    color_by: str = "plddt",
    width: int = 600,
    height: int = 400,
    style: str = "cartoon",
):
    """Render a single predicted structure interactively.

    Parameters
    ----------
    prediction : StructurePrediction
        Structure to visualise.
    color_by : str
        Colouring scheme:
        * ``"plddt"`` — blue (high) to red (low) by confidence
        * ``"chain"`` — colour by chain ID
        * ``"secondary"`` — helix/sheet/coil colouring
        * ``"uniform"`` — single colour (teal for L, coral for D)
    width, height : int
        Canvas size in pixels.
    style : str
        Molecular representation: ``"cartoon"``, ``"stick"``,
        ``"sphere"``, ``"line"``.

    Returns
    -------
    py3Dmol.view
        Interactive 3D viewer (renders in Jupyter notebooks).
    """
    try:
        import py3Dmol
    except ImportError:
        logger.error("py3Dmol not installed. Install with: pip install py3Dmol")
        return None

    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(prediction.pdb_string, "pdb")

    if color_by == "plddt":
        viewer.setStyle({style: {"colorscheme": {"prop": "b", "gradient": "rwb", "min": 0, "max": 100}}})
    elif color_by == "uniform":
        color = NEON_TEAL if prediction.chirality == "L" else CORAL
        viewer.setStyle({style: {"color": color}})
    elif color_by == "secondary":
        viewer.setStyle({style: {"colorscheme": "ssJmol"}})
    else:
        viewer.setStyle({style: {"colorscheme": "chainHetatm"}})

    viewer.setBackgroundColor(DEEP_NAVY)
    viewer.zoomTo()

    label = f"{prediction.chirality}-protein | pLDDT: {prediction.mean_plddt:.1f}"
    viewer.addLabel(
        label,
        {
            "position": {"x": 0, "y": 0, "z": 0},
            "backgroundColor": DEEP_NAVY,
            "fontColor": WHITE,
            "fontSize": 12,
        },
    )

    return viewer


def side_by_side(
    l_prediction: StructurePrediction,
    d_prediction: StructurePrediction,
    width: int = 800,
    height: int = 400,
    style: str = "cartoon",
):
    """Show L and D structures side by side in a split view.

    Parameters
    ----------
    l_prediction : StructurePrediction
        L-protein structure.
    d_prediction : StructurePrediction
        D-protein structure.
    width, height : int
        Total canvas size.
    style : str
        Molecular representation.

    Returns
    -------
    py3Dmol.view
        Interactive split viewer.
    """
    try:
        import py3Dmol
    except ImportError:
        logger.error("py3Dmol not installed.")
        return None

    viewer = py3Dmol.view(
        width=width, height=height,
        viewergrid=(1, 2),
    )

    # Left panel: L-protein
    viewer.addModel(l_prediction.pdb_string, "pdb", viewer=(0, 0))
    viewer.setStyle(
        {style: {"color": NEON_TEAL}},
        viewer=(0, 0),
    )

    # Right panel: D-protein
    viewer.addModel(d_prediction.pdb_string, "pdb", viewer=(0, 1))
    viewer.setStyle(
        {style: {"color": CORAL}},
        viewer=(0, 1),
    )

    for panel in [(0, 0), (0, 1)]:
        viewer.setBackgroundColor(DEEP_NAVY, viewer=panel)
        viewer.zoomTo(viewer=panel)

    return viewer


def overlay_structures(
    l_prediction: StructurePrediction,
    d_prediction: StructurePrediction,
    width: int = 600,
    height: int = 400,
    style: str = "cartoon",
    opacity: float = 0.7,
):
    """Overlay L and D structures in the same viewport.

    Parameters
    ----------
    l_prediction, d_prediction : StructurePrediction
        Structures to overlay.
    width, height : int
        Canvas size.
    style : str
        Representation.
    opacity : float
        Opacity for each structure.

    Returns
    -------
    py3Dmol.view
        Interactive overlay viewer.
    """
    try:
        import py3Dmol
    except ImportError:
        logger.error("py3Dmol not installed.")
        return None

    viewer = py3Dmol.view(width=width, height=height)

    # L-protein
    viewer.addModel(l_prediction.pdb_string, "pdb")
    viewer.setStyle(
        {"model": 0},
        {style: {"color": NEON_TEAL, "opacity": opacity}},
    )

    # D-protein
    viewer.addModel(d_prediction.pdb_string, "pdb")
    viewer.setStyle(
        {"model": 1},
        {style: {"color": CORAL, "opacity": opacity}},
    )

    viewer.setBackgroundColor(DEEP_NAVY)
    viewer.zoomTo()

    return viewer


def color_by_distance(
    comparison: ComparisonReport,
    width: int = 600,
    height: int = 400,
):
    """Colour the L-protein by per-residue distance to the D-form.

    Blue → well-conserved regions.
    Red → divergent regions.

    Parameters
    ----------
    comparison : ComparisonReport
        Structural comparison with per-residue distances.
    width, height : int
        Canvas size.

    Returns
    -------
    py3Dmol.view or None
    """
    try:
        import py3Dmol
    except ImportError:
        logger.error("py3Dmol not installed.")
        return None

    if not comparison.per_residue_distances:
        logger.warning("No per-residue distances available.")
        return None

    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(comparison.l_structure.pdb_string, "pdb")

    # Colour each residue by distance
    max_dist = max(comparison.per_residue_distances) if comparison.per_residue_distances else 1.0
    max_dist = max(max_dist, 0.01)  # avoid division by zero

    for i, dist in enumerate(comparison.per_residue_distances):
        # Normalise to [0, 1]
        frac = min(dist / max_dist, 1.0)

        # Blue → Red gradient
        r = int(255 * frac)
        g = 0
        b = int(255 * (1 - frac))
        hex_color = f"#{r:02x}{g:02x}{b:02x}"

        viewer.setStyle(
            {"resi": i + 1},
            {"cartoon": {"color": hex_color}},
        )

    viewer.setBackgroundColor(DEEP_NAVY)
    viewer.zoomTo()

    return viewer


# ── Ramachandran Plot ──────────────────────────────────────────────

def _calculate_torsion_angles(
    pdb_string: str,
) -> tuple[list[float], list[float]]:
    """Extract φ and ψ torsion angles from a PDB string.

    Returns lists of phi and psi angles in degrees.
    Terminal residues have no φ (first) or ψ (last).
    """
    # Extract backbone atoms
    backbone: dict[int, dict[str, np.ndarray]] = {}

    for line in pdb_string.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name not in ("N", "CA", "C"):
            continue
        res_seq = int(line[22:26].strip())
        coords = np.array([
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        ])
        if res_seq not in backbone:
            backbone[res_seq] = {}
        backbone[res_seq][atom_name] = coords

    indices = sorted(backbone.keys())
    phis: list[float] = []
    psis: list[float] = []

    for i, res_idx in enumerate(indices):
        atoms = backbone[res_idx]
        if not all(k in atoms for k in ("N", "CA", "C")):
            continue

        # φ: C(i-1) — N(i) — CA(i) — C(i)
        if i > 0:
            prev = backbone.get(indices[i - 1], {})
            if "C" in prev:
                phi = _dihedral(prev["C"], atoms["N"], atoms["CA"], atoms["C"])
                phis.append(phi)

        # ψ: N(i) — CA(i) — C(i) — N(i+1)
        if i < len(indices) - 1:
            nxt = backbone.get(indices[i + 1], {})
            if "N" in nxt:
                psi = _dihedral(atoms["N"], atoms["CA"], atoms["C"], nxt["N"])
                psis.append(psi)

    return phis, psis


def _dihedral(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> float:
    """Compute dihedral angle in degrees."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    norm1 = np.linalg.norm(n1)
    norm2 = np.linalg.norm(n2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    n1 /= norm1
    n2 /= norm2

    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))

    return math.degrees(math.atan2(y, x))


def ramachandran_comparison(
    l_prediction: StructurePrediction,
    d_prediction: StructurePrediction,
    save_path: Optional[str] = None,
    figsize: tuple[float, float] = (12, 5),
):
    """Plot Ramachandran diagrams for L and D protein side by side.

    For L-proteins, the populated regions are in the upper-left
    quadrant (φ < 0).  For D-proteins, the distribution should be
    *mirrored* — populated regions shift to the lower-right
    quadrant (φ > 0, ψ < 0).

    This mirror relationship is a direct consequence of the
    inverted backbone chirality.

    Parameters
    ----------
    l_prediction, d_prediction : StructurePrediction
        Predicted structures.
    save_path : str, optional
        Save figure to this path.  If None, ``plt.show()`` is called.
    figsize : tuple
        Figure dimensions.

    Returns
    -------
    matplotlib.figure.Figure
        The Ramachandran plot figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not installed.")
        return None

    phi_l, psi_l = _calculate_torsion_angles(l_prediction.pdb_string)
    phi_d, psi_d = _calculate_torsion_angles(d_prediction.pdb_string)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor(DEEP_NAVY)

    for ax, phis, psis, label, color in [
        (ax1, phi_l, psi_l, "L-protein", NEON_TEAL),
        (ax2, phi_d, psi_d, "D-protein", CORAL),
    ]:
        ax.set_facecolor(DEEP_NAVY)
        ax.scatter(phis, psis, c=color, s=15, alpha=0.7, edgecolors="none")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xlabel("φ (degrees)", color=WHITE, fontsize=10)
        ax.set_ylabel("ψ (degrees)", color=WHITE, fontsize=10)
        ax.set_title(f"Ramachandran — {label}", color=WHITE, fontsize=12)
        ax.axhline(0, color=GREY, linewidth=0.5, alpha=0.5)
        ax.axvline(0, color=GREY, linewidth=0.5, alpha=0.5)
        ax.tick_params(colors=WHITE)
        for spine in ax.spines.values():
            spine.set_color(GREY)
        ax.set_aspect("equal")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=DEEP_NAVY)
        logger.info("Ramachandran plot saved to %s", save_path)

    return fig


# ── Summary HTML ───────────────────────────────────────────────────

def generate_comparison_html(
    comparison: ComparisonReport,
    output_path: Optional[str] = None,
) -> str:
    """Generate a self-contained HTML report with embedded 3Dmol.js.

    This creates a standalone HTML file that can be opened in any
    browser, with interactive 3D viewers for both L and D structures.

    Parameters
    ----------
    comparison : ComparisonReport
        Structural comparison.
    output_path : str, optional
        Path to write HTML file.

    Returns
    -------
    str
        HTML content.
    """
    rmsd = comparison.rmsd or 0.0
    tm = comparison.tm_score or 0.0
    ss_match = (comparison.secondary_structure_match or 0.0) * 100
    n_conserved = len(comparison.conserved_regions)
    n_divergent = len(comparison.divergent_regions)

    pdb_l = comparison.l_structure.pdb_string.replace("`", "\\`").replace("${", "\\${")
    pdb_d = comparison.d_structure.pdb_string.replace("`", "\\`").replace("${", "\\${")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MirrorFold — L vs D Comparison</title>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {{
            background-color: {DEEP_NAVY};
            color: {WHITE};
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0; padding: 20px;
        }}
        h1 {{ color: {NEON_TEAL}; text-align: center; }}
        .metrics {{
            display: flex; justify-content: center; gap: 40px;
            margin: 20px 0;
        }}
        .metric {{
            text-align: center; padding: 15px;
            border: 1px solid {NEON_TEAL}; border-radius: 8px;
        }}
        .metric-value {{ font-size: 2em; color: {NEON_TEAL}; }}
        .metric-label {{ font-size: 0.9em; color: {GREY}; }}
        .viewers {{
            display: flex; justify-content: center; gap: 20px;
            margin: 20px 0;
        }}
        .viewer-panel {{ text-align: center; }}
        .viewer-label {{ font-size: 1.1em; margin-bottom: 8px; }}
        .l-label {{ color: {NEON_TEAL}; }}
        .d-label {{ color: {CORAL}; }}
    </style>
</head>
<body>
    <h1>🪞 MirrorFold — L vs D Protein Comparison</h1>

    <div class="metrics">
        <div class="metric">
            <div class="metric-value">{rmsd:.2f} Å</div>
            <div class="metric-label">RMSD</div>
        </div>
        <div class="metric">
            <div class="metric-value">{tm:.3f}</div>
            <div class="metric-label">TM-score</div>
        </div>
        <div class="metric">
            <div class="metric-value">{ss_match:.0f}%</div>
            <div class="metric-label">SS Match</div>
        </div>
        <div class="metric">
            <div class="metric-value">{n_conserved}</div>
            <div class="metric-label">Conserved Regions</div>
        </div>
        <div class="metric">
            <div class="metric-value">{n_divergent}</div>
            <div class="metric-label">Divergent Regions</div>
        </div>
    </div>

    <div class="viewers">
        <div class="viewer-panel">
            <div class="viewer-label l-label">L-Protein</div>
            <div id="viewer-l" style="width: 400px; height: 350px; border: 1px solid {NEON_TEAL}; border-radius: 4px;"></div>
        </div>
        <div class="viewer-panel">
            <div class="viewer-label d-label">D-Protein</div>
            <div id="viewer-d" style="width: 400px; height: 350px; border: 1px solid {CORAL}; border-radius: 4px;"></div>
        </div>
    </div>

    <script>
        var viewerL = $3Dmol.createViewer("viewer-l", {{backgroundColor: "{DEEP_NAVY}"}});
        viewerL.addModel(`{pdb_l}`, "pdb");
        viewerL.setStyle({{cartoon: {{color: "{NEON_TEAL}"}}}});
        viewerL.zoomTo();
        viewerL.render();

        var viewerD = $3Dmol.createViewer("viewer-d", {{backgroundColor: "{DEEP_NAVY}"}});
        viewerD.addModel(`{pdb_d}`, "pdb");
        viewerD.setStyle({{cartoon: {{color: "{CORAL}"}}}});
        viewerD.zoomTo();
        viewerD.render();
    </script>
</body>
</html>"""

    if output_path:
        with open(output_path, "w") as f:
            f.write(html)
        logger.info("HTML report saved to %s", output_path)

    return html
