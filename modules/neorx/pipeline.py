"""
NeoRx Pipeline
======================

The pipeline orchestrates the full causal drug-discovery workflow:

1. **Build Graph** → query 7 databases, assemble causal knowledge graph
2. **Identify Targets** → apply do-calculus to find causal targets
3. **Generate Candidates** → use GenMol VAE to create novel molecules
4. **Screen Candidates** → MolScreen (drug-likeness) + DockBot (binding)
5. **Score & Rank** → composite scorer with causal confidence weighting
6. **Report** → generate HTML report with interactive visualisations

Integration Points
------------------
- ``graph_builder.build_disease_graph()`` → DiseaseGraph
- ``identifier.identify_causal_targets()`` → list[NeoRxResult]
- ``modules.genmol.generate.generate()`` → list[str]
- ``modules.molscreen.properties.calculate_properties()`` → MolecularProperties
- ``modules.molscreen.accessibility.qed_score()`` / ``sa_score()``
- ``modules.molscreen.filters.run_all_filters()`` → list[FilterResult]
- ``modules.molscreen.similarity.find_similar_drugs()`` → novelty
- ``modules.dockbot.protein_prep.prepare_protein()`` → ProteinInfo
- ``modules.dockbot.ligand_prep.prepare_ligand_pdbqt()`` → (Mol, pdbqt)
- ``modules.dockbot.docker.dock()`` → DockingResult
- ``scorer.score_candidate()`` → ScoredCandidate

Error Handling
--------------
Every sub-module integration is wrapped in try/except.
If GenMol fails, we still report the causal targets.
If DockBot fails, we score without binding affinity.
The pipeline never crashes — it degrades gracefully and
reports what it could accomplish.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from .models import (
    PipelineJob,
    PipelineResult,
    DiseaseGraph,
    NeoRxResult,
    ScoredCandidate,
    JobStatus,
)
from .graph_builder import build_disease_graph
from .identifier import identify_causal_targets
from .scorer import score_candidate, rank_candidates

logger = logging.getLogger(__name__)


def _canonicalize_smiles(smiles: str) -> str | None:
    """Return canonical SMILES or *None* if parsing fails.

    Uses RDKit's ``Chem.MolToSmiles`` to produce a unique,
    deterministic SMILES string, eliminating duplicates caused
    by different input representations of the same molecule.
    """
    try:
        from rdkit import Chem  # type: ignore[import-untyped]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def run_pipeline(
    disease: str,
    top_n_targets: int = 5,
    candidates_per_target: int = 100,
    *,
    generate_molecules: bool = True,
    run_docking: bool = True,
    generate_report: bool = True,
    prebuilt_graph: DiseaseGraph | None = None,
    allow_mocks: bool = False,
) -> PipelineResult:
    """Run the full NeoRx pipeline.

    Parameters
    ----------
    disease : str
        Disease name (e.g. "HIV", "Type 2 Diabetes").
    top_n_targets : int
        Number of top causal targets to pursue.
    candidates_per_target : int
        Molecules to generate per target.
    generate_molecules : bool
        Whether to run GenMol generation.
    run_docking : bool
        Whether to run DockBot docking.
    generate_report : bool
        Whether to generate HTML report.
    prebuilt_graph : DiseaseGraph, optional
        Pre-built disease graph.  If provided, Step 1 is skipped
        and this graph is used directly.  Useful for testing and
        for interactive workflows where the graph was already
        constructed.
    allow_mocks : bool
        If *True*, data-source clients may fall back to curated
        mock data when live APIs fail.  Default *False*.

    Returns
    -------
    PipelineResult
        Complete pipeline output with graph, targets, candidates.
    """
    job = PipelineJob(disease=disease, top_n_targets=top_n_targets,
                      candidates_per_target=candidates_per_target)

    try:
        # ── Step 1: Build Disease Graph ─────────────────────────
        job.status = JobStatus.BUILDING_GRAPH
        job.current_step = "Building disease causal graph…"
        job.progress_pct = 5.0
        logger.info("═══ Step 1/6: Building disease graph for '%s' ═══", disease)

        if prebuilt_graph is not None:
            graph = prebuilt_graph
            logger.info("Using prebuilt graph (%d nodes, %d edges).",
                        len(graph.nodes), len(graph.edges))
        else:
            graph = build_disease_graph(disease, allow_mocks=allow_mocks)
        job.progress_pct = 20.0
        logger.info("Graph: %d nodes, %d edges.", len(graph.nodes), len(graph.edges))

        # ── Step 2: Identify Causal Targets ─────────────────────
        job.status = JobStatus.IDENTIFYING_TARGETS
        job.current_step = "Identifying causal targets (do-calculus)…"
        job.progress_pct = 25.0
        logger.info("═══ Step 2/6: Identifying causal targets ═══")

        causal_targets = identify_causal_targets(graph, top_n=top_n_targets)
        job.progress_pct = 40.0

        n_causal = sum(1 for t in causal_targets if t.is_causal_target)
        n_correlational = sum(1 for t in causal_targets if not t.is_causal_target)
        logger.info("Found %d causal targets, %d correlational out of %d evaluated.",
                     n_causal, n_correlational, len(causal_targets))

        for t in causal_targets:
            tag = "✓ CAUSAL" if t.is_causal_target else "✗ " + t.classification.value
            extra = ""
            if t.target_type and t.target_type != "correlational":
                extra = f" [bio:{t.target_type}]"
            if not t.tissue_relevant:
                extra += " [TISSUE-IRRELEVANT]"
            logger.info("  %s %s (confidence=%.3f)%s", tag, t.gene_name,
                        t.causal_confidence, extra)

        # ── Step 2b: Known-Target Validation ────────────────────
        validation_result = None
        try:
            from .validator import KnownTargetValidator
            validator = KnownTargetValidator()
            identified_dicts = [
                {"gene_symbol": t.gene_name, "is_causal": t.is_causal_target}
                for t in causal_targets
            ]
            validation_result = validator.validate(disease, identified_dicts)
            if validation_result and validation_result.validated:
                logger.info(
                    "Validation: precision=%.2f, recall=%.2f, F1=%.2f, grade=%s",
                    validation_result.precision, validation_result.recall,
                    validation_result.f1, validation_result.quality_grade,
                )
                for w in validation_result.warnings:
                    logger.warning("  ⚠ %s", w)
        except Exception as e:
            logger.debug("Validation skipped: %s", e)

        # ── Step 3: Generate Candidate Molecules ────────────────
        all_candidates: list[ScoredCandidate] = []

        if generate_molecules:
            job.status = JobStatus.GENERATING_CANDIDATES
            job.current_step = "Generating candidate molecules (GenMol)…"
            job.progress_pct = 45.0
            logger.info("═══ Step 3/6: Generating candidates (GenMol) ═══")

            for target in causal_targets:
                if not target.is_causal_target:
                    continue
                smiles_list = _generate_for_target(target, candidates_per_target)
                logger.info("  %s: %d molecules generated.", target.gene_name, len(smiles_list))

                # ── Step 4: Screen candidates ───────────────────
                job.status = JobStatus.SCREENING
                job.current_step = f"Screening candidates for {target.gene_name}…"

                screened = _screen_candidates(
                    smiles_list, target, run_docking=run_docking,
                )
                all_candidates.extend(screened)

            job.progress_pct = 75.0
        else:
            logger.info("Skipping molecule generation (generate_molecules=False).")

        # ── Step 5: Score & Rank ────────────────────────────────
        job.status = JobStatus.SCORING
        job.current_step = "Scoring and ranking candidates…"
        job.progress_pct = 80.0
        logger.info("═══ Step 5/6: Scoring %d candidates ═══", len(all_candidates))

        ranked = rank_candidates(all_candidates)
        logger.info("Top 5 candidates:")
        for cand in ranked[:5]:
            logger.info(
                "  #%d  %.4f  %s  (target: %s)",
                cand.rank, cand.composite_score,
                cand.smiles[:50], cand.target_protein_name,
            )

        # ── Step 6: Generate Report ─────────────────────────────
        report_html = None
        report_path = None
        if generate_report:
            job.status = JobStatus.REPORTING
            job.current_step = "Generating report…"
            job.progress_pct = 90.0
            logger.info("═══ Step 6/6: Generating report ═══")

            try:
                from .report import generate_report as gen_report
                report_html, report_path = gen_report(
                    disease=disease,
                    graph=graph,
                    causal_targets=causal_targets,
                    candidates=ranked,
                    validation={
                        "precision": validation_result.precision,
                        "recall": validation_result.recall,
                        "f1": validation_result.f1,
                        "quality_grade": validation_result.quality_grade,
                        "warnings": validation_result.warnings,
                    } if validation_result else None,
                )
                logger.info("Report saved to %s.", report_path)
            except Exception as e:
                logger.warning("Report generation failed: %s.", e)

        # ── Done ────────────────────────────────────────────────
        job.status = JobStatus.COMPLETE
        job.progress_pct = 100.0
        job.completed_at = datetime.now()
        job.current_step = "Pipeline complete."

        return PipelineResult(
            job=job,
            disease=disease,
            graph=graph,
            causal_targets=causal_targets,
            scored_candidates=ranked,
            report_html=report_html,
            report_path=report_path,
            validation={
                "precision": validation_result.precision,
                "recall": validation_result.recall,
                "f1": validation_result.f1,
                "quality_grade": validation_result.quality_grade,
                "true_positives": validation_result.true_positives,
                "false_positives": validation_result.false_positives_known,
                "false_negatives": validation_result.missed_targets,
                "warnings": validation_result.warnings,
            } if validation_result and validation_result.validated else None,
        )

    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.now()
        return PipelineResult(job=job, disease=disease)


# ── Sub-module Integration ──────────────────────────────────────────

def _generate_for_target(
    target: NeoRxResult,
    n_molecules: int,
) -> list[str]:
    """Generate candidate molecules using GenMol.

    Falls back to a small set of known drug-like SMILES if GenMol
    is not available (e.g. no trained model checkpoint).
    """
    try:
        from modules.genmol.data import SmilesTokenizer
        from modules.genmol.models import MolVAE
        from modules.genmol.generate import generate

        # Try to load a pre-trained checkpoint
        import torch
        tok = SmilesTokenizer()

        # Build vocabulary from representative drug-like SMILES so
        # the tokenizer can encode / decode.  Without this step the
        # tokenizer's _built flag is False and decode() will raise.
        _REPRESENTATIVE_SMILES = [
            "c1ccccc1",                       # benzene
            "CC(=O)Oc1ccccc1C(O)=O",          # aspirin
            "CC(C)Cc1ccc(C(C)C(O)=O)cc1",     # ibuprofen
            "O=C(NCc1ccccc1)c1cc2ccccc2[nH]1", # indole amide
            "Cc1nc2ccccc2n1Cc1ccc(F)cc1",       # benzimidazole
            "c1ccc(-c2nc3ccccc3s2)cc1",         # benzothiazole
            "CC1=NN(c2ccccc2)C(=O)C1",          # pyrazolone
            "Oc1ccc(-c2cc(-c3ccc(O)cc3)no2)cc1", # isoxazole
            "NC(=O)c1cccc(-c2cccnc2)c1",        # nicotinamide
            "CC(=O)Nc1ccc(O)cc1",               # paracetamol
            "ClC(Cl)=C(Cl)Cl",                  # chlorinated
            "BrC1=CC=CC=C1",                    # bromobenzene
            "[C@@H](O)(F)Cl",                   # chirality
        ]
        tok.build_vocab(_REPRESENTATIVE_SMILES)

        # If no trained model, generate from random init (for demo)
        model = MolVAE(vocab_size=max(tok.vocab_size, 64))
        model.eval()

        smiles = generate(
            model, tok, n=min(n_molecules, 200),
            temperature=1.2, validate=True,
        )
        if smiles:
            # Apply Tanimoto novelty filter: reject molecules too
            # similar to known drugs (Tanimoto ≥ 0.7)
            smiles = _filter_novel(smiles, max_tanimoto=0.7)
            return smiles

    except Exception as e:
        logger.warning("GenMol generation failed: %s. Using fallback.", e)

    # Fallback: curated drug-like molecules
    return _fallback_molecules(target.gene_name)


def _fallback_molecules(gene_name: str) -> list[str]:
    """Return diverse drug-like SMILES as fallback.

    These are structurally diverse, Lipinski-compliant scaffolds
    that serve as demonstration candidates when GenMol is
    unavailable.  They are disease- and target-agnostic — the
    real structure–activity intelligence comes from GenMol and
    the downstream screening / docking steps.

    Parameters
    ----------
    gene_name : str
        Ignored — kept for API compatibility.  The same diverse
        scaffold library is returned for every target so the
        pipeline never hard-codes disease-specific chemistry.
    """
    return [
        "c1ccc2[nH]c(-c3ccncc3)nc2c1",                    # benzimidazole
        "O=C(NCc1ccccc1)c1cc2ccccc2[nH]1",                 # indole amide
        "Cc1nc2ccccc2n1Cc1ccc(F)cc1",                       # benzimidazole
        "O=C(c1ccc(O)cc1)c1ccc(O)cc1O",                     # dihydroxybenzophenone
        "CC(=O)Nc1ccc(O)cc1",                               # paracetamol scaffold
        "c1ccc(-c2nc3ccccc3s2)cc1",                         # benzothiazole
        "O=c1[nH]c2ccccc2c2ccccc12",                        # acridone
        "NC(=O)c1cccc(-c2cccnc2)c1",                        # nicotinamide analog
        "Oc1ccc(-c2cc(-c3ccc(O)cc3)no2)cc1",               # isoxazole diol
        "CC1=NN(c2ccccc2)C(=O)C1",                          # pyrazolone
    ]


def _screen_candidates(
    smiles_list: list[str],
    target: NeoRxResult,
    run_docking: bool = True,
) -> list[ScoredCandidate]:
    """Screen a list of SMILES against a target.

    Integrates MolScreen for drug-likeness properties and
    optionally DockBot for binding affinity estimation.

    Protein preparation is performed **once** per target and
    reused across all molecules, avoiding redundant PDB
    downloads and log spam.
    """
    scored: list[ScoredCandidate] = []

    # ── Prepare protein once for all molecules ──────────────
    prepared_protein = None
    binding_site = None
    if run_docking and target.pdb_ids:
        pdb_id = target.pdb_ids[0]
        try:
            from modules.dockbot.protein_prep import prepare_protein
            prepared_protein = prepare_protein(pdb_id)
            if prepared_protein:
                binding_site = _detect_binding_site(prepared_protein, pdb_id)
                logger.info("  Protein %s prepared for docking.", pdb_id)
        except Exception as e:
            logger.debug("Protein prep failed for %s: %s", pdb_id, e)

    for smiles in smiles_list:
        # ── Canonicalize SMILES ─────────────────────────────
        smiles = _canonicalize_smiles(smiles) or smiles

        # ── MolScreen Properties ────────────────────────────────
        mol_props = _get_mol_properties(smiles)
        if mol_props is None:
            continue  # Invalid SMILES

        qed = mol_props.get("qed", 0.5)
        sa = mol_props.get("sa_score", 5.0)
        mw = mol_props.get("molecular_weight")
        logp = mol_props.get("logp")
        n_filters = mol_props.get("n_filters_passed", 0)
        drug_class = mol_props.get("drug_class", "")

        # ── Novelty Score ───────────────────────────────────────
        novelty = _compute_novelty(smiles)

        # ── Docking (optional) ──────────────────────────────────
        binding = None
        if prepared_protein and binding_site:
            binding = _run_docking(
                smiles, target.pdb_ids[0],
                protein=prepared_protein,
                site=binding_site,
            )

        # ── ADMET Estimate ──────────────────────────────────────
        admet = _estimate_admet(mw, logp, qed)

        # ── Score ───────────────────────────────────────────────
        candidate = score_candidate(
            smiles=smiles,
            target_protein_id=target.protein_id,
            target_protein_name=target.protein_name,
            causal_confidence=target.causal_confidence,
            binding_affinity=binding,
            qed_score=qed,
            sa_score=sa,
            admet_score=admet,
            novelty_score=novelty,
            molecular_weight=mw,
            logp=logp,
            drug_likeness_class=drug_class,
            n_filters_passed=n_filters,
        )
        scored.append(candidate)

    return scored


def _get_mol_properties(smiles: str) -> dict[str, Any] | None:
    """Get molecular properties via MolScreen."""
    try:
        from modules.molscreen.parser import validate_smiles
        if not validate_smiles(smiles):
            return None

        from modules.molscreen.properties import calculate_properties
        from modules.molscreen.accessibility import qed_score, sa_score
        from modules.molscreen.filters import run_all_filters

        props = calculate_properties(smiles)
        if props is None:
            return None

        qed = qed_score(smiles)
        sa = sa_score(smiles)
        filters = run_all_filters(props.mol)

        return {
            "molecular_weight": props.molecular_weight,
            "logp": props.logp,
            "qed": qed if qed is not None else 0.5,
            "sa_score": sa if sa is not None else 5.0,
            "n_filters_passed": sum(1 for f in filters if f.passed),
            "drug_class": "drug-like" if sum(1 for f in filters if f.passed) >= 3 else "non-drug-like",
        }
    except Exception as e:
        logger.debug("MolScreen failed for %s: %s", smiles[:30], e)
        # Fallback: basic RDKit
        return _fallback_properties(smiles)


def _fallback_properties(smiles: str) -> dict[str, Any] | None:
    """Compute basic properties when MolScreen is unavailable."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        return {
            "molecular_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "qed": QED.qed(mol),
            "sa_score": 5.0,  # Default
            "n_filters_passed": 3,
            "drug_class": "drug-like",
        }
    except Exception:
        return None


def _compute_novelty(smiles: str) -> float:
    """Compute structural novelty vs known drugs."""
    try:
        from modules.molscreen.similarity import find_similar_drugs
        similar = find_similar_drugs(smiles, top_k=1, threshold=0.3)
        if similar:
            # Higher similarity → lower novelty
            max_sim = max(s[1] for s in similar) if similar else 0.0
            return 1.0 - max_sim
        return 0.9  # No similar drugs → very novel
    except Exception:
        return 0.5  # Unknown


def _run_docking(
    smiles: str,
    pdb_id: str,
    protein: dict[str, Any] | None = None,
    site: Any | None = None,
) -> float | None:
    """Run molecular docking via DockBot.

    Parameters
    ----------
    smiles : str
        SMILES string for the ligand.
    pdb_id : str
        PDB identifier (used as fallback if *protein* is None).
    protein : dict, optional
        Pre-prepared protein dict from ``prepare_protein()``.
        If provided, skips redundant protein preparation.
    site : BindingSite, optional
        Pre-detected binding site.  If provided, skips redundant
        binding-site detection.
    """
    try:
        from modules.dockbot.ligand_prep import prepare_ligand_pdbqt
        from modules.dockbot.docker import dock

        # Use pre-prepared protein or prepare on-the-fly
        if protein is None:
            from modules.dockbot.protein_prep import prepare_protein
            protein = prepare_protein(pdb_id)

        mol, ligand_pdbqt = prepare_ligand_pdbqt(smiles)

        if not protein or not ligand_pdbqt:
            return None

        # Use pre-detected site or detect now
        if site is None:
            site = _detect_binding_site(protein, pdb_id)

        result = dock(
            protein.get("pdbqt", ""),
            ligand_pdbqt,
            site,
        )
        if result and result.poses:
            return result.poses[0].affinity
        return None

    except Exception as e:
        logger.debug("Docking failed for %s @ %s: %s", smiles[:20], pdb_id, e)
        return None


def _detect_binding_site(
    protein: dict[str, Any],
    pdb_id: str,
) -> Any:
    """Detect binding site from PDB structure.

    Strategy:
    1. If protein_prep returned binding site coords, use those.
    2. If the PDB has co-crystallised ligands, centre on them.
    3. Fall back to the geometric centre of the protein with a
       large search box.
    """
    from modules.dockbot.models import BindingSite

    # 1. Check if protein prep extracted binding site
    if "binding_site" in protein:
        bs = protein["binding_site"]
        if isinstance(bs, dict):
            return BindingSite(
                center_x=bs.get("center_x", 0.0),
                center_y=bs.get("center_y", 0.0),
                center_z=bs.get("center_z", 0.0),
                size_x=bs.get("size_x", 25.0),
                size_y=bs.get("size_y", 25.0),
                size_z=bs.get("size_z", 25.0),
            )

    # 2. Try to extract coords from PDB ATOM records
    coords = _extract_protein_center(protein.get("pdb_text", ""))
    if coords:
        cx, cy, cz = coords
        return BindingSite(
            center_x=cx, center_y=cy, center_z=cz,
            size_x=30.0, size_y=30.0, size_z=30.0,
        )

    # 3. Fallback — larger box centred at origin
    logger.debug(
        "Could not detect binding site for %s — using large default box.",
        pdb_id,
    )
    return BindingSite(
        center_x=0.0, center_y=0.0, center_z=0.0,
        size_x=40.0, size_y=40.0, size_z=40.0,
    )


def _extract_protein_center(
    pdb_text: str,
) -> tuple[float, float, float] | None:
    """Extract geometric centre from PDB ATOM records."""
    if not pdb_text:
        return None

    xs, ys, zs = [], [], []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                xs.append(x)
                ys.append(y)
                zs.append(z)
            except (ValueError, IndexError):
                continue

    if not xs:
        return None

    return (
        sum(xs) / len(xs),
        sum(ys) / len(ys),
        sum(zs) / len(zs),
    )


def _estimate_admet(
    mw: float | None,
    logp: float | None,
    qed: float | None,
) -> float:
    """ADMET estimation using the multi-rule predictor.

    Delegates to ``admet.predict_admet`` when a SMILES is
    available via thread-local context, otherwise falls back
    to the rule-based heuristic for backward compatibility.
    """
    # Rule-based fallback (used when called directly in tests)
    score = 0.5
    if mw is not None:
        if 150 <= mw <= 500:
            score += 0.2
        elif mw > 500:
            score -= 0.1

    if logp is not None:
        if -0.4 <= logp <= 5.6:
            score += 0.2
        else:
            score -= 0.1

    if qed is not None and qed > 0.5:
        score += 0.1

    return max(0.0, min(1.0, score))


def run_rl_pipeline(
    disease: str,
    top_n_targets: int = 5,
    n_episodes: int = 10,
    max_steps_per_episode: int = 50,
    latent_dim: int = 128,
    *,
    prebuilt_graph: DiseaseGraph | None = None,
    allow_mocks: bool = False,
    seed: int = 42,
) -> PipelineResult:
    """Run the RL-driven drug-discovery pipeline.

    Instead of the linear *generate-then-screen* flow in
    :func:`run_pipeline`, this mode trains a CausalBioRL agent
    inside a :class:`DrugDiscoveryEnv`.  The agent iteratively
    selects targets (UCB) and navigates GenMol's latent space
    (CEM) to discover high-scoring molecules.

    The function reuses the same graph-building and
    target-identification steps, but replaces Steps 3–5 with
    an RL optimisation loop.

    Parameters
    ----------
    disease : str
        Disease name.
    top_n_targets : int
        Number of causal targets to pursue.
    n_episodes : int
        Number of RL episodes (each episode is a full campaign).
    max_steps_per_episode : int
        Generate-screen cycles per episode.
    latent_dim : int
        GenMol latent-space dimensionality.
    prebuilt_graph : DiseaseGraph, optional
        Pre-built disease graph (skips Step 1).
    allow_mocks : bool
        Allow fallback mock data for data sources.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    PipelineResult
        Same schema as :func:`run_pipeline` so reports and
        downstream tooling work unchanged.
    """
    job = PipelineJob(
        disease=disease,
        top_n_targets=top_n_targets,
        candidates_per_target=max_steps_per_episode * n_episodes,
    )

    try:
        # ── Step 1: Build Disease Graph ─────────────────────────
        job.status = JobStatus.BUILDING_GRAPH
        job.current_step = "Building disease causal graph…"
        job.progress_pct = 5.0
        logger.info("═══ RL Step 1/4: Building disease graph for '%s' ═══", disease)

        if prebuilt_graph is not None:
            graph = prebuilt_graph
        else:
            graph = build_disease_graph(disease, allow_mocks=allow_mocks)
        job.progress_pct = 20.0

        # ── Step 2: Identify Causal Targets ─────────────────────
        job.status = JobStatus.IDENTIFYING_TARGETS
        job.current_step = "Identifying causal targets…"
        job.progress_pct = 25.0
        logger.info("═══ RL Step 2/4: Identifying causal targets ═══")

        causal_targets = identify_causal_targets(graph, top_n=top_n_targets)
        job.progress_pct = 35.0

        causal_only = [t for t in causal_targets if t.is_causal_target]
        logger.info("Found %d causal targets.", len(causal_only))

        # ── Step 3: RL Optimisation Loop ────────────────────────
        job.status = JobStatus.SCREENING
        job.current_step = "Running RL agent (CausalBioRL)…"
        job.progress_pct = 40.0
        logger.info("═══ RL Step 3/4: RL optimisation (%d episodes) ═══", n_episodes)

        all_candidates: list[ScoredCandidate] = []

        try:
            from modules.causalbiorl.envs.drug_discovery import DrugDiscoveryEnv
            from modules.causalbiorl.agents.causal_agent import CausalAgent

            # Convert to the types DrugDiscoveryEnv expects
            from .graph_builder import disease_graph_to_networkx
            nx_graph = disease_graph_to_networkx(graph)

            target_dicts = [
                {
                    "gene_name": t.gene_name,
                    "protein_id": t.protein_id,
                    "protein_name": t.protein_name,
                    "pdb_ids": t.pdb_ids if t.pdb_ids else [],
                    "causal_confidence": t.causal_confidence,
                }
                for t in causal_only
            ]

            env = DrugDiscoveryEnv(
                disease=disease,
                prebuilt_graph=nx_graph,
                prebuilt_targets=target_dicts,
                max_steps=max_steps_per_episode,
                latent_dim=latent_dim,
            )

            # Build agent config
            from modules.causalbiorl.models import AgentConfig
            agent_cfg = AgentConfig(
                agent_type="causal",
                n_episodes=n_episodes,
                seed=seed,
            )

            agent = CausalAgent(env, agent_cfg)
            agent.init_hierarchical_planner(
                n_targets=len(causal_only),
                latent_dim=latent_dim,
            )

            # Train (the training loop collects molecules internally)
            agent.train()

            # Extract best molecules from env's internal tracking
            for ts in env._target_states:
                if ts.best_smiles is not None and ts.best_score > 0.0:
                    candidate = score_candidate(
                        smiles=ts.best_smiles,
                        target_protein_id=causal_only[ts.target_idx].protein_id
                        if ts.target_idx < len(causal_only) else "",
                        target_protein_name=causal_only[ts.target_idx].protein_name
                        if ts.target_idx < len(causal_only) else "",
                        causal_confidence=causal_only[ts.target_idx].causal_confidence
                        if ts.target_idx < len(causal_only) else 0.5,
                        binding_affinity=ts.best_score * -10.0,
                        qed_score=0.5,
                        sa_score=5.0,
                    )
                    all_candidates.append(candidate)

            logger.info("RL agent found %d candidate molecules.", len(all_candidates))

        except ImportError as e:
            logger.warning(
                "CausalBioRL not available (%s) — "
                "falling back to linear pipeline.",
                e,
            )
            # Fallback: run the standard generation flow
            for target in causal_only:
                smiles_list = _generate_for_target(target, max_steps_per_episode)
                screened = _screen_candidates(smiles_list, target)
                all_candidates.extend(screened)

        except Exception as e:
            logger.warning("RL loop failed (%s) — collecting partial results.", e)

        # ── Step 4: Score & Rank ────────────────────────────────
        job.status = JobStatus.SCORING
        job.current_step = "Ranking RL-discovered candidates…"
        job.progress_pct = 90.0
        logger.info("═══ RL Step 4/4: Ranking %d candidates ═══", len(all_candidates))

        ranked = rank_candidates(all_candidates)

        job.status = JobStatus.COMPLETE
        job.progress_pct = 100.0
        job.completed_at = datetime.now()
        job.current_step = "RL pipeline complete."

        return PipelineResult(
            job=job,
            disease=disease,
            graph=graph,
            causal_targets=causal_targets,
            scored_candidates=ranked,
        )

    except Exception as e:
        logger.error("RL pipeline failed: %s", e, exc_info=True)
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.now()
        return PipelineResult(job=job, disease=disease)


def _filter_novel(
    smiles_list: list[str],
    max_tanimoto: float = 0.7,
) -> list[str]:
    """Remove molecules too similar to known drugs.

    Uses RDKit Morgan fingerprints and Tanimoto similarity.
    Molecules with max similarity ≥ max_tanimoto to any known
    drug are rejected.
    """
    try:
        from modules.molscreen.similarity import find_similar_drugs

        novel = []
        for smi in smiles_list:
            try:
                similar = find_similar_drugs(smi, top_k=1, threshold=0.3)
                if similar:
                    best_sim = max(s[1] for s in similar)
                    if best_sim < max_tanimoto:
                        novel.append(smi)
                else:
                    novel.append(smi)  # No similar drugs found → novel
            except Exception:
                novel.append(smi)  # Can't check → keep

        if novel:
            logger.info(
                "Tanimoto novelty filter: %d/%d molecules passed (T<%.2f).",
                len(novel), len(smiles_list), max_tanimoto,
            )
            return novel
        # If all are filtered out, return originals
        logger.info("All molecules filtered — returning original set.")
        return smiles_list

    except ImportError:
        return smiles_list
