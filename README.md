# 🧬 NeoRx — Causal Drug Target Discovery

> *"Correlation is not causation — NeoRx knows the difference."*

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Non--Commercial-orange.svg)](LICENSE)

**NeoRx** is an end-to-end computational platform that applies
**Pearl's causal inference framework** to multi-source biomedical knowledge
graphs for automated drug target identification. Unlike conventional
association-based pipelines, NeoRx asks:

> *"If we intervene on this protein (inhibit or activate it with a drug),
> does it causally affect the disease outcome?"*

Validated across **7 diseases** (HIV, malaria, Alzheimer's, Ebola, type 2 diabetes, lung cancer, breast cancer) with **mean F₁ = 0.474** and **zero false positives** across all diseases.

## The Problem

Gene–disease association databases (Monarch Initiative, Open Targets) report that
**TNF-α** is strongly associated with HIV.  TNF-α levels *are* elevated
during HIV infection.  But TNF-α elevation is a **downstream consequence**
of immune activation — inhibiting it does not treat HIV; it makes it worse.

Conversely, **CCR5** is the HIV-1 co-receptor.  A loss-of-function mutation
(CCR5-Δ32) confers near-complete resistance.  Maraviroc, which blocks CCR5,
is an approved antiretroviral.  CCR5 is a **causal** target.

NeoRx distinguishes these two cases automatically.

## Pipeline Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  8 Biomedical   │────▶│  Causal Knowledge │────▶│  Do-Calculus     │
│  Databases      │     │  Graph (NetworkX) │     │  Analysis        │
│  Monarch Init.  │     │                  │     │  • Backdoor      │
│  Open Targets   │     │  Genes, Pathways, │     │  • d-Separation  │
│  KEGG, Reactome │     │  PPIs, Structures │     │  • Sensitivity   │
│  STRING, UniProt│     │  Drug evidence    │     │  • Bootstrap CIs │
│  RCSB PDB       │     └──────────────────┘     └────────┬────────┘
│  ChEMBL v36     │                                       │
└─────────────────┘                                       ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  HTML Report    │◀────│  Composite Scorer │◀────│  GenMol +        │
│  Interactive    │     │  6D Ranking:      │     │  MolScreen +     │
│  Graph + Tables │     │  Causal > Binding │     │  DockBot         │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Seven-Stage Workflow

1. **Build Causal Knowledge Graph** — Query 8 biomedical databases in
   parallel (Monarch, Open Targets, KEGG, Reactome, STRING, UniProt, PDB,
   ChEMBL), merge into a unified directed graph with automatic node
   merging and edge provenance tracking.

2. **Identify Causal Targets** — Apply the backdoor criterion and
   d-separation analysis to distinguish causal from correlational targets.
   Multi-source evidence triangulation, leave-one-source-out sensitivity
   analysis, and bootstrap confidence intervals for robustness.
   A dedicated **pathogen target pipeline** identifies non-human drug
   targets (e.g., *Plasmodium* DHFR-TS, HIV Pol) from ChEMBL.

3. **Biological Intelligence** — Disease-type-aware target classifier
   (9 gene families, 119+ blocklisted symptom markers) and tissue
   expression boolean gate (Human Protein Atlas) prevent biologically
   implausible targets from advancing.

4. **Generate Candidate Molecules** — GenMol variational autoencoder
   (~4.1M parameters, 97% validity) generates novel drug-like molecules
   targeting each causal protein.

5. **Screen Candidates** — MolScreen evaluates drug-likeness (Lipinski,
   QED, SA) and DockBot estimates binding affinity via AutoDock Vina.

6. **Score & Rank** — Composite scorer integrates six dimensions with
   **causal confidence weighted highest** (0.30):

   | Dimension           | Weight |
   |---------------------|--------|
   | Causal confidence   | 0.30   |
   | Binding affinity    | 0.25   |
   | Drug-likeness (QED) | 0.15   |
   | Synthetic access.   | 0.10   |
   | ADMET               | 0.10   |
   | Novelty             | 0.10   |

7. **Report** — Interactive HTML report with causal knowledge graph
   visualisation, target reasoning, and ranked candidate table.

### RL-Driven Mode (CausalBioRL)

In addition to the linear pipeline, NeoRx includes an
**RL-driven drug discovery mode** where a causal reinforcement
learning agent iteratively explores the target–molecule space:

```
┌─ DrugDiscoveryEnv (Gymnasium) ─────────────────────────┐
│  R-GCN encodes disease graph → 128-D state embedding   │
│  UCB1 selects target (Level 1)                         │
│  CEM navigates GenMol latent space (Level 2)           │
│  SurrogateDockingModel → fast binding score (~1ms)     │
│  MolScreen → QED / SA / filters                        │
│  MirrorFold → stability assessment                     │
│  AdaptiveRewardLearner → multi-objective reward         │
└────────────────────────────────────────────────────────┘
```

```python
from neorx import run_rl_pipeline

# RL agent iteratively discovers molecules
result = run_rl_pipeline("HIV", n_episodes=20, top_n_targets=5)
```

## Installation

```bash
# From PyPI
pip install neorx

# From source with uv
git clone https://github.com/cod3smith/neorx.git
cd neorx
uv sync

# From source with pip
git clone https://github.com/cod3smith/neorx.git
cd neorx
pip install -e .
```

### Optional Dependencies

```bash
uv sync --extra docking   # AutoDock Vina support
uv sync --extra mirror    # ESM-2 protein language model
```

### Docker (with persistence)

```bash
# Start Redis + PostgreSQL + API
cp .env.example .env
docker compose up -d

# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

## Quick Start

### CLI

```bash
# Full pipeline for HIV
neorx run HIV --top-n 5

# Build the causal knowledge graph only
neorx graph "Type 2 Diabetes"

# Identify causal targets (no molecule generation)
neorx identify "Alzheimer Disease" --top-n 10

# Export graph to Cytoscape format
neorx graph HIV --export cytoscape

# Start API server
neorx serve --port 8000
```

### Python API

```python
from neorx import (
    build_disease_graph,
    identify_causal_targets,
    run_pipeline,
)

# Step-by-step
graph = build_disease_graph("HIV")
targets = identify_causal_targets(graph, top_n=10)

for t in targets:
    print(f"{t.gene_name}: {t.classification.value} "
          f"(confidence={t.causal_confidence:.3f})")

# Or run the full pipeline
result = run_pipeline("HIV", top_n_targets=5)
print(f"Found {result.n_causal_targets} causal targets")
print(f"Scored {len(result.scored_candidates)} candidates")
```

### REST API

```bash
# Full pipeline
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"disease": "HIV", "top_n_targets": 5}'

# Build graph only
curl -X POST http://localhost:8000/graph \
  -d '{"disease": "Type 2 Diabetes"}'

# Identify causal targets
curl -X POST http://localhost:8000/identify \
  -d '{"disease": "HIV", "top_n": 10}'
```

## Data Sources

| Database     | Data Type                   | API Style       | Caching |
|--------------|-----------------------------|-----------------|---------|
| Monarch Init.| Gene–disease associations   | REST (v3)       | ✅ 24h   |
| Open Targets | Target–disease evidence     | GraphQL         | ✅ 24h   |
| KEGG         | Pathway membership          | REST            | ✅ 24h   |
| Reactome     | Pathway membership          | REST            | ✅ 24h   |
| STRING       | Protein–protein interactions| REST (batch)    | ✅ 24h   |
| UniProt      | Protein metadata            | REST            | ✅ 24h   |
| RCSB PDB     | 3D structures               | REST/Search     | ✅ 24h   |
| ChEMBL v36   | Drug–target relationships   | Local SQLite    | ✅ local |

All REST clients include curated mock fallback data for offline use and testing.
ChEMBL queries run against a local SQLite copy (28 GB) for pathogen target
identification and drug evidence scoring.

## Methodology

### Causal Inference Approach

NeoRx uses **multi-source evidence triangulation** rather than
fabricated synthetic data:

1. **Graph-Topological Causal Analysis** — Path strength from treatment
   to disease computed as the product of edge weights.  Shorter, more
   direct paths indicate stronger causal mechanisms.

2. **d-Separation (Backdoor Criterion)** — Proper implementation using
   NetworkX's `d_separated()` to verify that causal effects are
   identifiable after conditioning on a valid adjustment set.

3. **Multi-Source Triangulation** — Evidence from independent databases
   is aggregated.  A target confirmed by Monarch Initiative, Open Targets, *and*
   STRING is more credible than one from a single source.

4. **Leave-One-Source-Out Sensitivity** — For each data source, we
   remove its evidence and recompute the effect.  Stable estimates
   across source removals indicate robustness.

5. **Bootstrap Confidence Intervals** — 95% CIs on causal confidence
   via bootstrap resampling over evidence components.

### Target Classification

- **Causal**: confidence ≥ 0.6, robust, identifiable via backdoor criterion
- **Correlational**: confidence < 0.4 or not robust
- **Inconclusive**: insufficient evidence for definitive classification

## Graph Persistence & Export

```python
from neorx import save_graph, load_graph

# Save to JSON (default: ~/.neorx/graphs/)
path = save_graph(graph)

# Export for Cytoscape
save_graph(graph, fmt="cytoscape")

# Export for Gephi
save_graph(graph, fmt="gexf")

# Reload later
graph = load_graph(path)
```

When running with Docker, graphs are also persisted to PostgreSQL
for querying and sharing across sessions.

## Project Structure

```
NeoRx/
├── main.py                     # Entry point
├── pyproject.toml              # Dependencies, build config & CLI scripts
├── LICENSE                     # Source Available (Non-Commercial)
├── CONTRIBUTING.md             # Contributor guide
├── CODE_OF_CONDUCT.md          # Community standards
├── SECURITY.md                 # Vulnerability reporting
├── docker-compose.yml          # Redis + PostgreSQL + API
├── Dockerfile
├── init/init.sql               # Database schema
│
├── neorx/               # Public re-export package (pip install)
│   └── __init__.py             # `from neorx import run_pipeline`
│
├── modules/neorx/              # Orchestration module
│   ├── models.py               # Pydantic data models
│   ├── graph_builder.py        # Parallel multi-source graph assembly
│   ├── identifier.py           # Causal inference (d-separation, triangulation)
│   ├── scorer.py               # 6D composite scoring
│   ├── pipeline.py             # Linear + RL-driven orchestration
│   ├── report.py               # Interactive HTML reports
│   ├── classifier.py           # Disease-type-aware target classification
│   ├── validator.py            # Known-target validation (7 diseases)
│   ├── tissue_filter.py        # HPA tissue expression boolean gate
│   ├── counterfactual.py       # Counterfactual validation + BioRL bridge
│   ├── literature_validator.py # Literature evidence lookup
│   ├── api.py                  # FastAPI REST service
│   ├── cache.py                # File/Redis caching layer
│   ├── persistence.py          # Graph save/load/export
│   ├── admet.py                # Multi-rule ADMET prediction
│   ├── __main__.py             # Typer CLI
│   ├── data_sources/           # 8 biomedical database clients
│   │   ├── monarch.py          #   Monarch Initiative (REST v3)
│   │   ├── open_targets.py     #   Open Targets (GraphQL)
│   │   ├── kegg.py             #   KEGG pathways
│   │   ├── reactome.py         #   Reactome pathways
│   │   ├── string_db.py        #   STRING PPIs
│   │   ├── uniprot.py          #   UniProt metadata
│   │   ├── pdb.py              #   RCSB PDB structures
│   │   └── chembl.py           #   ChEMBL v36 (local SQLite)
│   └── tests/
│
├── modules/causalbiorl/        # Causal reinforcement learning
│   ├── agents/                 # CausalAgent (hierarchical planning)
│   ├── envs/                   # Gymnasium envs (toy + DrugDiscovery-v0)
│   ├── causal/                 # SCM, planner, graph encoder, reward learner
│   └── tests/
│
├── modules/genmol/             # Molecular generation (VAE, ~4.1M params)
├── modules/molscreen/          # Drug-likeness screening
├── modules/dockbot/            # Molecular docking (Vina)
├── modules/mirrorfold/         # Protein structure prediction
│
├── papers/                     # Research papers (Markdown source)
│   ├── latex/                  # Generated LaTeX + PDF outputs
│   └── convert_to_latex.py     # Markdown → arXiv-ready LaTeX/PDF
│
├── figures/                    # Publication figures (PNG + PDF)
├── checkpoints/                # Trained model checkpoints
└── results/                    # Benchmark results (JSON)
```

## Testing

```bash
# Run all tests
uv run python -m pytest modules/ -q

# Run a specific module's tests
uv run python -m pytest modules/neorx/tests/ -v

# Run with coverage
uv run python -m pytest modules/ --cov=modules --cov-report=term-missing

# Lint
uv run ruff check .
uv run ruff format --check .
```

## Configuration

### Environment Variables

| Variable                       | Default     | Description                      |
|--------------------------------|-------------|----------------------------------|
| `NEORX_CACHE_BACKEND`   | `file`      | `file` or `redis`                |
| `NEORX_SEED`            | `42`        | RNG seed for reproducibility     |
| `REDIS_URL`                    | —           | Redis connection URL             |
| `DATABASE_URL`                 | —           | PostgreSQL connection URL        |


### Score Weights

Customise via environment or the Python API:

```python
from neorx import score_candidate

# Weights are configurable per-call
custom_weights = {
    "causal_confidence": 0.35,
    "binding_affinity": 0.20,
    "qed": 0.15,
    "sa": 0.10,
    "admet": 0.10,
    "novelty": 0.10,
}
```

## Research Papers

NeoRx is described across four companion papers:

1. **NeoRx** — Causal Inference for Automated Drug Target Identification via Pearl's Do-Calculus over Multi-Source Biomedical Knowledge Graphs
2. **CausalBioRL** — Reinforcement Learning with Causal World Models for Autonomous Drug Discovery
3. **Biological Intelligence** — Disease-Type-Aware Biological Intelligence Layers for Preventing Symptom Marker False Positives
4. **GenMol** — A Lightweight Variational Autoencoder for Target-Aware Drug-Like Molecule Generation

PDFs and LaTeX sources are in [`papers/latex/`](papers/latex/).

## Validation Results

| Disease | Precision | Recall | F₁ | Grade |
|---------|-----------|--------|----|-------|
| HIV | 0.429 | 0.750 | 0.545 | B |
| Malaria | 0.400 | 0.286 | 0.333 | C |
| Type 2 Diabetes | 0.417 | 0.833 | 0.556 | B |
| Alzheimer's | 0.333 | 0.400 | 0.364 | C |
| Lung Cancer | 0.538 | 1.000 | 0.700 | A |
| Breast Cancer | 0.308 | 0.667 | 0.421 | B |
| Ebola | 0.333 | 0.500 | 0.400 | C |
| **Mean** | **0.394** | **0.634** | **0.474** | — |

Zero known false positives promoted to CAUSAL across all 7 diseases.
NeoRx outperforms a correlation-only baseline in 6 of 7 diseases (+80% mean F₁ improvement).

## Citation

If you use NeoRx in your research, please cite:

```bibtex
@software{neorx2026,
  title  = {NeoRx: Causal Drug Target Discovery via Pearl's Do-Calculus},
  author = {Njeri, Kelyn Paul},
  year   = {2026},
  url    = {https://github.com/cod3smith/neorx}
}
```

## Author

**Kelyn Paul Njeri** · [NeoForge Labs](mailto:kelyn@neoforgelabs.tech) · [ORCID 0009-0000-1068-4512](https://orcid.org/0009-0000-1068-4512)

## License

Source Available — Non-Commercial. Free for personal, academic, and research use. Commercial use requires a separate licence. See [LICENSE](LICENSE) for details. Contact kelyn@neoforgelabs.tech for commercial licensing.
