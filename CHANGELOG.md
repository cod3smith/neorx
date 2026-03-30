# Changelog

All notable changes to NeoRx will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] — 2026-03-29

### Added

- **CausalBioRL integration** — RL agent now drives the full drug
  discovery pipeline through `DrugDiscoveryEnv` (Gymnasium).
- **`run_rl_pipeline()`** — RL-driven alternative to the linear
  `run_pipeline()`. Agent iteratively selects targets and generates
  molecules via latent-space navigation.
- **R-GCN graph encoder** — `DiseaseGraphEncoder` maps disease knowledge
  graphs to fixed 128-D embeddings using relational graph convolution.
- **Surrogate docking model** — `SurrogateDockingModel` (MLP) provides
  ~1ms binding affinity predictions, trained on DockBot observations.
- **Adaptive reward learner** — `AdaptiveRewardLearner` with 6 per-objective
  critics and difficulty-adaptive weighting (hindsight shaping).
- **Hierarchical planner** — `HierarchicalPlanner` with UCB1 target
  selection (Level 1) and CEM molecule generation (Level 2).
- **Typed edges in SCM** — Edges tagged with provenance (`api` vs
  `learned`), `augment_graph()` for merging discovered edges,
  `from_disease_graph()` classmethod.
- **CounterfactualValidator bridge** — `validate_with_biorl_scm()` method
  delegates to the shared SCM when CausalBioRL is available.
- **DrugDiscovery-v0** environment registered in Gymnasium.

### Fixed

- **SCM self-loop bug** — autoregressive dependencies (`s0→s0`) were
  being stripped, preventing linear mechanisms from seeing state input.
- **Matplotlib `tostring_rgb` deprecation** — all 3 toy env renderers
  now use `buffer_rgba()` (works on macOS and headless Linux).
- **Pydantic v2 deprecation** — `class Config` replaced with
  `model_config = ConfigDict(...)` in `EpisodeResult` and
  `BenchmarkResult`.
- **Docstring escape sequence** — invalid `\s` in planner docstring.
- **SCM test flakiness** — increased learning rate for reliable
  convergence in unit tests.

### Infrastructure

- **GitHub Actions CI** — lint (Ruff), test (pytest on Ubuntu + macOS),
  type check (mypy), coverage upload (Codecov).
- **CONTRIBUTING.md** — contributor guide with style, testing, and PR
  conventions.
- **CODE_OF_CONDUCT.md** — Contributor Covenant v2.1.
- **SECURITY.md** — vulnerability reporting policy.
- **Ruff configuration** — formatter + linter in `pyproject.toml`.
- **pytest-cov integration** — coverage config with source filtering.
- **Expanded .gitignore** — reports/, results/, *.pdb, *.pdbqt, model
  weights, IDE files, OS files.

## [0.1.0] — 2026-03-29

### Added

- **Causal inference engine** — multi-source evidence triangulation via
  path strength × d-separation quality × centrality × source corroboration.
- **Backdoor criterion** — proper d-separation analysis using
  `networkx.d_separated()` for identifiability testing.
- **Bootstrap confidence intervals** — 95% CIs on causal confidence
  (200 resamples).
- **Leave-one-source-out sensitivity analysis** — robustness validation
  by systematically removing each data source.
- **7 biomedical data source clients** — DisGeNET, Open Targets (GraphQL),
  KEGG, Reactome, STRING, UniProt, RCSB PDB — all with curated mock
  fallbacks for offline use.
- **Parallel API queries** — `ThreadPoolExecutor` for concurrent database
  queries (~2× speedup on graph building).
- **File/Redis caching layer** — 24h TTL for API responses, 7d for graphs.
  Configurable via `NEORX_CACHE_BACKEND` env var.
- **Multi-rule ADMET predictor** — 8 rule systems: Lipinski RO5, Veber,
  Ghose, Egan egg, PAINS (RDKit FilterCatalog), BBB permeability, hERG
  liability, reactive group alerts.  Weighted composite score.
- **Graph persistence & export** — JSON, GraphML, GEXF, Cytoscape formats.
  PostgreSQL persistence for Docker deployments.
- **Disease ontology resolution** — `resolve_disease_id()` maps free-text
  disease names to EFO/MONDO identifiers via Open Targets search.
- **Configurable scorer weights** — override via Python parameter, env var
  (`NEORX_WEIGHTS`), or defaults.  Auto-normalised to sum=1.0.
- **SMILES canonicalization** — duplicate elimination via RDKit canonical
  SMILES before screening.
- **Interactive HTML reports** — vis.js causal knowledge graph, 95% CI
  column, UTC timestamps, updated methodology section.
- **Non-blocking FastAPI endpoints** — `asyncio.to_thread()` on all
  CPU/IO-bound handlers.
- **Rich CLI** — progress bar, `--seed` for reproducibility, `--export`
  for graph formats, `--log-file`, `--no-cache`.
- **Docker Compose infrastructure** — Redis 7 (AOF + LRU), PostgreSQL 16
  (5-table schema with indexes), API container with healthchecks.
- **PEP 561 `py.typed` marker** — full static typing support.
- **Clean public API** — `from neorx import run_pipeline` works
  after `pip install neorx`.
- **108 tests** covering data sources, graph builder, identifier, pipeline,
  scorer, cache, persistence, ADMET, configurable weights, SMILES
  canonicalization, confidence intervals, and disease ID resolution.

[Unreleased]: https://github.com/NeoForge/NeoRx/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/NeoForge/NeoRx/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NeoForge/NeoRx/releases/tag/v0.1.0
