# Contributing to NeoRx

Thank you for your interest in contributing to NeoRx! This guide
will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/NeoForge/NeoRx.git
cd NeoRx

# Create virtual environment and install dependencies
uv sync

# Install development dependencies
uv sync --group dev

# Verify everything works
uv run python -m pytest modules/ -q
```

## Code Quality

### Style

- **Formatter**: [Ruff](https://docs.astral.sh/ruff/) (line length 100)
- **Linter**: Ruff (selected rules: E, F, W, I, UP, B, SIM, TCH)
- **Type checker**: [mypy](https://mypy-lang.org/) (strict mode on new code)

```bash
# Format
uv run ruff format .

# Lint
uv run ruff check . --fix

# Type check
uv run mypy modules/ --ignore-missing-imports
```

### Pre-commit (recommended)

```bash
uv pip install pre-commit
pre-commit install
```

## Testing

We use [pytest](https://docs.pytest.org/) for all tests.

```bash
# Run all tests
uv run python -m pytest modules/ -q

# Run a specific module's tests
uv run python -m pytest modules/neorx/tests/ -v

# Run with coverage
uv run python -m pytest modules/ --cov=modules --cov-report=term-missing
```

### Writing Tests

- Place tests in a `tests/` directory inside the relevant module.
- Name test files `test_<module>.py`.
- Use descriptive class and method names: `TestGraphBuilder.test_empty_disease_returns_empty_graph`.
- Never use mocks for logic that can be tested directly. Mock only
  external network calls.
- All new features must include tests.

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`.
2. **Write tests** for any new functionality.
3. **Run the full test suite** and ensure all tests pass.
4. **Run the linter** (`ruff check .`) and fix any issues.
5. **Update documentation** if you changed public APIs.
6. **Update CHANGELOG.md** under an `[Unreleased]` section.
7. **Open a PR** with a clear title and description.

### PR Title Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add MirrorFold stability reward to BioRL
fix: correct d-separation conditioning set
docs: update CLI examples in README
test: add coverage for surrogate docking model
refactor: extract graph encoding into separate module
```

## Architecture Overview

```
modules/
├── neorx/           # Orchestration: graph → targets → molecules → report
├── causalbiorl/     # RL agent + environments + causal reasoning
├── genmol/          # Molecular generation (VAE)
├── molscreen/       # Drug-likeness screening
├── dockbot/         # Molecular docking (AutoDock Vina)
└── mirrorfold/      # Protein structure prediction
```

### Key Design Principles

1. **Graceful degradation** — Every sub-module integration is wrapped
   in `try/except`. If DockBot fails, we score without binding affinity.
   The pipeline never crashes.

2. **No mock contamination** — Tests must not use mocks for internal
   logic. Only external HTTP calls may be mocked.

3. **Causal > Correlational** — The scorer weights causal confidence
   highest (0.30). This is a deliberate design choice, not an accident.

4. **Typed edges** — The SCM distinguishes API-derived edges from
   learned edges. Provenance matters.

## Reporting Bugs

Open an issue with:
- Python version (`python --version`)
- OS and architecture
- Full traceback
- Minimal reproduction steps

## Requesting Features

Open an issue with the `enhancement` label. Describe the use case,
not just the solution.

## License

By contributing, you agree that your contributions will be licensed
under the MIT License.
