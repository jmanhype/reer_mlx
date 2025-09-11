# Repository Guidelines

## Project Structure & Module Organization
- Source: `core/` (pipeline, training, scoring), `plugins/` (providers/integrations), `dspy_program/` (DSPy programs), `social/` (platform logic), `schemas/` (validation), `tools/` (utilities: memory profiler, schema checks), `scripts/` (aux CLI), `templates/`, `config/`.
- Tests: `tests/unit/`, `tests/integration/`, `tests/e2e/`, plus `tests/contract/` fixtures. Example: `tests/unit/test_candidate_scorer.py`.
- Assets/data: `data/`, `test_data/`, `examples/`, `docs/`.

## Build, Test, and Development Commands
- Install: `make install` (prod) or `make install-dev` (dev + extras); full setup: `make setup` (dev install + pre-commit hooks).
- Lint/Type/Format: `make lint` (ruff + mypy), `make format` (black + ruff format + autofix).
- Tests: `make test` (all), or targeted: `make test-unit`, `make test-integration`, `make test-e2e`.
- Coverage: `make coverage` (terminal + HTML). Filter slow tests: `pytest -m 'not slow'`.
- Build package: `make build`. Clean artifacts: `make clean`.

## Coding Style & Naming Conventions
- Python 3.11+. Use type hints everywhere; MyPy runs in strict mode.
- Formatting: Black (line length 88). Lint: Ruff (pycodestyle/pyflakes/isort/bugbear/etc.). Import order managed by Ruff isort.
- Naming: modules/files `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`; constants `UPPER_SNAKE_CASE`.
- Keep functions focused; prefer pure, testable units in `core/`. Public APIs live under package `__init__.py` as needed.

## Testing Guidelines
- Framework: Pytest with markers `unit`, `integration`, `e2e`, `slow` (see `pytest.ini`).
- Locations/patterns: place tests alongside the matching area (unit vs. integration). Names: `test_*.py`, classes `Test*`, functions `test_*`.
- Run examples: `pytest tests/unit -q`; with coverage: `coverage run -m pytest tests && coverage report`.
- Provide fixtures in `tests/conftest.py`; prefer async-friendly tests where applicable.

## Commit & Pull Request Guidelines
- Commits: Prefer Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`). Be concise and scoped.
- PRs: include a clear description, linked issue, test coverage for changes, and any docs updates (`docs/`), plus logs/output if behavior changes.
- Run `make check` locally before opening PRs; ensure pre-commit passes: `make pre-commit`.

## Security & Configuration Tips
- Secrets: never commit real credentials. Copy `.env.example` to `.env` for local runs.
- Config/schema: validate with `python tools/schema_check.py`; place configuration in `config/` and JSON/YAML under `schemas/`.
