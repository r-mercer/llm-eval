# AGENTS.md — llm-eval

This file gives pragmatic rules and commands for autonomous coding agents working in this repository.
Keep changes conservative, explain assumptions in PRs or commit messages, and prefer small, testable edits.

## Quick Project Overview

Python CLI for evaluating LLM outputs with pairwise comparisons, configurable rubrics, and Elo-style ranking. Key packages: Typer (CLI), SQLModel (ORM), LiteLLM (providers), Alembic (migrations). Source root: `src/llm_eval/`.

## Build, Lint, and Test Commands

Use the venv in this repo or create a fresh one. Recommended Python: 3.10+ (project uses 3.12 locally in the dev environment).

Install (editable/dev):

```bash
pip install -e '.[dev]'
```

Lint & format:

```bash
ruff check src/ tests/            # lint
ruff check src/ tests/ --fix      # auto-fix lint issues when safe
black src/ tests/                 # format code
pyright                           # type-check (pyrightconfig.json provided)
```

Tests:

```bash
pytest                             # run all tests
pytest -q                          # quieter output
pytest tests/test_file.py          # run a single test file
pytest tests/test_file.py::test_fn # run a single test
pytest -k 'substring'              # run tests matching expression
pytest --maxfail=1 --disable-warnings

# Coverage
pytest --cov=src/llm_eval --cov-report=term-missing
```

Database migrations:

```bash
alembic revision --autogenerate -m "msg"
alembic upgrade head
```

CLI entrypoint (when package installed): `eval --help`

## High‑level Architecture

- CLI: `src/llm_eval/cli/main.py` (Typer)
- DB: `src/llm_eval/db/` (SQLModel models + session)
- Eval core: `src/llm_eval/eval/` (runner, comparator, judge, ranking)
- Provider abstraction: `src/llm_eval/models/provider.py`
- Configs: `config/*.yaml` and `src/llm_eval/config/*`
- Prompts: `prompts/*.j2`
- Migrations: `alembic/`

## Code Style Guidelines (for agents)

- Imports: group and order — standard library, third-party, local. Use absolute imports for project modules.

  Example:

  ```python
  import json
  from pathlib import Path

  from sqlmodel import select
  from rich.console import Console

  from llm_eval.db.models import Experiment
  from llm_eval.eval.comparator import Comparator
  ```

- Types & annotations:
  - Provide full type signatures for functions and methods (params and return types).
  - Use `Optional[X]` (not `X | None`) for nullable types to keep compatibility with TYPE_CHECKING guards.
  - Prefer built-in generics: `list[str]`, `dict[str, Any]`.
  - Use `from typing import TYPE_CHECKING` to avoid circular import runtime costs.

- Naming:
  - `snake_case` for functions, methods, variables and module-level constants
  - `PascalCase` for classes and dataclasses
  - `UPPER_SNAKE_CASE` for constants
  - Private helpers prefixed with `_`
  - Exception classes end with `Error` (e.g., `JudgeError`)

- Docstrings:
  - Use Google-style docstrings with `Args`, `Returns`, and `Raises` sections.

- Formatting & structure:
  - Keep files focused; if a file exceeds ~400 lines, consider splitting it.
  - Use banner comments to separate logical sections when helpful:

    ```python
    # =============================================================================
    # Section Name
    # =============================================================================
    ```

- Error handling:
  - Define domain-specific exception hierarchies (module base -> specific errors).
  - Fail fast with input validation and raise descriptive exceptions.
  - Re-raise exceptions with `from e` to preserve tracebacks.
  - Wrap external library errors and map them to domain exceptions in helper methods (e.g., `_handle_error`).

- Database models (SQLModel):
  - Specify `__tablename__` on models.
  - Use `Field(..., description=...)` for columns.
  - For JSON-like columns use `sa_type=JSON`.
  - Use relationships with `back_populates` for bidirectional relationships.
  - Prefer UUID primary keys with `default_factory=uuid.uuid4` where appropriate.

- Factories & constructors:
  - Use `create_*` factory functions to assemble complex objects and hide wiring logic.

- Configuration & secrets:
  - Keep API keys and secrets in environment variables. Resolve them at config boundaries, not deep inside business logic.
  - DB environment variables should use `DB_` prefix.

- CLI patterns:
  - Use Typer `typer.Option(..., help="...")` for flags and arguments.
  - Validate user input early and raise `typer.BadParameter` for invalid values.
  - Use Rich for formatted output (tables, panels) to improve CLI UX.

## Testing Guidelines

- Tests live in `tests/` and use pytest. Name test modules `test_*.py` and test functions `test_*`.
- Use fixtures for DB/session setup and teardown. Keep tests deterministic and fast; mock external LLM/network calls where possible.
- When adding tests, run the matching subset locally (`pytest tests/test_x.py::test_y -q`).

## Git & Commit Rules for Agents

- NEVER run destructive git commands like `git reset --hard` or `git checkout --` unless explicitly requested.
- Do not amend commits unless the user explicitly requests it and the safety rules in repo tooling allow it.
- Do not commit secrets (.env, credentials). If discovered, notify the user and redact before committing.
- Only create commits when explicitly asked. When committing, use clear messages describing the "why" (1–2 sentence summary).

## Agent Behaviour & Safety

- Default to making minimal, well-tested changes; include tests when behavior changes.
- If a requested change is ambiguous and would materially alter behavior, make a safe default and document it in the PR/commit message; ask one targeted question if necessary.
- If you must ask a question, do all non-blocked work first and then ask exactly one targeted question.

## Local Rules / Tooling Overrides

- Cursor/Copilot rules: no `.cursor` or `.cursorrules` directories were found in the repository. No `.github/copilot-instructions.md` present. If these files are added later, follow their directives and surface any conflicts to the user.

## Useful Paths

- `src/llm_eval/cli/main.py`
- `src/llm_eval/db/models.py`
- `src/llm_eval/db/session.py`
- `src/llm_eval/eval/runner.py`
- `src/llm_eval/eval/judge.py`
- `src/llm_eval/eval/comparator.py`
- `config/*.yaml`, `prompts/*.j2`, `alembic/`

## Next Steps Agents Should Offer

1. Run lint/format and fix trivial issues (ruff --fix, black) and report remaining lint failures.
2. Add or update tests for behavioral changes and run the targeted test file locally.
3. Open a branch and create a focused PR describing the change, tests added, and rationale.
