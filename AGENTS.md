# AGENTS.md - LLM Eval

This file provides instructions for AI coding agents working in this repository.

## Project Overview

Python CLI tool for evaluating and comparing LLM outputs using pairwise comparisons, configurable rubrics, and Elo rating leaderboards. Built with Typer (CLI), SQLModel (database), LiteLLM (multi-provider LLM access), and PostgreSQL.

Package source: `src/llm_eval/` with subpackages: `cli`, `config`, `db`, `eval`, `models`.

## Build, Lint, and Test Commands

```bash
# Install (editable/dev mode)
pip install -e ".[dev]"

# Lint
ruff check src/ tests/
ruff check src/ tests/ --fix   # auto-fix

# Format
black src/ tests/

# Type-check (no mypy configured yet - use pyright if needed)

# Run all tests
pytest

# Run a single test file
pytest tests/test_file.py

# Run a single test
pytest tests/test_file.py::test_function_name

# Run tests with coverage
pytest --cov=src/llm_eval --cov-report=term-missing

# Database migrations
alembic revision --autogenerate -m "description"
alembic upgrade head

# CLI entry point (installed as `eval` command)
eval --help
```

## Architecture

- **`src/llm_eval/cli/main.py`**: Typer CLI with subcommands: `config`, `task`, `experiment`, `run`, `results`
- **`src/llm_eval/db/`**: SQLModel models (`models.py`) and session management (`session.py`)
- **`src/llm_eval/eval/`**: Core evaluation logic - `runner.py` (orchestrator), `comparator.py` (pairwise), `judge.py` (LLM-as-judge), `ranking.py` (Elo)
- **`src/llm_eval/models/provider.py`**: LiteLLM provider abstraction via `ModelProvider` and `ProviderFactory`
- **`src/llm_eval/config/`**: YAML config loading for models and rubrics
- **`prompts/`**: Jinja2 templates for judge prompts (`.j2` files)
- **`config/`**: Default YAML configs (models, rubrics, settings)
- **`alembic/`**: Database migration scripts

## Code Style Guidelines

### Imports

Group imports in order: stdlib, third-party, local. Use absolute imports for project modules.

```python
import json
from pathlib import Path
from typing import Optional

from sqlmodel import select
from rich.console import Console

from llm_eval.db.models import Experiment, Result
from llm_eval.eval.comparator import Comparator
```

### Types and Annotations

- Use full type hints on all function signatures (parameters and return types)
- Use `Optional[X]` for nullable values (not `X | None`)
- Use built-in generics: `list[X]`, `dict[K, V]`, `set[X]` (Python 3.10+ style)
- Use `TYPE_CHECKING` guard for import-only type hints to avoid circular imports
- Prefer frozen `@dataclass(frozen=True)` for immutable value objects

### Naming Conventions

- `snake_case` for functions, methods, variables, module-level constants
- `PascalCase` for classes and type aliases
- `UPPER_SNAKE_CASE` for module-level constants
- Prefix private methods with `_` (e.g., `_get_experiment`)
- Exception classes end with `Error` (e.g., `JudgeError`, `RunnerError`)

### Docstrings

Use Google-style docstrings with `Args`, `Returns`, and `Raises` sections:

```python
def evaluate_pairwise(self, response_a: str, response_b: str) -> JudgeResult:
    """Evaluate two responses in a pairwise comparison.

    Args:
        response_a: First response to evaluate.
        response_b: Second response to evaluate.

    Returns:
        JudgeResult with winner, justification, and optional reasoning.

    Raises:
        JudgeError: If evaluation fails.
    """
```

### Section Separators

Use banner comments to separate logical sections within files:

```python
# =============================================================================
# Section Name
# =============================================================================
```

### Error Handling

- Define custom exception hierarchies with a base class per module (e.g., `RunnerError` -> `ExperimentNotFoundError`)
- Fail fast with guard clauses and early validation at function entry
- Raise specific exceptions, not generic `Exception`
- Re-raise with `from e` to preserve tracebacks
- Map external library errors to domain exceptions in `_handle_error` style methods

### Database Models (SQLModel)

- Define `__tablename__` explicitly on each model class
- Use `Field()` with `description=` for all columns
- Use `sa_type=JSON` for dict/list columns
- Use `Relationship(back_populates=...)` for bidirectional relationships
- Define `__all__` in models module for clean re-exports
- Use UUID primary keys with `default_factory=uuid.uuid4`

### Factory Pattern

Use `create_*` factory functions for constructing complex objects:

```python
def create_comparator(judge_model: ModelConfig, rubric: Optional[Rubric] = None) -> Comparator:
    ...
```

### Configuration

- Environment variables use `DB_` prefix for database settings (via `pydantic-settings`)
- YAML files for model configs and rubrics
- Resolve API keys from env vars at config boundaries, not deep in business logic

### CLI Patterns

- Use Typer with `typer.Option(..., help="...")` for all CLI parameters
- Validate inputs early, raise `typer.BadParameter` for user errors
- Use Rich `Table` and `Panel` for formatted output
- Wrap command bodies in try/except with `raise typer.Exit(code=1)` on failure

### Testing

- Test files in `tests/` directory, named `test_*.py`
- Use pytest fixtures for database sessions and common test data
- No existing test framework patterns established yet - follow pytest conventions
