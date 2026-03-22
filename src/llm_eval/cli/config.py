"""Config management commands."""

import json
from typing import Optional

import typer

from llm_eval.cli._common import console, print_error, print_info, print_success
from llm_eval.db.models import ModelConfig, Rubric
from llm_eval.db.session import get_session, init_db


def init_db_command() -> None:
    """Initialize database tables."""
    try:
        init_db()
        print_success("Database initialized successfully!")
    except Exception as e:
        print_error(f"Failed to initialize database: {e}")
        raise typer.Exit(code=1)


def add_model_command(
    name: str = typer.Option(..., help="Unique model name (e.g., gpt-4o, lmstudio-local)"),
    provider: str = typer.Option(..., help="Provider name (openai, anthropic, ollama, etc.)"),
    model: str = typer.Option(..., help="Model name passed to API"),
    base_url: Optional[str] = typer.Option(
        None, help="Base URL for API endpoints (for local models)"
    ),
    api_key: Optional[str] = typer.Option(None, help="API key or env var reference"),
) -> None:
    """Add a model configuration to the database."""
    if not provider:
        print_error("Provider is required")
        raise typer.BadParameter("Provider cannot be empty")

    if not model:
        print_error("Model is required")
        raise typer.BadParameter("Model cannot be empty")

    try:
        with get_session() as session:
            existing = session.get(ModelConfig, name)
            if existing:
                print_error(f"Model '{name}' already exists")
                raise typer.BadParameter(f"Model '{name}' already exists")

            model_config = ModelConfig(
                name=name,
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
            )
            session.add(model_config)
            session.flush()

        print_success(f"Model '{name}' added successfully!")
    except typer.BadParameter:
        raise
    except Exception as e:
        print_error(f"Failed to add model: {e}")
        raise typer.Exit(code=1)


def add_rubric_command(
    name: str = typer.Option(..., help="Rubric name (e.g., quality, helpfulness)"),
    weights_json: str = typer.Option(
        ...,
        "--weights-json",
        help='JSON string of criterion weights (e.g., \'{"accuracy": 0.5, "clarity": 0.5}\')',
    ),
) -> None:
    """Add a rubric to the database."""
    try:
        weights = json.loads(weights_json)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON format: {e}")
        raise typer.BadParameter("Invalid JSON format for weights")

    if not isinstance(weights, dict):
        print_error("Weights must be a JSON object")
        raise typer.BadParameter("Weights must be a JSON object")

    try:
        with get_session() as session:
            existing = session.get(Rubric, name)
            if existing:
                print_error(f"Rubric '{name}' already exists")
                raise typer.BadParameter(f"Rubric '{name}' already exists")

            rubric = Rubric(
                name=name,
                description=f"Rubric with {len(weights)} criteria",
                weights=weights,
            )
            session.add(rubric)
            session.flush()

        print_success(f"Rubric '{name}' added successfully!")
    except typer.BadParameter:
        raise
    except Exception as e:
        print_error(f"Failed to add rubric: {e}")
        raise typer.Exit(code=1)


def list_models_command() -> None:
    """List all model configurations."""
    from rich.table import Table
    from sqlmodel import select

    try:
        with get_session() as session:
            models = list(session.exec(select(ModelConfig).order_by(ModelConfig.name)).all())

        if not models:
            print_info("No models configured. Use 'llm-eval config add-model' to add one.")
            return

        table = Table(title="Configured Models", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="green")
        table.add_column("Provider", style="yellow")
        table.add_column("Model", style="blue")
        table.add_column("Base URL", style="magenta")
        table.add_column("Active", style="white")

        for m in models:
            table.add_row(
                m.name,
                m.provider,
                m.model,
                m.base_url or "-",
                "✓" if m.is_active else "✗",
            )

        console.print(table)
    except Exception as e:
        print_error(f"Failed to list models: {e}")
        raise typer.Exit(code=1)


def list_rubrics_command() -> None:
    """List all rubrics."""
    from rich.table import Table
    from sqlmodel import select

    try:
        with get_session() as session:
            rubrics = list(session.exec(select(Rubric).order_by(Rubric.name)).all())

        if not rubrics:
            print_info("No rubrics configured. Use 'llm-eval config add-rubric' to add one.")
            return

        table = Table(title="Configured Rubrics", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="green")
        table.add_column("Description", style="yellow")
        table.add_column("Criteria", style="blue")
        table.add_column("Created", style="magenta")

        for r in rubrics:
            criteria_count = len(r.weights) if r.weights else 0
            created_str = r.created_at.strftime("%Y-%m-%d") if r.created_at else "-"
            table.add_row(
                r.name,
                r.description or "-",
                str(criteria_count),
                created_str,
            )

        console.print(table)
    except Exception as e:
        print_error(f"Failed to list rubrics: {e}")
        raise typer.Exit(code=1)
