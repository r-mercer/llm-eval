"""Main CLI entry point for llm_eval."""

import json
import uuid
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
from sqlalchemy import func as sql_func
from sqlmodel import desc

# Import database components
from llm_eval.db.models import (
    Experiment,
    JudgeRun,
    ModelConfig,
    Result,
    Rubric,
    Task,
)
from llm_eval.db.session import get_session, init_db

# =============================================================================
# Typer App Setup with Rich
# =============================================================================

custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
    }
)

console = Console(theme=custom_theme)

app = typer.Typer(
    name="llm-eval",
    help="[bold]LLM Evaluation Framework CLI[/bold]",
    add_completion=False,
    rich_markup_mode="rich",
)

# =============================================================================
# Helper Functions
# =============================================================================


def print_error(message: str) -> None:
    """Print error message in red."""
    console.print(f"[error]Error:[/error] {message}")


def print_success(message: str) -> None:
    """Print success message in green."""
    console.print(f"[success]{message}[/success]")


def print_info(message: str) -> None:
    """Print info message in cyan."""
    console.print(f"[info]{message}[/info]")


def validate_uuid(value: str) -> uuid.UUID:
    """Parse and validate UUID string. Fail fast with clear error."""
    try:
        return uuid.UUID(value)
    except ValueError:
        raise typer.BadParameter(f"Invalid UUID format: {value}")


# =============================================================================
# Config Commands
# =============================================================================


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
    # Validate required provider
    if not provider:
        print_error("Provider is required")
        raise typer.BadParameter("Provider cannot be empty")

    if not model:
        print_error("Model is required")
        raise typer.BadParameter("Model cannot be empty")

    try:
        with get_session() as session:
            # Check if model with same name exists - early exit
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
    # Parse weights JSON at boundary - fail fast on invalid JSON
    try:
        weights = json.loads(weights_json)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON format: {e}")
        raise typer.BadParameter("Invalid JSON format for weights")

    # Validate weights is a dict
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
    try:
        with get_session() as session:
            models = session.query(ModelConfig).order_by(ModelConfig.name).all()

        # Early exit if no models
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
    try:
        with get_session() as session:
            rubrics = session.query(Rubric).order_by(Rubric.name).all()

        # Early exit if no rubrics
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


# =============================================================================
# Task Commands
# =============================================================================


def add_task_command(
    name: str = typer.Option(..., help="Task name"),
    input_text: str = typer.Option(..., help="The prompt or task input text"),
    expected_output: Optional[str] = typer.Option(None, help="Reference expected output"),
) -> None:
    """Add a task to the database."""
    if not name:
        print_error("Task name is required")
        raise typer.BadParameter("Task name cannot be empty")

    if not input_text:
        print_error("Input text is required")
        raise typer.BadParameter("Input text cannot be empty")

    try:
        with get_session() as session:
            task = Task(
                name=name,
                input_text=input_text,
                expected_output=expected_output,
            )
            session.add(task)
            session.flush()

        print_success(f"Task '{name}' added successfully!")
    except Exception as e:
        print_error(f"Failed to add task: {e}")
        raise typer.Exit(code=1)


def list_tasks_command() -> None:
    """List all tasks."""
    try:
        with get_session() as session:
            tasks = session.query(Task).order_by(desc(Task.created_at)).all()

        # Early exit if no tasks
        if not tasks:
            print_info("No tasks configured. Use 'llm-eval task add' to add one.")
            return

        table = Table(title="Evaluation Tasks", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="green")
        table.add_column("Input Preview", style="yellow")
        table.add_column("Has Expected Output", style="blue")
        table.add_column("Created", style="magenta")

        for t in tasks:
            input_preview = t.input_text[:50] + "..." if len(t.input_text) > 50 else t.input_text
            has_expected = "✓" if t.expected_output else "✗"
            created_str = t.created_at.strftime("%Y-%m-%d") if t.created_at else "-"
            table.add_row(
                t.name,
                input_preview,
                has_expected,
                created_str,
            )

        console.print(table)
    except Exception as e:
        print_error(f"Failed to list tasks: {e}")
        raise typer.Exit(code=1)


def import_tasks_command(
    file_path: str = typer.Option(..., help="Path to JSON file with tasks"),
) -> None:
    """Import tasks from a JSON file."""
    # Validate file exists - fail fast
    path = Path(file_path)
    if not path.exists():
        print_error(f"File not found: {file_path}")
        raise typer.BadParameter(f"File not found: {file_path}")

    # Parse JSON at boundary
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        raise typer.BadParameter(f"Invalid JSON in file: {e}")
    except Exception as e:
        print_error(f"Failed to read file: {e}")
        raise typer.BadParameter(f"Failed to read file: {e}")

    # Validate data structure
    if not isinstance(data, list):
        print_error("JSON file must contain a list of tasks")
        raise typer.BadParameter("JSON file must contain a list of tasks")

    # Validate each task has required fields
    for i, task_data in enumerate(data):
        if not isinstance(task_data, dict):
            print_error(f"Task at index {i} is not an object")
            raise typer.BadParameter(f"Task at index {i} is not an object")
        if "name" not in task_data or "input_text" not in task_data:
            print_error(f"Task at index {i} missing required fields (name, input_text)")
            raise typer.BadParameter(
                f"Task at index {i} missing required fields (name, input_text)"
            )

    # Import tasks
    imported_count = 0
    try:
        with get_session() as session:
            for task_data in data:
                task = Task(
                    name=task_data["name"],
                    input_text=task_data["input_text"],
                    expected_output=task_data.get("expected_output"),
                    task_metadata=task_data.get("metadata"),
                )
                session.add(task)
                imported_count += 1
            session.flush()

        print_success(f"Imported {imported_count} tasks successfully!")
    except Exception as e:
        print_error(f"Failed to import tasks: {e}")
        raise typer.Exit(code=1)


# =============================================================================
# Experiment Commands
# =============================================================================


def create_experiment_command(
    name: str = typer.Option(..., help="Experiment name"),
    description: Optional[str] = typer.Option(None, help="Experiment description"),
    judge_model: str = typer.Option(..., help="Judge model name"),
    rubric: str = typer.Option(..., help="Rubric name"),
    baseline_model: Optional[str] = typer.Option(None, help="Baseline model name for comparison"),
) -> None:
    """Create a new experiment."""
    if not name:
        print_error("Experiment name is required")
        raise typer.BadParameter("Experiment name cannot be empty")

    try:
        with get_session() as session:
            # Validate judge model exists
            judge = (
                session.query(ModelConfig)
                .filter(ModelConfig.is_active)
                .filter(ModelConfig.name == judge_model)
                .first()
            )
            if not judge:
                print_error(f"Judge model '{judge_model}' not found")
                raise typer.BadParameter(f"Judge model '{judge_model}' not found")

            # Validate rubric exists
            rubric_obj = session.query(Rubric).filter(Rubric.name == rubric).first()
            if not rubric_obj:
                print_error(f"Rubric '{rubric}' not found")
                raise typer.BadParameter(f"Rubric '{rubric}' not found")

            # Validate baseline model if provided
            baseline = None
            if baseline_model:
                baseline = (
                    session.query(ModelConfig)
                    .filter(ModelConfig.is_active)
                    .filter(ModelConfig.name == baseline_model)
                    .first()
                )
                if not baseline:
                    print_error(f"Baseline model '{baseline_model}' not found")
                    raise typer.BadParameter(f"Baseline model '{baseline_model}' not found")

            experiment = Experiment(
                name=name,
                description=description,
                status="pending",
                judge_model_id=judge.id,
                rubric_id=rubric_obj.id,
                baseline_model_id=baseline.id if baseline else None,
                config_snapshot={
                    "judge_model": judge_model,
                    "rubric": rubric,
                    "baseline_model": baseline_model,
                },
            )
            session.add(experiment)
            session.flush()

        print_success(f"Experiment '{name}' created successfully!")
    except typer.BadParameter:
        raise
    except Exception as e:
        print_error(f"Failed to create experiment: {e}")
        raise typer.Exit(code=1)


def list_experiments_command() -> None:
    """List all experiments."""
    try:
        with get_session() as session:
            experiments = session.query(Experiment).order_by(desc(Experiment.created_at)).all()

        # Early exit if no experiments
        if not experiments:
            print_info("No experiments found. Use 'llm-eval experiment create' to create one.")
            return

        table = Table(title="Experiments", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Judge Model", style="blue")
        table.add_column("Rubric", style="magenta")
        table.add_column("Created", style="white")

        for e in experiments:
            judge_name = e.judge_model.name if e.judge_model else "-"
            rubric_name = e.rubric.name if e.rubric else "-"
            created_str = e.created_at.strftime("%Y-%m-%d") if e.created_at else "-"

            status_color = {
                "pending": "yellow",
                "running": "cyan",
                "completed": "green",
                "failed": "red",
            }.get(e.status, "white")

            table.add_row(
                e.name,
                f"[{status_color}]{e.status}[/{status_color}]",
                judge_name,
                rubric_name,
                created_str,
            )

        console.print(table)
    except Exception as e:
        print_error(f"Failed to list experiments: {e}")
        raise typer.Exit(code=1)


def experiment_status_command(
    experiment_id: str = typer.Option(..., help="Experiment ID"),
) -> None:
    """Show the status of an experiment."""
    exp_id = validate_uuid(experiment_id)

    try:
        with get_session() as session:
            experiment = session.get(Experiment, exp_id)
            if not experiment:
                print_error(f"Experiment '{experiment_id}' not found")
                raise typer.BadParameter("Experiment not found")

            # Get related data
            results_count = session.query(Result).filter(Result.experiment_id == exp_id).count()
            judge_runs_count = (
                session.query(JudgeRun).filter(JudgeRun.experiment_id == exp_id).count()
            )

        # Display status panel
        status_color = {
            "pending": "yellow",
            "running": "cyan",
            "completed": "green",
            "failed": "red",
        }.get(experiment.status, "white")

        content = f"""
[bold]Name:[/bold] {experiment.name}
[bold]Status:[/bold] [{status_color}]{experiment.status}[/{status_color}]
[bold]Description:[/bold] {experiment.description or "N/A"}
[bold]Judge Model:[/bold] {experiment.judge_model.name if experiment.judge_model else "N/A"}
[bold]Rubric:[/bold] {experiment.rubric.name if experiment.rubric else "N/A"}
[bold]Baseline Model:[/bold] {experiment.baseline_model.name if experiment.baseline_model else "N/A"}
[bold]Results:[/bold] {results_count}
[bold]Judge Runs:[/bold] {judge_runs_count}
[bold]Created:[/bold] {experiment.created_at.strftime("%Y-%m-%d %H:%M") if experiment.created_at else "N/A"}
[bold]Completed:[/bold] {experiment.completed_at.strftime("%Y-%m-%d %H:%M") if experiment.completed_at else "N/A"}
        """

        console.print(Panel(content.strip(), title="Experiment Status", border_style="cyan"))
    except typer.BadParameter:
        raise
    except Exception as e:
        print_error(f"Failed to get experiment status: {e}")
        raise typer.Exit(code=1)


# =============================================================================
# Run Commands
# =============================================================================


def run_experiment_command(
    experiment_id: str = typer.Option(..., help="Experiment ID to run"),
    models: Optional[str] = typer.Option(
        None,
        help="Comma-separated list of model names to evaluate (if not all)",
    ),
) -> None:
    """Run an experiment by ID."""
    exp_id = validate_uuid(experiment_id)

    try:
        with get_session() as session:
            experiment = session.get(Experiment, exp_id)
            if not experiment:
                print_error(f"Experiment '{experiment_id}' not found")
                raise typer.BadParameter("Experiment not found")

            # Check experiment status
            if experiment.status == "completed":
                print_error("Experiment already completed. Create a new experiment to run again.")
                raise typer.BadParameter("Experiment already completed")

            # Update status to running
            experiment.status = "running"
            session.flush()

        print_info(f"Running experiment '{experiment.name}'...")
        print_info("Note: Full execution not implemented - this is a placeholder")

        # TODO: Implement actual experiment execution
        # 1. Get all tasks
        # 2. For each task, run each model
        # 3. Run judge evaluations
        # 4. Calculate scores

        with get_session() as session:
            experiment = session.get(Experiment, exp_id)
            experiment.status = "completed"
            session.flush()

        print_success(f"Experiment '{experiment.name}' completed!")
    except typer.BadParameter:
        raise
    except Exception as e:
        # Mark experiment as failed on error
        try:
            with get_session() as session:
                experiment = session.get(Experiment, exp_id)
                if experiment:
                    experiment.status = "failed"
                    session.flush()
        except Exception:
            pass

        print_error(f"Failed to run experiment: {e}")
        raise typer.Exit(code=1)


def run_compare_command(
    model_a: str = typer.Option(..., help="First model name"),
    model_b: str = typer.Option(..., help="Second model name"),
    judge_model: str = typer.Option(..., help="Judge model name"),
    rubric: str = typer.Option(..., help="Rubric name"),
    tasks: Optional[str] = typer.Option(
        None,
        help="Comma-separated list of task names (if not all)",
    ),
) -> None:
    """Run comparison between two models."""
    try:
        with get_session() as session:
            # Validate model A exists
            model_config_a = session.query(ModelConfig).filter(ModelConfig.name == model_a).first()
            if not model_config_a:
                print_error(f"Model '{model_a}' not found")
                raise typer.BadParameter(f"Model '{model_a}' not found")

            # Validate model B exists
            model_config_b = session.query(ModelConfig).filter(ModelConfig.name == model_b).first()
            if not model_config_b:
                print_error(f"Model '{model_b}' not found")
                raise typer.BadParameter(f"Model '{model_b}' not found")

            # Validate judge model exists
            judge = session.query(ModelConfig).filter(ModelConfig.name == judge_model).first()
            if not judge:
                print_error(f"Judge model '{judge_model}' not found")
                raise typer.BadParameter(f"Judge model '{judge_model}' not found")

            # Validate rubric exists
            rubric_obj = session.query(Rubric).filter(Rubric.name == rubric).first()
            if not rubric_obj:
                print_error(f"Rubric '{rubric}' not found")
                raise typer.BadParameter(f"Rubric '{rubric}' not found")

            # Get tasks
            task_query = session.query(Task)
            if tasks:
                task_names = [t.strip() for t in tasks.split(",")]
                task_query = task_query.filter(sql_func.in_(Task.name, task_names))
            task_list = task_query.all()

            if not task_list:
                print_error("No tasks found to compare")
                raise typer.BadParameter("No tasks found to compare")

        print_info(f"Comparing '{model_a}' vs '{model_b}' on {len(task_list)} tasks...")
        print_info("Note: Full execution not implemented - this is a placeholder")

        # TODO: Implement actual comparison
        # 1. Run both models on all tasks
        # 2. Run judge to compare outputs
        # 3. Calculate win rate

        print_success("Comparison complete!")

        # Display comparison results table
        table = Table(title="Comparison Results", show_header=True, header_style="bold cyan")
        table.add_column("Model", style="green")
        table.add_column("Wins", style="yellow")
        table.add_column("Losses", style="red")
        table.add_column("Ties", style="blue")
        table.add_column("Win Rate", style="magenta")

        # Placeholder data
        table.add_row(model_a, "0", "0", "0", "0%")
        table.add_row(model_b, "0", "0", "0", "0%")

        console.print(table)
    except typer.BadParameter:
        raise
    except Exception as e:
        print_error(f"Failed to run comparison: {e}")
        raise typer.Exit(code=1)


# =============================================================================
# Results Commands
# =============================================================================


def show_results_command(
    experiment_id: str = typer.Option(..., help="Experiment ID"),
    limit: int = typer.Option(10, help="Number of results to show"),
) -> None:
    """Show results for an experiment."""
    exp_id = validate_uuid(experiment_id)

    # Validate limit
    if limit < 1:
        print_error("Limit must be at least 1")
        raise typer.BadParameter("Limit must be at least 1")

    try:
        with get_session() as session:
            experiment = session.get(Experiment, exp_id)
            if not experiment:
                print_error(f"Experiment '{experiment_id}' not found")
                raise typer.BadParameter("Experiment not found")

            results = (
                session.query(Result)
                .filter(Result.experiment_id == exp_id)
                .order_by(desc(Result.created_at))
                .limit(limit)
                .all()
            )

        # Early exit if no results
        if not results:
            print_info("No results found for this experiment.")
            return

        table = Table(
            title=f"Results for '{experiment.name}'", show_header=True, header_style="bold cyan"
        )
        table.add_column("Task", style="green")
        table.add_column("Model", style="yellow")
        table.add_column("Output Preview", style="blue")
        table.add_column("Score", style="magenta")
        table.add_column("Created", style="white")

        for r in results:
            task_name = r.task.name if r.task else "-"
            model_name = r.model.name if r.model else "-"
            output_preview = (
                r.output_text[:40] + "..." if len(r.output_text) > 40 else r.output_text
            )
            score_str = f"{r.score:.2f}" if r.score is not None else "-"
            created_str = r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "-"

            table.add_row(
                task_name,
                model_name,
                output_preview,
                score_str,
                created_str,
            )

        console.print(table)

        # Show judge runs if available
        with get_session() as session:
            judge_runs = (
                session.query(JudgeRun).filter(JudgeRun.experiment_id == exp_id).limit(limit).all()
            )

        if judge_runs:
            judge_table = Table(
                title="Judge Evaluations", show_header=True, header_style="bold cyan"
            )
            judge_table.add_column("Winner", style="green")
            judge_table.add_column("Justification Preview", style="yellow")

            for jr in judge_runs:
                justification_preview = (
                    jr.justification[:60] + "..."
                    if len(jr.justification) > 60
                    else jr.justification
                )
                judge_table.add_row(jr.winner, justification_preview)

            console.print(judge_table)

    except typer.BadParameter:
        raise
    except Exception as e:
        print_error(f"Failed to show results: {e}")
        raise typer.Exit(code=1)


def export_results_command(
    experiment_id: str = typer.Option(..., help="Experiment ID"),
    output_file: str = typer.Option(..., help="Output file path"),
    format: str = typer.Option("json", help="Export format (json or csv)"),
) -> None:
    """Export results to JSON or CSV."""
    exp_id = validate_uuid(experiment_id)

    # Validate format
    if format.lower() not in ("json", "csv"):
        print_error("Format must be 'json' or 'csv'")
        raise typer.BadParameter("Format must be 'json' or 'csv'")

    try:
        with get_session() as session:
            experiment = session.get(Experiment, exp_id)
            if not experiment:
                print_error(f"Experiment '{experiment_id}' not found")
                raise typer.BadParameter("Experiment not found")

            results = session.query(Result).filter(Result.experiment_id == exp_id).all()
            judge_runs = session.query(JudgeRun).filter(JudgeRun.experiment_id == exp_id).all()

        # Build export data
        export_data = {
            "experiment": {
                "id": str(experiment.id),
                "name": experiment.name,
                "description": experiment.description,
                "status": experiment.status,
                "judge_model": experiment.judge_model.name if experiment.judge_model else None,
                "rubric": experiment.rubric.name if experiment.rubric else None,
            },
            "results": [
                {
                    "id": str(r.id),
                    "task_name": r.task.name if r.task else None,
                    "model_name": r.model.name if r.model else None,
                    "input_text": r.input_text,
                    "output_text": r.output_text,
                    "score": r.score,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in results
            ],
            "judge_runs": [
                {
                    "id": str(jr.id),
                    "winner": jr.winner,
                    "justification": jr.justification,
                    "reasoning": jr.reasoning,
                    "rubric_scores": jr.rubric_scores,
                    "created_at": jr.created_at.isoformat() if jr.created_at else None,
                }
                for jr in judge_runs
            ],
        }

        # Export based on format
        if format.lower() == "json":
            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2)
        elif format.lower() == "csv":
            import csv

            with open(output_file, "w", newline="") as f:
                if results:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "task_name",
                            "model_name",
                            "input_text",
                            "output_text",
                            "score",
                        ],
                    )
                    writer.writeheader()
                    for r in results:
                        writer.writerow(
                            {
                                "task_name": r.task.name if r.task else "",
                                "model_name": r.model.name if r.model else "",
                                "input_text": r.input_text,
                                "output_text": r.output_text,
                                "score": r.score if r.score is not None else "",
                            }
                        )

        print_success(f"Results exported to '{output_file}'")
    except typer.BadParameter:
        raise
    except Exception as e:
        print_error(f"Failed to export results: {e}")
        raise typer.Exit(code=1)


# =============================================================================
# Register Subcommands
# =============================================================================

# Config subcommands
config_app = typer.Typer(name="config", help="Manage configuration")
app.add_typer(config_app, name="config")

config_app.command("init", help="Initialize database (create tables)")(init_db_command)
config_app.command("add-model", help="Add a model configuration")(add_model_command)
config_app.command("add-rubric", help="Add a scoring rubric")(add_rubric_command)
config_app.command("list-models", help="List all configured models")(list_models_command)
config_app.command("list-rubrics", help="List all rubrics")(list_rubrics_command)


# Task subcommands
task_app = typer.Typer(name="task", help="Manage evaluation tasks")
app.add_typer(task_app, name="task")

task_app.command("add", help="Add a task/test case")(add_task_command)
task_app.command("list", help="List all tasks")(list_tasks_command)
task_app.command("import", help="Import tasks from JSON file")(import_tasks_command)


# Experiment subcommands
experiment_app = typer.Typer(name="experiment", help="Manage experiments")
app.add_typer(experiment_app, name="experiment")

experiment_app.command("create", help="Create a new experiment")(create_experiment_command)
experiment_app.command("list", help="List all experiments")(list_experiments_command)
experiment_app.command("status", help="Show experiment status")(experiment_status_command)


# Run subcommands
run_app = typer.Typer(name="run", help="Run evaluations and comparisons")
app.add_typer(run_app, name="run")

run_app.command("experiment", help="Run an experiment by ID")(run_experiment_command)
run_app.command("compare", help="Run comparison between models")(run_compare_command)


# Results subcommands
results_app = typer.Typer(name="results", help="View and export results")
app.add_typer(results_app, name="results")

results_app.command("show", help="Show results for an experiment")(show_results_command)
results_app.command("export", help="Export results to JSON or CSV")(export_results_command)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    app()
