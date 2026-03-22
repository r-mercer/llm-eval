"""Experiment management commands."""

from typing import Optional

import typer
from rich.panel import Panel
from sqlmodel import desc, select

from llm_eval.cli._common import (
    console,
    print_error,
    print_info,
    print_success,
    validate_uuid,
)
from llm_eval.db.models import Experiment, JudgeRun, ModelConfig, Result, Rubric
from llm_eval.db.session import get_session


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
            judge = session.query(ModelConfig).filter_by(is_active=True, name=judge_model).first()
            if not judge:
                print_error(f"Judge model '{judge_model}' not found")
                raise typer.BadParameter(f"Judge model '{judge_model}' not found")

            rubric_obj = session.query(Rubric).filter_by(name=rubric).first()
            if not rubric_obj:
                print_error(f"Rubric '{rubric}' not found")
                raise typer.BadParameter(f"Rubric '{rubric}' not found")

            baseline = None
            if baseline_model:
                baseline = (
                    session.query(ModelConfig)
                    .filter_by(is_active=True, name=baseline_model)
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
    from rich.table import Table

    try:
        with get_session() as session:
            experiments = list(
                session.exec(select(Experiment).order_by(desc(Experiment.created_at)))
            )

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

            results_count = session.query(Result).filter_by(experiment_id=exp_id).count()
            judge_runs_count = session.query(JudgeRun).filter_by(experiment_id=exp_id).count()

        status_color = {
            "pending": "yellow",
            "running": "cyan",
            "completed": "green",
            "failed": "red",
        }.get(experiment.status, "white")

        content = f"""
[b]Name:[/b] {experiment.name}
[b]Status:[/b] [{status_color}]{experiment.status}[/{status_color}]
[b]Description:[/b] {experiment.description or "N/A"}
[b]Judge Model:[/b] {experiment.judge_model.name if experiment.judge_model else "N/A"}
[b]Rubric:[/b] {experiment.rubric.name if experiment.rubric else "N/A"}
[b]Baseline Model:[/b] {experiment.baseline_model.name if experiment.baseline_model else "N/A"}
[b]Results:[/b] {results_count}
[b]Judge Runs:[/b] {judge_runs_count}
[b]Created:[/b] {experiment.created_at.strftime("%Y-%m-%d %H:%M") if experiment.created_at else "N/A"}
[b]Completed:[/b] {experiment.completed_at.strftime("%Y-%m-%d %H:%M") if experiment.completed_at else "N/A"}
        """

        console.print(Panel(content.strip(), title="Experiment Status", border_style="cyan"))
    except typer.BadParameter:
        raise
    except Exception as e:
        print_error(f"Failed to get experiment status: {e}")
        raise typer.Exit(code=1)


def leaderboard_command(
    experiment_id: str = typer.Option(..., help="Experiment ID"),
) -> None:
    """Show the model rankings/leaderboard for an experiment."""
    from rich.table import Table
    from sqlmodel import select

    exp_id = validate_uuid(experiment_id)

    try:
        with get_session() as session:
            experiment = session.get(Experiment, exp_id)
            if not experiment:
                print_error(f"Experiment '{experiment_id}' not found")
                raise typer.BadParameter("Experiment not found")

            from llm_eval.db.models import ModelRating

            ratings = list(
                session.exec(
                    select(ModelRating)
                    .where(ModelRating.experiment_id == exp_id)
                    .order_by(desc(ModelRating.rating))
                )
            )

        if not ratings:
            print_info("No rankings found for this experiment. Run the experiment first.")
            return

        table = Table(
            title=f"Leaderboard for '{experiment.name}'",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Rank", style="cyan", justify="right")
        table.add_column("Model", style="green")
        table.add_column("Rating", style="yellow", justify="right")
        table.add_column("Wins", style="green", justify="right")
        table.add_column("Losses", style="red", justify="right")
        table.add_column("Ties", style="blue", justify="right")
        table.add_column("Matches", style="white", justify="right")
        table.add_column("Win Rate", style="magenta", justify="right")

        for rank, rating in enumerate(ratings, start=1):
            model_name = rating.model.name if rating.model else "-"
            matches = rating.matches_played
            win_rate = rating.wins / matches * 100 if matches > 0 else 0
            table.add_row(
                str(rank),
                model_name,
                f"{rating.rating:.0f}",
                str(rating.wins),
                str(rating.losses),
                str(rating.ties),
                str(matches),
                f"{win_rate:.1f}%",
            )

        console.print(table)
    except typer.BadParameter:
        raise
    except Exception as e:
        print_error(f"Failed to show leaderboard: {e}")
        raise typer.Exit(code=1)
