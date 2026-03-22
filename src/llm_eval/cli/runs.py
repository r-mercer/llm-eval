"""Run commands for executing experiments and comparisons."""

import uuid
from typing import Optional

import typer
from rich.table import Table
from sqlalchemy import func as sql_func

from llm_eval.cli._common import (
    console,
    print_error,
    print_info,
    print_success,
    validate_uuid,
)
from llm_eval.db.models import ModelConfig, Rubric, Task
from llm_eval.db.session import get_session
from llm_eval.eval.runner import ExperimentRunner


def run_experiment_command(
    experiment_id: str = typer.Option(..., help="Experiment ID to run"),
) -> None:
    """Run an experiment by ID."""
    exp_id = validate_uuid(experiment_id)

    experiment_name = None
    try:
        with get_session() as session:
            from llm_eval.db.models import Experiment

            experiment = session.get(Experiment, exp_id)
            if not experiment:
                print_error(f"Experiment '{experiment_id}' not found")
                raise typer.BadParameter("Experiment not found")

            experiment_name = experiment.name
            if experiment.status == "completed":
                print_error("Experiment already completed. Create a new experiment to run again.")
                raise typer.BadParameter("Experiment already completed")

            experiment.status = "running"
            session.flush()

        print_info(f"Running experiment '{experiment_name}'...")

        with get_session() as session:
            runner = ExperimentRunner(session)
            runner.run_evaluation(exp_id)

        print_success(f"Experiment '{experiment_name}' completed!")
    except typer.BadParameter:
        raise
    except Exception as e:
        try:
            with get_session() as session:
                from llm_eval.db.models import Experiment

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
    model_ids: list = []
    task_ids: list = []
    rubric_id: uuid.UUID = uuid.UUID(int=0)

    try:
        with get_session() as session:
            model_config_a = session.query(ModelConfig).filter_by(name=model_a).first()
            if not model_config_a:
                print_error(f"Model '{model_a}' not found")
                raise typer.BadParameter(f"Model '{model_a}' not found")
            model_ids.append(model_config_a.id)

            model_config_b = session.query(ModelConfig).filter_by(name=model_b).first()
            if not model_config_b:
                print_error(f"Model '{model_b}' not found")
                raise typer.BadParameter(f"Model '{model_b}' not found")
            model_ids.append(model_config_b.id)

            judge = session.query(ModelConfig).filter_by(name=judge_model).first()
            if not judge:
                print_error(f"Judge model '{judge_model}' not found")
                raise typer.BadParameter(f"Judge model '{judge_model}' not found")

            rubric_obj = session.query(Rubric).filter_by(name=rubric).first()
            if not rubric_obj:
                print_error(f"Rubric '{rubric}' not found")
                raise typer.BadParameter(f"Rubric '{rubric}' not found")
            rubric_id = rubric_obj.id

            task_query = session.query(Task)
            if tasks:
                task_names = [t.strip() for t in tasks.split(",")]
                task_query = task_query.filter(sql_func.in_(Task.name, task_names))
            task_list = task_query.all()

            if not task_list:
                print_error("No tasks found to compare")
                raise typer.BadParameter("No tasks found to compare")

            for t in task_list:
                task_ids.append(t.id)

        print_info(f"Comparing '{model_a}' vs '{model_b}' on {len(task_list)} tasks...")

        with get_session() as session:
            runner = ExperimentRunner(session)
            judge_runs = runner.run_comparison(
                model_ids=model_ids,
                task_ids=task_ids,
                judge_model_id=judge.id,
                rubric_id=rubric_id,
            )

        wins_a = sum(1 for jr in judge_runs if jr.winner == "a")
        wins_b = sum(1 for jr in judge_runs if jr.winner == "b")
        ties = sum(1 for jr in judge_runs if jr.winner == "tie")
        total = len(judge_runs) if judge_runs else 1
        win_rate_a = wins_a / total * 100 if total > 0 else 0
        win_rate_b = wins_b / total * 100 if total > 0 else 0

        print_success("Comparison complete!")

        table = Table(title="Comparison Results", show_header=True, header_style="bold cyan")
        table.add_column("Model", style="green")
        table.add_column("Wins", style="yellow")
        table.add_column("Losses", style="red")
        table.add_column("Ties", style="blue")
        table.add_column("Win Rate", style="magenta")

        table.add_row(model_a, str(wins_a), str(wins_b), str(ties), f"{win_rate_a:.1f}%")
        table.add_row(model_b, str(wins_b), str(wins_a), str(ties), f"{win_rate_b:.1f}%")

        console.print(table)
    except typer.BadParameter:
        raise
    except Exception as e:
        print_error(f"Failed to run comparison: {e}")
        raise typer.Exit(code=1)
