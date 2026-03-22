"""Results commands for viewing and exporting experiment results."""

import csv
import json

import typer
from rich.table import Table
from sqlmodel import desc, select

from llm_eval.cli._common import (
    console,
    print_error,
    print_info,
    print_success,
    validate_uuid,
)
from llm_eval.db.models import Experiment, JudgeRun, Result
from llm_eval.db.session import get_session


def show_results_command(
    experiment_id: str = typer.Option(..., help="Experiment ID"),
    limit: int = typer.Option(10, help="Number of results to show"),
) -> None:
    """Show results for an experiment."""
    exp_id = validate_uuid(experiment_id)

    if limit < 1:
        print_error("Limit must be at least 1")
        raise typer.BadParameter("Limit must be at least 1")

    try:
        with get_session() as session:
            experiment = session.get(Experiment, exp_id)
            if not experiment:
                print_error(f"Experiment '{experiment_id}' not found")
                raise typer.BadParameter("Experiment not found")

            results = list(
                session.exec(
                    select(Result)
                    .where(Result.experiment_id == exp_id)
                    .order_by(desc(Result.created_at))
                    .limit(limit)
                )
            )

        if not results:
            print_info("No results found for this experiment.")
            return

        table = Table(
            title=f"Results for '{experiment.name}'",
            show_header=True,
            header_style="bold cyan",
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

        with get_session() as session:
            judge_runs = list(
                session.exec(select(JudgeRun).where(JudgeRun.experiment_id == exp_id).limit(limit))
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

    if format.lower() not in ("json", "csv"):
        print_error("Format must be 'json' or 'csv'")
        raise typer.BadParameter("Format must be 'json' or 'csv'")

    try:
        with get_session() as session:
            experiment = session.get(Experiment, exp_id)
            if not experiment:
                print_error(f"Experiment '{experiment_id}' not found")
                raise typer.BadParameter("Experiment not found")

            results = list(session.exec(select(Result).where(Result.experiment_id == exp_id)))
            judge_runs = list(
                session.exec(select(JudgeRun).where(JudgeRun.experiment_id == exp_id))
            )

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

        if format.lower() == "json":
            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2)
        elif format.lower() == "csv":
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
