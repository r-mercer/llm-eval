"""Task management commands."""

import json
from pathlib import Path
from typing import Optional

import typer

from llm_eval.cli._common import console, print_error, print_info, print_success
from llm_eval.db.models import Task
from llm_eval.db.session import get_session


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
    from rich.table import Table
    from sqlmodel import desc, select

    try:
        with get_session() as session:
            tasks = list(session.exec(select(Task).order_by(desc(Task.created_at))).all())

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
    path = Path(file_path)
    if not path.exists():
        print_error(f"File not found: {file_path}")
        raise typer.BadParameter(f"File not found: {file_path}")

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        raise typer.BadParameter(f"Invalid JSON in file: {e}")
    except Exception as e:
        print_error(f"Failed to read file: {e}")
        raise typer.BadParameter(f"Failed to read file: {e}")

    if not isinstance(data, list):
        print_error("JSON file must contain a list of tasks")
        raise typer.BadParameter("JSON file must contain a list of tasks")

    for i, task_data in enumerate(data):
        if not isinstance(task_data, dict):
            print_error(f"Task at index {i} is not an object")
            raise typer.BadParameter(f"Task at index {i} is not an object")
        if "name" not in task_data or "input_text" not in task_data:
            print_error(f"Task at index {i} missing required fields (name, input_text)")
            raise typer.BadParameter(
                f"Task at index {i} missing required fields (name, input_text)"
            )

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
