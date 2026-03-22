"""Shared CLI helpers."""

import uuid

import typer
from rich.console import Console
from rich.theme import Theme


custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
    }
)

console = Console(theme=custom_theme)


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
