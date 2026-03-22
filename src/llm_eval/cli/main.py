"""Main CLI entry point for llm_eval."""

import typer

from llm_eval.cli import config as config_cmds
from llm_eval.cli import experiments as experiment_cmds
from llm_eval.cli import results as results_cmds
from llm_eval.cli import runs as run_cmds
from llm_eval.cli import tasks as task_cmds

app = typer.Typer(
    name="llm-eval",
    help="[bold]LLM Evaluation Framework CLI[/bold]",
    add_completion=False,
    rich_markup_mode="rich",
)

# =============================================================================
# Register Subcommands
# =============================================================================

config_app = typer.Typer(name="config", help="Manage configuration")
app.add_typer(config_app, name="config")

config_app.command("init", help="Initialize database (create tables)")(config_cmds.init_db_command)
config_app.command("add-model", help="Add a model configuration")(config_cmds.add_model_command)
config_app.command("add-rubric", help="Add a scoring rubric")(config_cmds.add_rubric_command)
config_app.command("list-models", help="List all configured models")(
    config_cmds.list_models_command
)
config_app.command("list-rubrics", help="List all rubrics")(config_cmds.list_rubrics_command)


task_app = typer.Typer(name="task", help="Manage evaluation tasks")
app.add_typer(task_app, name="task")

task_app.command("add", help="Add a task/test case")(task_cmds.add_task_command)
task_app.command("list", help="List all tasks")(task_cmds.list_tasks_command)
task_app.command("import", help="Import tasks from JSON file")(task_cmds.import_tasks_command)


experiment_app = typer.Typer(name="experiment", help="Manage experiments")
app.add_typer(experiment_app, name="experiment")

experiment_app.command("create", help="Create a new experiment")(
    experiment_cmds.create_experiment_command
)
experiment_app.command("list", help="List all experiments")(
    experiment_cmds.list_experiments_command
)
experiment_app.command("status", help="Show experiment status")(
    experiment_cmds.experiment_status_command
)
experiment_app.command("leaderboard", help="Show model rankings for an experiment")(
    experiment_cmds.leaderboard_command
)


run_app = typer.Typer(name="run", help="Run evaluations and comparisons")
app.add_typer(run_app, name="run")

run_app.command("experiment", help="Run an experiment by ID")(run_cmds.run_experiment_command)
run_app.command("compare", help="Run comparison between models")(run_cmds.run_compare_command)


results_app = typer.Typer(name="results", help="View and export results")
app.add_typer(results_app, name="results")

results_app.command("show", help="Show results for an experiment")(
    results_cmds.show_results_command
)
results_app.command("export", help="Export results to JSON or CSV")(
    results_cmds.export_results_command
)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    app()
