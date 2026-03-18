"""Experiment runner module for orchestrating evaluation experiments.

This module provides the ExperimentRunner class which orchestrates running
evaluation experiments including generating model outputs, running judge comparisons,
and updating Elo ratings.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from sqlmodel import select

from llm_eval.db.models import (
    Experiment,
    JudgeRun,
    ModelConfig,
    ModelRating,
    Result,
    Rubric,
    Task,
)
from llm_eval.db.session import Session
from llm_eval.eval.comparator import Comparator, create_comparator
from llm_eval.eval.ranking import EloRating
from llm_eval.models.provider import ProviderFactory


# =============================================================================
# Custom Exceptions
# =============================================================================


class RunnerError(Exception):
    """Base exception for runner-related errors."""

    pass


class ExperimentNotFoundError(RunnerError):
    """Raised when an experiment cannot be found."""

    pass


class ModelNotFoundError(RunnerError):
    """Raised when a model configuration cannot be found."""

    pass


class TaskNotFoundError(RunnerError):
    """Raised when a task cannot be found."""

    pass


class ExperimentStatusError(RunnerError):
    """Raised when experiment status is invalid for the operation."""

    pass


# =============================================================================
# ExperimentRunner Class
# =============================================================================


class ExperimentRunner:
    """Orchestrates running evaluation experiments.

    This class handles the full lifecycle of an experiment:
    1. Generate model outputs for all tasks
    2. Run pairwise comparisons using a judge model
    3. Update Elo ratings based on results

    All database operations are performed within the provided session.
    """

    def __init__(self, session: Session) -> None:
        """Initialize the experiment runner.

        Args:
            session: The database session for persistence operations.

        Raises:
            ValueError: If session is None.
        """
        # Guard clause: Validate session
        if session is None:
            raise ValueError("session is required")

        self._session = session

    def run_evaluation(self, experiment_id: UUID) -> Experiment:
        """Run a full evaluation experiment.

        This method orchestrates the complete evaluation pipeline:
        1. Fetch experiment and validate it's in pending state
        2. Generate outputs for each model on each task
        3. Run judge comparisons between all model pairs
        4. Update Elo ratings based on results

        Args:
            experiment_id: The UUID of the experiment to run.

        Returns:
            The updated experiment with completed status.

        Raises:
            ExperimentNotFoundError: If experiment doesn't exist.
            ExperimentStatusError: If experiment is not in pending state.
            RunnerError: If any step in the pipeline fails.
        """
        # Fetch experiment with validation
        experiment = self._get_experiment(experiment_id)

        # Guard clause: Validate experiment status
        if experiment.status != "pending":
            raise ExperimentStatusError(
                f"Experiment {experiment_id} is not in pending state. "
                f"Current status: {experiment.status}"
            )

        # Update status to running
        self._update_experiment_status(experiment, "running")

        try:
            # Step 1: Generate model outputs
            self._generate_outputs(experiment)

            # Step 2: Run judge comparisons
            self._run_judgments(experiment)

            # Step 3: Update Elo ratings
            self._update_rankings(experiment)

            # Mark as completed
            self._update_experiment_status(experiment, "completed")

            return experiment

        except Exception as e:
            # Mark as failed on any error
            self._update_experiment_status(experiment, "failed")
            raise RunnerError(f"Experiment {experiment_id} failed: {e}") from e

    def run_comparison(
        self,
        model_ids: list[UUID],
        task_ids: list[UUID],
        judge_model_id: UUID,
        rubric_id: UUID,
    ) -> list[JudgeRun]:
        """Run a model comparison without creating a persisted experiment.

        This is a convenience method for running ad-hoc comparisons between models
        on specific tasks using a judge model.

        Args:
            model_ids: List of model UUIDs to compare.
            task_ids: List of task UUIDs to use for comparison.
            judge_model_id: UUID of the model to use as judge.
            rubric_id: UUID of the rubric to use for evaluation.

        Returns:
            List of JudgeRun results from the comparisons.

        Raises:
            ModelNotFoundError: If any model is not found.
            TaskNotFoundError: If any task is not found.
            RunnerError: If comparison fails.
        """
        # Guard clause: Validate inputs
        if not model_ids:
            raise ValueError("model_ids cannot be empty")

        if len(model_ids) < 2:
            raise ValueError("At least 2 models are required for comparison")

        if not task_ids:
            raise ValueError("task_ids cannot be empty")

        # Fetch models and tasks
        models = self._get_models(model_ids)
        tasks = self._get_tasks(task_ids)
        judge_model = self._get_model(judge_model_id)
        rubric = self._get_rubric(rubric_id)

        # Generate outputs for all model-task combinations
        results: dict[UUID, dict[UUID, Result]] = {}  # model_id -> task_id -> Result

        for model in models:
            results[model.id] = {}
            provider = ProviderFactory.create(model)

            for task in tasks:
                # Generate output
                response = provider.generate(prompt=task.input_text)

                # Store result
                result = Result(
                    experiment_id=None,  # No experiment for ad-hoc comparison
                    task_id=task.id,
                    model_id=model.id,
                    input_text=task.input_text,
                    output_text=response.content,
                    raw_response=response.raw_response,
                )
                self._session.add(result)
                results[model.id][task.id] = result

        self._session.commit()

        # Run comparisons
        comparator = create_comparator(judge_model, rubric)
        judge_runs: list[JudgeRun] = []

        # Compare all pairs on all tasks
        for task in tasks:
            for i, model_a in enumerate(models):
                for model_b in models[i + 1:]:
                    result_a = results[model_a.id][task.id]
                    result_b = results[model_b.id][task.id]

                    comparison = comparator.compare(
                        result_a=result_a,
                        result_b=result_b,
                        task_description=task.name,
                    )

                    # Create JudgeRun
                    judge_run = JudgeRun(
                        experiment_id=None,
                        result_a_id=result_a.id,
                        result_b_id=result_b.id,
                        winner=comparison.winner,
                        justification=comparison.judge_result.justification,
                        reasoning=comparison.judge_result.reasoning,
                        judge_model_id=judge_model_id,
                        rubric_scores=comparison.judge_result.criteria_scores,
                    )
                    self._session.add(judge_run)
                    judge_runs.append(judge_run)

        self._session.commit()
        return judge_runs

    def _generate_outputs(self, experiment: Experiment) -> None:
        """Generate model outputs for all tasks in the experiment.

        Args:
            experiment: The experiment to generate outputs for.

        Raises:
            RunnerError: If output generation fails.
        """
        # Get all models and tasks for this experiment
        # Note: In a full implementation, this would fetch from experiment associations
        # For now, we fetch all active models
        models = self._get_all_active_models()
        tasks = self._get_all_tasks()

        # Guard clause: Validate we have models and tasks
        if not models:
            raise RunnerError("No models available for evaluation")

        if not tasks:
            raise RunnerError("No tasks available for evaluation")

        # Generate outputs for each model-task combination
        for model in models:
            provider = ProviderFactory.create(model)

            for task in tasks:
                # Skip if result already exists
                existing = self._get_existing_result(experiment.id, task.id, model.id)
                if existing:
                    continue

                try:
                    response = provider.generate(prompt=task.input_text)

                    result = Result(
                        experiment_id=experiment.id,
                        task_id=task.id,
                        model_id=model.id,
                        input_text=task.input_text,
                        output_text=response.content,
                        raw_response=response.raw_response,
                    )
                    self._session.add(result)

                except Exception as e:
                    # Log error but continue with other outputs
                    result = Result(
                        experiment_id=experiment.id,
                        task_id=task.id,
                        model_id=model.id,
                        input_text=task.input_text,
                        output_text=f"[ERROR: {type(e).__name__}] {str(e)}",
                        raw_response=None,
                    )
                    self._session.add(result)

        self._session.commit()

    def _run_judgments(self, experiment: Experiment) -> None:
        """Run judge comparisons between all model pairs.

        Args:
            experiment: The experiment to run judgments for.

        Raises:
            RunnerError: If judgment execution fails.
        """
        # Get judge model and rubric
        judge_model = self._get_model(experiment.judge_model_id)
        rubric = self._get_rubric(experiment.rubric_id) if experiment.rubric_id else None

        # Create comparator
        comparator = create_comparator(judge_model, rubric)

        # Get all results for this experiment
        results = self._get_experiment_results(experiment.id)

        # Group results by task
        results_by_task: dict[UUID, list[Result]] = {}
        for result in results:
            if result.task_id not in results_by_task:
                results_by_task[result.task_id] = []
            results_by_task[result.task_id].append(result)

        # Get task info for descriptions
        tasks = {t.id: t for t in self._get_all_tasks()}

        # Run pairwise comparisons
        for task_id, task_results in results_by_task.items():
            task = tasks.get(task_id)
            if not task:
                continue

            # Compare all pairs
            for i, result_a in enumerate(task_results):
                for result_b in task_results[i + 1:]:
                    # Skip if comparison already exists
                    existing = self._get_existing_judge_run(
                        experiment.id, result_a.id, result_b.id
                    )
                    if existing:
                        continue

                    try:
                        comparison = comparator.compare(
                            result_a=result_a,
                            result_b=result_b,
                            task_description=task.name,
                        )

                        judge_run = JudgeRun(
                            experiment_id=experiment.id,
                            result_a_id=result_a.id,
                            result_b_id=result_b.id,
                            winner=comparison.winner,
                            justification=comparison.judge_result.justification,
                            reasoning=comparison.judge_result.reasoning,
                            judge_model_id=judge_model.id,
                            rubric_scores=comparison.judge_result.criteria_scores,
                        )
                        self._session.add(judge_run)

                    except Exception as e:
                        # Log error but continue
                        judge_run = JudgeRun(
                            experiment_id=experiment.id,
                            result_a_id=result_a.id,
                            result_b_id=result_b.id,
                            winner="invalid",
                            justification=f"Error during judgment: {type(e).__name__}: {str(e)}",
                            reasoning=None,
                            judge_model_id=judge_model.id,
                            rubric_scores=None,
                        )
                        self._session.add(judge_run)

        self._session.commit()

    def _update_rankings(self, experiment: Experiment) -> None:
        """Update Elo ratings based on judge results.

        Args:
            experiment: The experiment to update rankings for.

        Raises:
            RunnerError: If ranking update fails.
        """
        # Get all judge runs for this experiment
        judge_runs = self._get_judge_runs(experiment.id)

        if not judge_runs:
            # No results to rank
            return

        # Initialize Elo rating system
        elo = EloRating()

        # Update ratings based on judge results
        for judge_run in judge_runs:
            # Get model IDs from results
            result_a = self._session.get(Result, judge_run.result_a_id)
            result_b = self._session.get(Result, judge_run.result_b_id)

            if not result_a or not result_b:
                continue

            model_a_id = str(result_a.model_id)
            model_b_id = str(result_b.model_id)

            # Update ratings
            elo.update_ratings(
                winner=judge_run.winner,
                model_a_id=model_a_id,
                model_b_id=model_b_id,
            )

        # Persist ratings to database
        leaderboard = elo.get_leaderboard()

        for entry in leaderboard:
            model_id = UUID(entry["model_id"])

            # Check if rating exists
            existing = self._get_model_rating(experiment.id, model_id)

            if existing:
                # Update existing rating
                existing.rating = entry["rating"]
                existing.wins = entry["wins"]
                existing.losses = entry["losses"]
                existing.ties = entry["ties"]
                existing.matches_played = entry["matches"]
                existing.updated_at = datetime.utcnow()
            else:
                # Create new rating
                rating = ModelRating(
                    model_id=model_id,
                    experiment_id=experiment.id,
                    rating=entry["rating"],
                    wins=entry["wins"],
                    losses=entry["losses"],
                    ties=entry["ties"],
                    matches_played=entry["matches"],
                )
                self._session.add(rating)

        self._session.commit()

    # =============================================================================
    # Private Helper Methods
    # =============================================================================

    def _get_experiment(self, experiment_id: UUID) -> Experiment:
        """Fetch experiment by ID.

        Args:
            experiment_id: The experiment UUID.

        Returns:
            The experiment instance.

        Raises:
            ExperimentNotFoundError: If experiment doesn't exist.
        """
        experiment = self._session.get(Experiment, experiment_id)

        if experiment is None:
            raise ExperimentNotFoundError(f"Experiment {experiment_id} not found")

        return experiment

    def _get_model(self, model_id: UUID) -> ModelConfig:
        """Fetch model configuration by ID.

        Args:
            model_id: The model UUID.

        Returns:
            The model configuration.

        Raises:
            ModelNotFoundError: If model doesn't exist.
        """
        model = self._session.get(ModelConfig, model_id)

        if model is None:
            raise ModelNotFoundError(f"Model {model_id} not found")

        return model

    def _get_models(self, model_ids: list[UUID]) -> list[ModelConfig]:
        """Fetch multiple models by IDs.

        Args:
            model_ids: List of model UUIDs.

        Returns:
            List of model configurations.

        Raises:
            ModelNotFoundError: If any model doesn't exist.
        """
        models = []
        for model_id in model_ids:
            model = self._get_model(model_id)
            models.append(model)

        return models

    def _get_rubric(self, rubric_id: UUID) -> Rubric:
        """Fetch rubric by ID.

        Args:
            rubric_id: The rubric UUID.

        Returns:
            The rubric instance.

        Raises:
            RunnerError: If rubric doesn't exist.
        """
        rubric = self._session.get(Rubric, rubric_id)

        if rubric is None:
            raise RunnerError(f"Rubric {rubric_id} not found")

        return rubric

    def _get_tasks(self, task_ids: list[UUID]) -> list[Task]:
        """Fetch multiple tasks by IDs.

        Args:
            task_ids: List of task UUIDs.

        Returns:
            List of tasks.

        Raises:
            TaskNotFoundError: If any task doesn't exist.
        """
        tasks = []
        for task_id in task_ids:
            task = self._session.get(Task, task_id)
            if task is None:
                raise TaskNotFoundError(f"Task {task_id} not found")
            tasks.append(task)

        return tasks

    def _get_all_active_models(self) -> list[ModelConfig]:
        """Fetch all active models.

        Returns:
            List of active model configurations.
        """
        statement = select(ModelConfig).where(ModelConfig.is_active == True)
        return list(self._session.exec(statement).all())

    def _get_all_tasks(self) -> list[Task]:
        """Fetch all tasks.

        Returns:
            List of all tasks.
        """
        return list(self._session.exec(select(Task)).all())

    def _get_experiment_results(self, experiment_id: UUID) -> list[Result]:
        """Fetch all results for an experiment.

        Args:
            experiment_id: The experiment UUID.

        Returns:
            List of results.
        """
        statement = select(Result).where(Result.experiment_id == experiment_id)
        return list(self._session.exec(statement).all())

    def _get_judge_runs(self, experiment_id: UUID) -> list[JudgeRun]:
        """Fetch all judge runs for an experiment.

        Args:
            experiment_id: The experiment UUID.

        Returns:
            List of judge runs.
        """
        statement = select(JudgeRun).where(JudgeRun.experiment_id == experiment_id)
        return list(self._session.exec(statement).all())

    def _get_existing_result(
        self,
        experiment_id: UUID,
        task_id: UUID,
        model_id: UUID,
    ) -> Optional[Result]:
        """Check if a result already exists for the given combination.

        Args:
            experiment_id: The experiment UUID.
            task_id: The task UUID.
            model_id: The model UUID.

        Returns:
            Existing result or None.
        """
        statement = (
            select(Result)
            .where(Result.experiment_id == experiment_id)
            .where(Result.task_id == task_id)
            .where(Result.model_id == model_id)
        )
        return self._session.exec(statement).first()

    def _get_existing_judge_run(
        self,
        experiment_id: UUID,
        result_a_id: UUID,
        result_b_id: UUID,
    ) -> Optional[JudgeRun]:
        """Check if a judge run already exists for the given comparison.

        Args:
            experiment_id: The experiment UUID.
            result_a_id: First result UUID.
            result_b_id: Second result UUID.

        Returns:
            Existing judge run or None.
        """
        statement = (
            select(JudgeRun)
            .where(JudgeRun.experiment_id == experiment_id)
            .where(JudgeRun.result_a_id == result_a_id)
            .where(JudgeRun.result_b_id == result_b_id)
        )
        return self._session.exec(statement).first()

    def _get_model_rating(
        self,
        experiment_id: UUID,
        model_id: UUID,
    ) -> Optional[ModelRating]:
        """Fetch model rating for an experiment.

        Args:
            experiment_id: The experiment UUID.
            model_id: The model UUID.

        Returns:
            Existing rating or None.
        """
        statement = (
            select(ModelRating)
            .where(ModelRating.experiment_id == experiment_id)
            .where(ModelRating.model_id == model_id)
        )
        return self._session.exec(statement).first()

    def _update_experiment_status(self, experiment: Experiment, status: str) -> None:
        """Update experiment status.

        Args:
            experiment: The experiment to update.
            status: New status value.
        """
        experiment.status = status
        experiment.updated_at = datetime.utcnow()

        if status == "completed":
            experiment.completed_at = datetime.utcnow()

        self._session.add(experiment)
        self._session.commit()


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def create_runner(session: Session) -> ExperimentRunner:
    """Create an ExperimentRunner instance.

    Args:
        session: The database session.

    Returns:
        Configured ExperimentRunner instance.
    """
    return ExperimentRunner(session)
