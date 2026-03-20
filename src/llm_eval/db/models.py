"""Database models for LLM evaluation framework."""

import uuid
from datetime import datetime
from typing import Optional

from sqlmodel import Field, JSON, Relationship, SQLModel

# =============================================================================
# ModelConfig - Model configurations (endpoint, params)
# =============================================================================


class ModelConfig(SQLModel, table=True):
    """Model configuration for LLM providers."""

    __tablename__ = "model_configs"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(
        max_length=255,
        unique=True,
        index=True,
        description="Unique model name like 'gpt-4o', 'lmstudio-local'",
    )
    provider: str = Field(
        max_length=100, description="Provider name: openai, anthropic, ollama, etc."
    )
    base_url: Optional[str] = Field(
        default=None, max_length=500, description="Base URL for API endpoints (for local models)"
    )
    api_key: Optional[str] = Field(
        default=None, max_length=500, description="API key or env var reference"
    )
    model: str = Field(max_length=255, description="Actual model name passed to API")
    default_temperature: float = Field(default=0.0, description="Default temperature parameter")
    default_max_tokens: int = Field(default=4096, description="Default max tokens limit")
    is_active: bool = Field(default=True, description="Whether this config is active")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    # Relationships
    experiment_judge: list["Experiment"] = Relationship(
        back_populates="judge_model",
        sa_relationship_kwargs={"foreign_keys": "[Experiment.judge_model_id]"},
    )
    experiment_baseline: list["Experiment"] = Relationship(
        back_populates="baseline_model",
        sa_relationship_kwargs={"foreign_keys": "[Experiment.baseline_model_id]"},
    )
    results: list["Result"] = Relationship(back_populates="model")
    ratings: list["ModelRating"] = Relationship(back_populates="model")


# =============================================================================
# Rubric - Scoring criteria
# =============================================================================


class Rubric(SQLModel, table=True):
    """Scoring criteria for evaluations."""

    __tablename__ = "rubrics"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(
        max_length=255,
        unique=True,
        index=True,
        description="Rubric name like 'quality', 'helpfulness'",
    )
    description: str = Field(description="Human-readable description of the rubric")
    weights: dict = Field(
        default={}, sa_type=JSON, description="Dict of criterion_name -> float weight"
    )
    criteria_details: Optional[dict] = Field(
        default=None, sa_type=JSON, description="Full criteria with descriptions"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    # Relationships
    experiments: list["Experiment"] = Relationship(back_populates="rubric")


# =============================================================================
# Task - Test cases/inputs
# =============================================================================


class Task(SQLModel, table=True):
    """Test cases and inputs for evaluation."""

    __tablename__ = "tasks"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(max_length=255, description="Task name")
    input_text: str = Field(description="The prompt or task input text")
    expected_output: Optional[str] = Field(default=None, description="Reference expected output")
    task_metadata: Optional[dict] = Field(
        default=None, sa_type=JSON, description="Additional task metadata"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")

    # Relationships
    results: list["Result"] = Relationship(back_populates="task")


# =============================================================================
# Prompt - Prompt templates with versions
# =============================================================================


class Prompt(SQLModel, table=True):
    """Prompt templates with versioning support."""

    __tablename__ = "prompts"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(max_length=255, description="Prompt template name")
    version: str = Field(max_length=50, description="Version string like 'v1', 'v2.0'")
    template: str = Field(description="The actual prompt template with {variables}")
    variables: Optional[list[str]] = Field(
        default=None, sa_type=JSON, description="List of variable names in template"
    )
    is_active: bool = Field(default=True, description="Whether this prompt version is active")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    # Relationships
    results: list["Result"] = Relationship(back_populates="prompt")


# =============================================================================
# Experiment - Evaluation runs
# =============================================================================


class Experiment(SQLModel, table=True):
    """Evaluation experiment runs."""

    __tablename__ = "experiments"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(max_length=255, description="Experiment name")
    description: Optional[str] = Field(
        default=None, max_length=1000, description="Experiment description"
    )
    status: str = Field(
        default="pending",
        max_length=20,
        description="Experiment status: pending, running, completed, failed",
    )
    config_snapshot: dict = Field(
        default={}, sa_type=JSON, description="Snapshot of configuration at run time"
    )
    judge_model_id: uuid.UUID = Field(
        foreign_key="model_configs.id", description="ID of judge model"
    )
    rubric_id: uuid.UUID = Field(foreign_key="rubrics.id", description="ID of rubric")
    baseline_model_id: Optional[uuid.UUID] = Field(
        default=None,
        foreign_key="model_configs.id",
        description="ID of baseline model for comparison",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")

    # Relationships
    judge_model: "ModelConfig" = Relationship(
        back_populates="experiment_judge",
        sa_relationship_kwargs={"foreign_keys": "[Experiment.judge_model_id]"},
    )
    rubric: "Rubric" = Relationship(back_populates="experiments")
    baseline_model: Optional["ModelConfig"] = Relationship(
        back_populates="experiment_baseline",
        sa_relationship_kwargs={"foreign_keys": "[Experiment.baseline_model_id]"},
    )
    results: list["Result"] = Relationship(back_populates="experiment")
    judge_runs: list["JudgeRun"] = Relationship(back_populates="experiment")
    ratings: list["ModelRating"] = Relationship(back_populates="experiment")


# =============================================================================
# Result - Per-task, per-model outputs
# =============================================================================


class Result(SQLModel, table=True):
    """Evaluation results for each task and model combination."""

    __tablename__ = "results"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    experiment_id: Optional[uuid.UUID] = Field(
        default=None, foreign_key="experiments.id", description="Parent experiment ID"
    )
    task_id: uuid.UUID = Field(foreign_key="tasks.id", description="Task ID")
    model_id: uuid.UUID = Field(
        foreign_key="model_configs.id", description="Model configuration ID"
    )
    prompt_id: Optional[uuid.UUID] = Field(
        default=None, foreign_key="prompts.id", description="Prompt template ID"
    )
    input_text: str = Field(description="The input text/prompt used")
    output_text: str = Field(description="The model output text")
    raw_response: Optional[dict] = Field(
        default=None, sa_type=JSON, description="Full API response"
    )
    score: Optional[float] = Field(default=None, description="Numeric score if available")
    extra_metadata: Optional[dict] = Field(
        default=None, sa_type=JSON, description="Additional result metadata"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")

    # Relationships
    experiment: "Experiment" = Relationship(back_populates="results")
    task: "Task" = Relationship(back_populates="results")
    model: "ModelConfig" = Relationship(back_populates="results")
    prompt: Optional["Prompt"] = Relationship(back_populates="results")
    judge_runs_a: list["JudgeRun"] = Relationship(
        back_populates="result_a", sa_relationship_kwargs={"foreign_keys": "[JudgeRun.result_a_id]"}
    )
    judge_runs_b: list["JudgeRun"] = Relationship(
        back_populates="result_b", sa_relationship_kwargs={"foreign_keys": "[JudgeRun.result_b_id]"}
    )


# =============================================================================
# JudgeRun - LLM-as-judge responses
# =============================================================================


class JudgeRun(SQLModel, table=True):
    """LLM-as-judge evaluation results."""

    __tablename__ = "judge_runs"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    experiment_id: uuid.UUID = Field(
        foreign_key="experiments.id", description="Parent experiment ID"
    )
    result_a_id: uuid.UUID = Field(foreign_key="results.id", description="First result to compare")
    result_b_id: uuid.UUID = Field(foreign_key="results.id", description="Second result to compare")
    winner: str = Field(max_length=20, description="Winner: a, b, tie, or invalid")
    justification: str = Field(description="Human-readable justification for the decision")
    reasoning: Optional[str] = Field(
        default=None, description="Chain-of-thought reasoning from judge"
    )
    judge_model_id: uuid.UUID = Field(foreign_key="model_configs.id", description="Judge model ID")
    rubric_scores: Optional[dict] = Field(
        default=None, sa_type=JSON, description="Per-criterion scores"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")

    # Relationships
    experiment: "Experiment" = Relationship(back_populates="judge_runs")
    result_a: "Result" = Relationship(
        back_populates="judge_runs_a",
        sa_relationship_kwargs={"foreign_keys": "[JudgeRun.result_a_id]"},
    )
    result_b: "Result" = Relationship(
        back_populates="judge_runs_b",
        sa_relationship_kwargs={"foreign_keys": "[JudgeRun.result_b_id]"},
    )
    judge_model: "ModelConfig" = Relationship()


# =============================================================================
# ModelRating - Elo ratings for models
# =============================================================================


class ModelRating(SQLModel, table=True):
    """Elo ratings and statistics for models."""

    __tablename__ = "model_ratings"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    model_id: uuid.UUID = Field(
        foreign_key="model_configs.id", description="Model configuration ID"
    )
    experiment_id: uuid.UUID = Field(foreign_key="experiments.id", description="Experiment ID")
    rating: float = Field(default=1500.0, description="Current Elo rating")
    wins: int = Field(default=0, description="Number of wins")
    losses: int = Field(default=0, description="Number of losses")
    ties: int = Field(default=0, description="Number of ties")
    matches_played: int = Field(default=0, description="Total matches played")
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    # Relationships
    model: "ModelConfig" = Relationship(back_populates="ratings")
    experiment: "Experiment" = Relationship(back_populates="ratings")


# =============================================================================
# Re-export for convenience
# =============================================================================

__all__ = [
    "ModelConfig",
    "Rubric",
    "Task",
    "Prompt",
    "Experiment",
    "Result",
    "JudgeRun",
    "ModelRating",
]
