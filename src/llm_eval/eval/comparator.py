"""Pairwise comparison logic for model evaluation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from llm_eval.db.models import ModelConfig, Rubric
from llm_eval.eval.judge import Judge, JudgeConfig, JudgeResult

if TYPE_CHECKING:
    from llm_eval.db.models import Result


@dataclass(frozen=True)
class ComparisonResult:
    """Result of a single comparison between two model outputs."""

    winner: str
    """Winner of the comparison: 'a', 'b', or 'tie'."""

    model_a_id: str
    """ID of model A that was compared."""

    model_b_id: str
    """ID of model B that was compared."""

    judge_result: JudgeResult
    """The detailed result from the judge evaluation."""


class Comparator:
    """Handles pairwise comparisons between model outputs.

    This class wraps a Judge instance to provide a simplified interface
    for comparing two model outputs and returning a structured result.
    """

    def __init__(self, judge: Judge) -> None:
        """Initialize the comparator with a judge instance.

        Args:
            judge: The Judge instance to use for evaluations.
        """
        self._judge = judge

    def compare(
        self,
        result_a: "Result",
        result_b: "Result",
        task_description: str,
    ) -> ComparisonResult:
        """Compare two model outputs using the judge.

        Position randomization is handled internally by the judge to mitigate
        position bias. The winner is mapped back to the original model IDs.

        Args:
            result_a: The first result to compare (response A).
            result_b: The second result to compare (response B).
            task_description: Description of the task being evaluated.

        Returns:
            ComparisonResult with the winner mapped to original model IDs.

        Raises:
            ValueError: If required fields are missing from results.
        """
        # Guard clause: Validate inputs have required fields
        if not result_a.output_text or not result_a.output_text.strip():
            raise ValueError("result_a.output_text cannot be empty")

        if not result_b.output_text or not result_b.output_text.strip():
            raise ValueError("result_b.output_text cannot be empty")

        if not result_a.input_text or not result_a.input_text.strip():
            raise ValueError("result_a.input_text cannot be empty")

        if not task_description or not task_description.strip():
            raise ValueError("task_description cannot be empty")

        # Judge handles position randomization internally
        judge_result = self._judge.evaluate_pairwise(
            response_a=result_a.output_text,
            response_b=result_b.output_text,
            prompt=result_a.input_text,
            task_description=task_description,
        )

        # Map winner back to original model IDs
        return ComparisonResult(
            winner=judge_result.winner,
            model_a_id=str(result_a.model_id),
            model_b_id=str(result_b.model_id),
            judge_result=judge_result,
        )


def create_comparator(
    judge_model: ModelConfig,
    rubric: Optional[Rubric] = None,
) -> Comparator:
    """Create a comparator with the specified judge model.

    Args:
        judge_model: The model configuration for the judge LLM.
        rubric: Optional rubric for scoring criteria.

    Returns:
        Configured Comparator instance.

    Raises:
        ValueError: If judge_model is not provided.
    """
    # Guard clause: Validate required inputs
    if not judge_model:
        raise ValueError("judge_model is required")

    judge_config = JudgeConfig(
        model_config=judge_model,
        rubric=rubric,
    )
    judge = Judge(judge_config)
    return Comparator(judge)
