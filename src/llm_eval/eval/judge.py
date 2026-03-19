"""Judge evaluation logic for LLM-as-judge pairwise and pointwise evaluation."""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from jinja2 import Template

from llm_eval.db.models import ModelConfig, Rubric
from llm_eval.models.provider import ModelProvider, ProviderFactory


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass(frozen=True)
class JudgeConfig:
    """Configuration for the judge evaluation.

    This immutable dataclass holds all configuration needed to run judge evaluations.
    """

    model_config: ModelConfig
    """The model configuration for the judge LLM."""

    template_name: str = "judge_pairwise_coot.j2"
    """Template file name for rendering prompts."""

    temperature: float = 0.0
    """Temperature for LLM generation."""

    max_tokens: int = 4096
    """Maximum tokens for LLM generation."""

    rubric: Optional[Rubric] = None
    """Optional rubric for scoring criteria."""


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass(frozen=True)
class JudgeResult:
    """Result from a judge evaluation.

    This immutable dataclass represents the parsed output from a judge LLM.
    """

    winner: str
    """Winner of the comparison: 'a', 'b', 'tie', or 'invalid'."""

    justification: str
    """Human-readable justification for the decision."""

    raw_response: str
    """The raw response text from the judge LLM."""

    reasoning: Optional[str] = None
    """Chain-of-thought reasoning from the judge (for CoT templates)."""

    criteria_scores: Optional[dict[str, Any]] = None
    """Per-criterion scores from rubric-based evaluation."""


# =============================================================================
# Custom Exceptions
# =============================================================================


class JudgeError(Exception):
    """Base exception for judge-related errors."""

    pass


class TemplateNotFoundError(JudgeError):
    """Raised when a judge template file cannot be found."""

    pass


class JudgeParseError(JudgeError):
    """Raised when the judge response cannot be parsed."""

    pass


# =============================================================================
# Judge Class
# =============================================================================


class Judge:
    """LLM-as-judge evaluator for pairwise and pointwise comparisons.

    This class handles template rendering, LLM calls, and response parsing
    for evaluating LLM responses using another LLM as a judge.
    """

    # Valid winner values
    VALID_WINNERS = {"a", "b", "tie", "invalid"}

    # Default template directory
    DEFAULT_TEMPLATE_DIR = Path("prompts")

    def __init__(self, config: JudgeConfig) -> None:
        """Initialize the judge with configuration.

        Args:
            config: The judge configuration.

        Raises:
            JudgeError: If the judge model cannot be initialized.
        """
        # Early exit: Validate config
        if not config.model_config:
            raise JudgeError("model_config is required")

        if config.temperature < 0 or config.temperature > 2:
            raise JudgeError("temperature must be between 0 and 2")

        if config.max_tokens <= 0:
            raise JudgeError("max_tokens must be positive")

        self._config = config
        self._provider = self._create_provider()

    def _create_provider(self) -> ModelProvider:
        """Create the model provider for the judge.

        Returns:
            Configured ModelProvider instance.

        Raises:
            JudgeError: If provider creation fails.
        """
        try:
            return ProviderFactory.create(self._config.model_config)
        except Exception as e:
            raise JudgeError(f"Failed to create judge provider: {e}") from e

    def evaluate_pairwise(
        self,
        response_a: str,
        response_b: str,
        prompt: str,
        task_description: str,
    ) -> JudgeResult:
        """Evaluate two responses in a pairwise comparison.

        This method randomly assigns which response is "A" or "B" to mitigate
        position bias in the judge model.

        Args:
            response_a: First response to evaluate.
            response_b: Second response to evaluate.
            prompt: The original prompt that generated both responses.
            task_description: Description of the task being evaluated.

        Returns:
            JudgeResult with winner, justification, and optional reasoning.

        Raises:
            JudgeError: If evaluation fails.
        """
        # Early exit: Validate inputs
        if not response_a or not response_a.strip():
            raise JudgeError("response_a cannot be empty")

        if not response_b or not response_b.strip():
            raise JudgeError("response_b cannot be empty")

        if not prompt or not prompt.strip():
            raise JudgeError("prompt cannot be empty")

        if not task_description or not task_description.strip():
            raise JudgeError("task_description cannot be empty")

        # Position randomization to mitigate bias
        # Randomly swap which response is "A" and "B"
        swap = random.choice([True, False])
        rendered_a = response_b if swap else response_a
        rendered_b = response_a if swap else response_b

        # Load and render template
        template = self._load_template(self._config.template_name)
        rendered_prompt = self._render_prompt(
            template,
            task_description=task_description,
            prompt=prompt,
            response_a=rendered_a,
            response_b=rendered_b,
            rubric=self._config.rubric,
        )

        # Call judge LLM
        raw_response = self._call_judge(rendered_prompt)

        # Parse response
        parsed = self._parse_response(raw_response)

        # Map winner back to original positions if needed
        winner = parsed.get("winner", "invalid")
        if swap and winner in {"a", "b"}:
            winner = "b" if winner == "a" else "a"

        # Build result with original position mapping stored in criteria if needed
        criteria_scores = parsed.get("criteria_scores")
        if swap and criteria_scores:
            criteria_scores = self._remap_criteria_positions(criteria_scores, swap)

        return JudgeResult(
            winner=winner,
            justification=parsed.get("justification", ""),
            reasoning=parsed.get("reasoning"),
            criteria_scores=criteria_scores,
            raw_response=raw_response,
        )

    def evaluate_pointwise(
        self,
        response: str,
        prompt: str,
        task_description: str,
    ) -> dict[str, Any]:
        """Evaluate a single response against a rubric.

        Args:
            response: The response to evaluate.
            prompt: The original prompt that generated the response.
            task_description: Description of the task being evaluated.

        Returns:
            Dictionary with criteria_scores, overall_score, and justification.

        Raises:
            JudgeError: If evaluation fails.
        """
        # Early exit: Validate inputs
        if not response or not response.strip():
            raise JudgeError("response cannot be empty")

        if not prompt or not prompt.strip():
            raise JudgeError("prompt cannot be empty")

        if not task_description or not task_description.strip():
            raise JudgeError("task_description cannot be empty")

        # Load pointwise template
        template = self._load_template("judge_pointwise.j2")
        rendered_prompt = self._render_prompt(
            template,
            task_description=task_description,
            prompt=prompt,
            response=response,
            rubric=self._config.rubric,
        )

        # Call judge LLM
        raw_response = self._call_judge(rendered_prompt)

        # Parse response - pointwise returns full dict
        return self._parse_response(raw_response)

    def _load_template(self, template_name: str) -> Template:
        """Load a Jinja2 template from the prompts directory.

        Args:
            template_name: Name of the template file.

        Returns:
            Jinja2 Template object.

        Raises:
            TemplateNotFoundError: If template file cannot be found.
        """
        # Try multiple possible template directories
        search_dirs = [
            self.DEFAULT_TEMPLATE_DIR,
            Path.cwd() / "prompts",
            Path(__file__).parent.parent.parent.parent / "prompts",
        ]

        for search_dir in search_dirs:
            template_path = search_dir / template_name
            if template_path.exists():
                try:
                    content = template_path.read_text(encoding="utf-8")
                    return Template(content)
                except Exception as e:
                    raise JudgeError(f"Failed to load template {template_name}: {e}") from e

        # Template not found in any search directory
        raise TemplateNotFoundError(
            f"Template '{template_name}' not found in any search directory. "
            f"Searched: {[str(d) for d in search_dirs]}"
        )

    def _render_prompt(self, template: Template, **kwargs: Any) -> str:
        """Render a Jinja2 template with the given variables.

        Args:
            template: Jinja2 Template object.
            **kwargs: Variables to pass to the template.

        Returns:
            Rendered prompt string.
        """
        # Filter out None values to avoid Jinja2 errors
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return template.render(**filtered_kwargs)

    def _call_judge(self, prompt: str) -> str:
        """Call the judge LLM with the rendered prompt.

        Args:
            prompt: The rendered prompt to send to the judge.

        Returns:
            Raw response text from the judge LLM.

        Raises:
            JudgeError: If the LLM call fails.
        """
        try:
            response = self._provider.generate(
                prompt=prompt,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
            return response.content
        except Exception as e:
            raise JudgeError(f"Judge LLM call failed: {e}") from e

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from the judge response.

        This method handles JSON parsing gracefully with fallback to
        regex-based extraction if JSON parsing fails.

        Args:
            response: Raw response text from the judge.

        Returns:
            Parsed dictionary from JSON.

        Raises:
            JudgeParseError: If parsing fails completely.
        """
        # Try direct JSON parsing first
        try:
            # Find JSON block in response (in case there's markdown wrapper)
            json_match = _extract_json_block(response)
            if json_match:
                return json.loads(json_match)
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Fallback: Try regex extraction
        extracted = _extract_json_with_regex(response)
        if extracted:
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass

        # Final fallback: Return raw response with error indication
        raise JudgeParseError(
            f"Failed to parse judge response as JSON. Response: {response[:500]}..."
        )

    def _remap_criteria_positions(
        self,
        criteria_scores: dict[str, Any],
        swapped: bool,
    ) -> dict[str, Any]:
        """Remap criterion positions from swapped evaluation.

        Args:
            criteria_scores: The criteria scores dictionary.
            swapped: Whether positions were swapped.

        Returns:
            Criteria scores with positions remapped if needed.
        """
        if not swapped or not criteria_scores:
            return criteria_scores

        remapped = {}
        for criterion, scores in criteria_scores.items():
            if isinstance(scores, dict):
                new_scores = {}
                for key, value in scores.items():
                    if key == "a":
                        new_scores["b"] = value
                    elif key == "b":
                        new_scores["a"] = value
                    else:
                        new_scores[key] = value
                remapped[criterion] = new_scores
            else:
                remapped[criterion] = scores

        return remapped


# =============================================================================
# Helper Functions (Module-level for purity)
# =============================================================================


def _extract_json_block(text: str) -> Optional[str]:
    """Extract JSON block from markdown-wrapped JSON.

    Args:
        text: Response text that may contain ```json ... ``` blocks.

    Returns:
        Extracted JSON string, or None if not found.
    """
    import re

    # Match ```json ... ``` or ``` ... ```
    pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return matches[0]

    return None


def _extract_json_with_regex(text: str) -> Optional[str]:
    """Extract JSON object using regex as last resort.

    Args:
        text: Response text potentially containing JSON.

    Returns:
        Extracted JSON string, or None if not found.
    """

    # Find the first { and last } to get the JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return None


# =============================================================================
# Convenience Factory
# =============================================================================


def create_judge(
    model_config: ModelConfig,
    rubric: Optional[Rubric] = None,
    template_name: str = "judge_pairwise_coot.j2",
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> Judge:
    """Convenience function to create a Judge instance.

    Args:
        model_config: The model configuration for the judge.
        rubric: Optional rubric for scoring.
        template_name: Template file name.
        temperature: LLM temperature.
        max_tokens: Max tokens for generation.

    Returns:
        Configured Judge instance.
    """
    config = JudgeConfig(
        model_config=model_config,
        rubric=rubric,
        template_name=template_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return Judge(config)
