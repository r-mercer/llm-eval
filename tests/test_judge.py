"""Tests for judge parsing logic."""

import dataclasses
import pytest
from llm_eval.eval.judge import (
    JudgeConfig,
    JudgeResult,
    _extract_json_block,
    _extract_json_with_regex,
)


class TestJsonExtraction:
    """Test suite for JSON extraction helpers."""

    def test_extract_json_block_with_markdown(self) -> None:
        """Test extracting JSON from markdown code blocks."""
        text = """Here is my response:

```json
{"winner": "a", "justification": "Test"}
```

That's all."""
        result = _extract_json_block(text)
        assert result == '{"winner": "a", "justification": "Test"}'

    def test_extract_json_block_without_markdown(self) -> None:
        """Test extracting JSON without markdown returns None.

        _extract_json_block only handles markdown code blocks,
        plain JSON should be handled by json.loads directly.
        """
        text = '{"winner": "b", "reasoning": "Step 1..."}'
        result = _extract_json_block(text)
        # Plain JSON without markdown returns None
        assert result is None

    def test_extract_json_block_no_json(self) -> None:
        """Test extraction when no JSON present."""
        text = "Just some plain text"
        result = _extract_json_block(text)
        assert result is None

    def test_extract_json_with_regex(self) -> None:
        """Test regex-based JSON extraction."""
        text = 'Some text before {"key": "value"} some text after'
        result = _extract_json_with_regex(text)
        assert result == '{"key": "value"}'

    def test_extract_json_with_regex_malformed(self) -> None:
        """Test regex extraction with incomplete JSON."""
        text = 'Start { "key": "value" End'
        result = _extract_json_with_regex(text)
        assert result is None

    def test_extract_json_with_regex_no_braces(self) -> None:
        """Test regex extraction with no braces."""
        text = "No JSON here"
        result = _extract_json_with_regex(text)
        assert result is None


class TestJudgeConfig:
    """Test suite for JudgeConfig dataclass."""

    def test_config_with_defaults(self) -> None:
        """Test JudgeConfig with default values."""
        from llm_eval.db.models import ModelConfig

        model = ModelConfig(
            name="test",
            provider="openai",
            model="gpt-4",
        )

        config = JudgeConfig(model_config=model)
        assert config.model_config == model
        assert config.template_name == "judge_pairwise_coot.j2"
        assert config.temperature == 0.0
        assert config.max_tokens == 4096
        assert config.rubric is None

    def test_config_with_custom_values(self) -> None:
        """Test JudgeConfig with custom values."""
        from llm_eval.db.models import ModelConfig, Rubric

        model = ModelConfig(
            name="test",
            provider="openai",
            model="gpt-4",
        )
        rubric = Rubric(name="test", description="Test", weights={"a": 1.0})

        config = JudgeConfig(
            model_config=model,
            rubric=rubric,
            template_name="custom.j2",
            temperature=0.5,
            max_tokens=2048,
        )
        assert config.template_name == "custom.j2"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.rubric == rubric


class TestJudgeResult:
    """Test suite for JudgeResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating a JudgeResult."""
        result = JudgeResult(
            winner="a",
            justification="Model A was better",
            raw_response="...",
            reasoning="Step-by-step analysis...",
            criteria_scores={"accuracy": {"a": 8, "b": 6}},
        )

        assert result.winner == "a"
        assert result.justification == "Model A was better"
        assert result.reasoning == "Step-by-step analysis..."
        assert result.criteria_scores == {"accuracy": {"a": 8, "b": 6}}

    def test_result_immutable(self) -> None:
        """Test that JudgeResult is immutable (frozen dataclass)."""
        result = JudgeResult(
            winner="a",
            justification="Test",
            raw_response="...",
        )

        # Frozen dataclass should prevent attribute assignment
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            result.winner = "b"  # type: ignore[reportAttributeAccessIssue]

    def test_result_minimal(self) -> None:
        """Test creating JudgeResult with minimal fields."""
        result = JudgeResult(
            winner="tie",
            justification="Equal quality",
            raw_response="...",
        )

        assert result.winner == "tie"
        assert result.reasoning is None
        assert result.criteria_scores is None
