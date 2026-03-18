"""Configuration loader for evaluation settings."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from llm_eval.db.models import ModelConfig, Rubric


# =============================================================================
# EvaluationConfig - Main configuration container
# =============================================================================


@dataclass
class EvaluationConfig:
    """Complete evaluation configuration.
    
    Contains all models, rubrics, and default settings needed
    to run an evaluation.
    """
    
    models: list[ModelConfig] = field(default_factory=list)
    rubrics: list[Rubric] = field(default_factory=list)
    default_judge_model: str = ""
    default_rubric: str = ""


# =============================================================================
# Helper Functions
# =============================================================================


def resolve_api_key(api_key: Optional[str]) -> Optional[str]:
    """Resolve API key from direct value or env:VAR_NAME format.
    
    Handles two formats:
    - Direct value: "sk-12345" -> returns "sk-12345"
    - Environment variable: "env:OPENAI_API_KEY" -> returns os.environ["OPENAI_API_KEY"]
    """
    # Early exit for None or empty
    if not api_key:
        return None
    
    # Early exit for direct value (no env: prefix)
    if not api_key.startswith("env:"):
        return api_key
    
    # Parse env:VAR_NAME format
    var_name = api_key[4:]  # Remove "env:" prefix
    env_value = os.environ.get(var_name)
    
    # Fail fast if environment variable not found
    if env_value is None:
        raise ValueError(
            f"Environment variable '{var_name}' referenced in api_key not found"
        )
    
    return env_value


def load_yaml_file(path: str) -> dict:
    """Load and parse a YAML file.
    
    Fails fast if the file does not exist or cannot be parsed.
    """
    file_path = Path(path)
    
    # Fail fast if file does not exist
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e


# =============================================================================
# Model Configuration Loading
# =============================================================================


def load_models_config(path: str) -> list[ModelConfig]:
    """Load model configurations from YAML file.
    
    Expected format:
        models:
          - name: gpt-4o
            provider: openai
            model: gpt-4o
            api_key: env:OPENAI_API_KEY
            base_url: (optional)
    
    Args:
        path: Path to the models YAML file
        
    Returns:
        List of ModelConfig objects
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the YAML is invalid or required fields are missing
    """
    data = load_yaml_file(path)
    
    # Early exit for empty models list
    models_data = data.get("models", [])
    if not models_data:
        return []
    
    models = []
    for model_data in models_data:
        # Fail fast on missing required fields
        missing = [f for f in ("name", "provider", "model") if not model_data.get(f)]
        if missing:
            raise ValueError(
                f"Model configuration missing required fields: {missing}"
            )
        
        # Resolve API key if present
        raw_api_key = model_data.get("api_key")
        resolved_api_key = resolve_api_key(raw_api_key)
        
        model = ModelConfig(
            name=model_data["name"],
            provider=model_data["provider"],
            model=model_data["model"],
            api_key=resolved_api_key,
            base_url=model_data.get("base_url"),
            default_temperature=model_data.get("default_temperature", 0.0),
            default_max_tokens=model_data.get("default_max_tokens", 4096),
            is_active=model_data.get("is_active", True),
        )
        models.append(model)
    
    return models


# =============================================================================
# Rubric Loading
# =============================================================================


def load_rubric(path: str) -> Rubric:
    """Load a single rubric from YAML file.
    
    Expected format:
        name: quality
        description: General quality assessment
        weights:
          accuracy: 0.4
          clarity: 0.3
          safety: 0.3
    
    Args:
        path: Path to the rubric YAML file
        
    Returns:
        Rubric object
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the YAML is invalid or required fields are missing
    """
    data = load_yaml_file(path)
    
    # Fail fast on missing required fields
    if not data.get("name"):
        raise ValueError("Rubric configuration missing required field: name")
    if not data.get("description"):
        raise ValueError("Rubric configuration missing required field: description")
    if not data.get("weights"):
        raise ValueError("Rubric configuration missing required field: weights")
    
    # Parse weights - ensure they're floats
    weights = data["weights"]
    if not isinstance(weights, dict):
        raise ValueError("Rubric 'weights' must be a dictionary")
    
    parsed_weights = {}
    for criterion, weight in weights.items():
        try:
            parsed_weights[str(criterion)] = float(weight)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid weight for criterion '{criterion}': {weight}"
            ) from e
    
    rubric = Rubric(
        name=data["name"],
        description=data["description"],
        weights=parsed_weights,
        criteria_details=data.get("criteria_details"),
    )
    
    return rubric


def load_all_rubrics(rubrics_dir: str) -> list[Rubric]:
    """Load all rubrics from a directory.
    
    Args:
        rubrics_dir: Directory containing YAML rubric files
        
    Returns:
        List of Rubric objects
    """
    dir_path = Path(rubrics_dir)
    
    # Early exit if directory doesn't exist
    if not dir_path.exists():
        return []
    
    # Find all YAML files
    rubric_files = sorted(dir_path.glob("*.yaml"))
    
    rubrics = []
    for rubric_file in rubric_files:
        rubric = load_rubric(str(rubric_file))
        rubrics.append(rubric)
    
    return rubrics


# =============================================================================
# Full Configuration Loading
# =============================================================================


def load_config(config_dir: str) -> EvaluationConfig:
    """Load complete evaluation configuration from a directory.
    
    Expected directory structure:
        config_dir/
          models.yaml
          rubrics/
            default.yaml
            quality.yaml
          settings.yaml (optional)
    
    Args:
        config_dir: Path to the configuration directory
        
    Returns:
        EvaluationConfig containing all models, rubrics, and defaults
    """
    base_path = Path(config_dir)
    
    # Load models
    models_path = base_path / "models.yaml"
    models = load_models_config(str(models_path)) if models_path.exists() else []
    
    # Load rubrics
    rubrics_path = base_path / "rubrics"
    rubrics = load_all_rubrics(str(rubrics_path)) if rubrics_path.exists() else []
    
    # Load settings (optional)
    settings_path = base_path / "settings.yaml"
    settings = load_yaml_file(str(settings_path)) if settings_path.exists() else {}
    
    # Extract defaults from settings or use first available
    default_judge_model = settings.get("default_judge_model", "")
    default_rubric = settings.get("default_rubric", "")
    
    # Auto-select defaults if not specified
    if not default_judge_model and models:
        default_judge_model = models[0].name
    if not default_rubric and rubrics:
        default_rubric = rubrics[0].name
    
    config = EvaluationConfig(
        models=models,
        rubrics=rubrics,
        default_judge_model=default_judge_model,
        default_rubric=default_rubric,
    )
    
    return config


# =============================================================================
# Rubric Saving
# =============================================================================


def save_rubric(rubric: Rubric, path: str) -> None:
    """Save a rubric to a YAML file.
    
    Args:
        rubric: Rubric object to save
        path: Destination file path
        
    Raises:
        ValueError: If the rubric has invalid data
    """
    # Fail fast if rubric has no name
    if not rubric.name:
        raise ValueError("Cannot save rubric without a name")
    
    # Prepare data for serialization
    data = {
        "name": rubric.name,
        "description": rubric.description,
        "weights": rubric.weights,
    }
    
    # Include optional fields if present
    if rubric.criteria_details:
        data["criteria_details"] = rubric.criteria_details
    
    # Ensure parent directory exists
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write YAML with nice formatting
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
