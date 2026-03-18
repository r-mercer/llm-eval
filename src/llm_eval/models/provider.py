"""LLM provider integration using LiteLLM.

This module provides a unified interface for calling LLM models with support for
multiple providers (OpenAI, Anthropic, Ollama, LM Studio, etc.).
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import litellm
from litellm import acompletion, completion

from llm_eval.db.models import ModelConfig


# =============================================================================
# Custom Exceptions
# =============================================================================


class ProviderError(Exception):
    """Base exception for provider-related errors."""

    def __init__(self, message: str, model: Optional[str] = None):
        self.model = model
        super().__init__(message)


class AuthenticationError(ProviderError):
    """Raised when API authentication fails."""

    pass


class RateLimitError(ProviderError):
    """Raised when rate limit is exceeded."""

    pass


class InvalidConfigError(ProviderError):
    """Raised when model configuration is invalid."""

    pass


class ModelNotSupportedError(ProviderError):
    """Raised when model/provider is not supported."""

    pass


# =============================================================================
# Response Dataclass
# =============================================================================


@dataclass(frozen=True)
class Response:
    """Response from an LLM provider.

    This is an immutable dataclass representing the output from an LLM call.
    """

    content: str
    """The generated text content from the model."""

    model: str
    """The model identifier used for the request."""

    usage: dict = field(default_factory=dict)
    """Token usage information (prompt_tokens, completion_tokens, total_tokens)."""

    raw_response: Optional[dict] = field(default=None)
    """The full raw API response for debugging purposes."""

    @property
    def prompt_tokens(self) -> int:
        """Return the number of tokens in the prompt."""
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        """Return the number of tokens in the completion."""
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """Return the total number of tokens used."""
        return self.usage.get("total_tokens", 0)


# =============================================================================
# ModelProvider
# =============================================================================


class ModelProvider:
    """Unified interface for calling LLM models via LiteLLM.

    This class provides a clean abstraction over different LLM providers,
    handling API key resolution, base URLs, and parameter passing.
    """

    def __init__(self, model_config: ModelConfig):
        """Initialize the provider with a model configuration.

        Args:
            model_config: The model configuration from the database.

        Raises:
            InvalidConfigError: If the model configuration is invalid.
        """
        # Early exit: Validate required fields
        if not model_config.model:
            raise InvalidConfigError(
                "Model name is required",
                model=model_config.name,
            )

        if not model_config.provider:
            raise InvalidConfigError(
                "Provider is required",
                model=model_config.name,
            )

        self._config = model_config
        self._resolved_api_key: Optional[str] = None
        self._validate_and_resolve()

    def _validate_and_resolve(self) -> None:
        """Validate configuration and resolve API key.

        Parses the API key which can be either a direct key or an env var reference.
        This follows the Parse Don't Validate principle - we resolve once at the boundary.
        """
        # Resolve API key from env var if it's a reference like $OPENAI_API_KEY
        api_key = self._config.api_key
        if api_key and api_key.startswith("$"):
            env_var = api_key[1:]  # Remove the $ prefix
            resolved = os.environ.get(env_var)
            if not resolved:
                raise AuthenticationError(
                    f"Environment variable '{env_var}' is not set for API key",
                    model=self._config.name,
                )
            self._resolved_api_key = resolved
        else:
            self._resolved_api_key = api_key

    @property
    def model(self) -> str:
        """Return the model identifier."""
        return self._config.model

    @property
    def provider(self) -> str:
        """Return the provider name."""
        return self._config.provider

    @property
    def name(self) -> str:
        """Return the config name."""
        return self._config.name

    def _build_litellm_params(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Build parameters for LiteLLM call.

        Args:
            temperature: Override for temperature. Uses default if None.
            max_tokens: Override for max_tokens. Uses default if None.

        Returns:
            Dictionary of LiteLLM-compatible parameters.
        """
        params = {
            "model": self._config.model,
            "temperature": temperature if temperature is not None else self._config.default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._config.default_max_tokens,
        }

        # Add optional base_url for local models
        if self._config.base_url:
            params["base_url"] = self._config.base_url

        # Add API key if resolved
        if self._resolved_api_key:
            params["api_key"] = self._resolved_api_key

        return params

    def _handle_error(self, e: Exception) -> ProviderError:
        """Convert LiteLLM exceptions to our custom exceptions.

        This provides a clean, consistent error interface.
        """
        error_message = str(e)

        # Map LiteLLM errors to our custom exceptions
        if "authentication" in error_message.lower() or "api key" in error_message.lower():
            return AuthenticationError(f"Authentication failed: {error_message}", model=self._model_identifier)

        if "rate limit" in error_message.lower():
            return RateLimitError(f"Rate limit exceeded: {error_message}", model=self._model_identifier)

        if "context length" in error_message.lower():
            return ProviderError(f"Context length exceeded: {error_message}", model=self._model_identifier)

        return ProviderError(f"Provider error: {error_message}", model=self._model_identifier)

    @property
    def _model_identifier(self) -> str:
        """Return the full model identifier for error messages."""
        return f"{self._config.provider}/{self._config.model}"

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Response:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user prompt to send to the model.
            temperature: Sampling temperature (overrides default).
            max_tokens: Maximum tokens to generate (overrides default).

        Returns:
            Response object containing the generated content.

        Raises:
            ProviderError: If the API call fails.
        """
        # Early exit: Validate prompt
        if not prompt or not prompt.strip():
            raise ProviderError("Prompt cannot be empty", model=self._config.name)

        litellm_params = self._build_litellm_params(temperature, max_tokens)
        litellm_params["messages"] = [{"role": "user", "content": prompt}]

        try:
            response = completion(**litellm_params)
            return self._parse_response(response)
        except Exception as e:
            raise self._handle_error(e) from e

    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Response:
        """Generate a completion with a system prompt and user prompt.

        Args:
            system_prompt: The system prompt that sets the context/behavior.
            user_prompt: The user prompt with the actual request.
            temperature: Sampling temperature (overrides default).
            max_tokens: Maximum tokens to generate (overrides default).

        Returns:
            Response object containing the generated content.

        Raises:
            ProviderError: If the API call fails.
        """
        # Early exit: Validate prompts
        if not system_prompt or not system_prompt.strip():
            raise ProviderError("System prompt cannot be empty", model=self._config.name)

        if not user_prompt or not user_prompt.strip():
            raise ProviderError("User prompt cannot be empty", model=self._config.name)

        litellm_params = self._build_litellm_params(temperature, max_tokens)
        litellm_params["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = completion(**litellm_params)
            return self._parse_response(response)
        except Exception as e:
            raise self._handle_error(e) from e

    async def generate_async(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Response:
        """Generate a completion asynchronously.

        Args:
            prompt: The user prompt to send to the model.
            temperature: Sampling temperature (overrides default).
            max_tokens: Maximum tokens to generate (overrides default).

        Returns:
            Response object containing the generated content.

        Raises:
            ProviderError: If the API call fails.
        """
        # Early exit: Validate prompt
        if not prompt or not prompt.strip():
            raise ProviderError("Prompt cannot be empty", model=self._config.name)

        litellm_params = self._build_litellm_params(temperature, max_tokens)
        litellm_params["messages"] = [{"role": "user", "content": prompt}]

        try:
            response = await acompletion(**litellm_params)
            return self._parse_response(response)
        except Exception as e:
            raise self._handle_error(e) from e

    async def generate_with_system_async(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Response:
        """Generate a completion with system and user prompts asynchronously.

        Args:
            system_prompt: The system prompt that sets the context/behavior.
            user_prompt: The user prompt with the actual request.
            temperature: Sampling temperature (overrides default).
            max_tokens: Maximum tokens to generate (overrides default).

        Returns:
            Response object containing the generated content.

        Raises:
            ProviderError: If the API call fails.
        """
        # Early exit: Validate prompts
        if not system_prompt or not system_prompt.strip():
            raise ProviderError("System prompt cannot be empty", model=self._config.name)

        if not user_prompt or not user_prompt.strip():
            raise ProviderError("User prompt cannot be empty", model=self._config.name)

        litellm_params = self._build_litellm_params(temperature, max_tokens)
        litellm_params["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await acompletion(**litellm_params)
            return self._parse_response(response)
        except Exception as e:
            raise self._handle_error(e) from e

    def _parse_response(self, response) -> Response:
        """Parse LiteLLM response into our Response dataclass.

        This follows Atomic Predictability - the function returns a new
        immutable Response object.
        """
        # Extract content from the first choice
        choices = response.choices
        if not choices:
            raise ProviderError("No choices in response", model=self._model_identifier)

        content = choices[0].message.content or ""

        # Extract usage information
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        # Get model identifier from response or use our config
        model = response.model or self._model_identifier

        # Store raw response for debugging
        raw = response.model_dump() if hasattr(response, "model_dump") else {}

        return Response(
            content=content,
            model=model,
            usage=usage,
            raw_response=raw,
        )


# =============================================================================
# ProviderFactory
# =============================================================================


class ProviderFactory:
    """Factory for creating ModelProvider instances.

    This factory handles different provider types and ensures consistent
    configuration resolution.
    """

    # Supported providers mapped to their LiteLLM model prefixes
    SUPPORTED_PROVIDERS = {
        "openai": "openai",
        "anthropic": "anthropic",
        "azure": "azure/",
        "ollama": "ollama/",
        "lmstudio": "lmstudio/",
        "local": "openai/",  # Local models often use OpenAI-compatible API
        "cohere": "cohere",
        "mistral": "mistral",
        "bedrock": "bedrock/",
        "vertex": "vertex_ai",
        "sagemaker": "sagemaker",
    }

    @staticmethod
    def create(model_config: ModelConfig) -> ModelProvider:
        """Create a ModelProvider from a ModelConfig.

        Args:
            model_config: The model configuration from the database.

        Returns:
            A configured ModelProvider instance.

        Raises:
            InvalidConfigError: If the configuration is invalid.
            ModelNotSupportedError: If the provider is not supported.
        """
        # Early exit: Validate provider
        provider = model_config.provider.lower() if model_config.provider else ""

        if not provider:
            raise InvalidConfigError(
                "Provider is required",
                model=model_config.name,
            )

        if provider not in ProviderFactory.SUPPORTED_PROVIDERS:
            raise ModelNotSupportedError(
                f"Provider '{provider}' is not supported. "
                f"Supported providers: {', '.join(ProviderFactory.SUPPORTED_PROVIDERS.keys())}",
                model=model_config.name,
            )

        # Validate that we have at least a model name
        if not model_config.model:
            raise InvalidConfigError(
                "Model name is required",
                model=model_config.name,
            )

        return ModelProvider(model_config)

    @staticmethod
    def is_supported(provider: str) -> bool:
        """Check if a provider is supported.

        Args:
            provider: The provider name to check.

        Returns:
            True if the provider is supported, False otherwise.
        """
        return provider.lower() in ProviderFactory.SUPPORTED_PROVIDERS

    @staticmethod
    def supported_providers() -> list[str]:
        """Get list of supported provider names.

        Returns:
            List of supported provider names.
        """
        return list(ProviderFactory.SUPPORTED_PROVIDERS.keys())


# =============================================================================
# Convenience Functions
# =============================================================================


def create_provider(model_config: ModelConfig) -> ModelProvider:
    """Convenience function to create a provider.

    This is a simple wrapper around ProviderFactory.create().

    Args:
        model_config: The model configuration from the database.

    Returns:
        A configured ModelProvider instance.
    """
    return ProviderFactory.create(model_config)
