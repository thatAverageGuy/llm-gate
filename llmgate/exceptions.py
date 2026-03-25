"""
llmgate.exceptions
~~~~~~~~~~~~~~~~~~
Layered exception hierarchy for llmgate.

All exceptions inherit from ``LLMGateError`` so callers can catch broadly or
narrow down to specific error cases.

Hierarchy:
    LLMGateError
    ├── ProviderError           upstream provider rejected the call
    │   ├── AuthError           401 / invalid API key
    │   ├── RateLimitError      429 / quota exceeded
    │   └── ProviderAPIError    other 4xx/5xx from the provider
    ├── ModelNotFoundError      unknown model string (no provider matched)
    ├── ConfigError             missing env var / bad config
    ├── StreamingNotSupported   stream=True before streaming is implemented
    └── EmbeddingsNotSupported  provider does not offer an embeddings API
"""
from __future__ import annotations


class LLMGateError(Exception):
    """Base class for all llmgate errors."""

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


# ---------------------------------------------------------------------------
# Provider errors
# ---------------------------------------------------------------------------


class ProviderError(LLMGateError):
    """The upstream provider returned an error response."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, provider=provider)
        self.status_code = status_code


class AuthError(ProviderError):
    """Authentication failed — check your API key."""


class RateLimitError(ProviderError):
    """Rate limit or quota exceeded."""


class ProviderAPIError(ProviderError):
    """Unexpected 4xx/5xx from the provider."""


# ---------------------------------------------------------------------------
# Routing / config errors
# ---------------------------------------------------------------------------


class ModelNotFoundError(LLMGateError):
    """No registered provider supports the requested model string."""

    def __init__(self, model: str) -> None:
        super().__init__(
            f"No provider found for model '{model}'. "
            "For Groq models use the 'groq/' prefix, e.g. 'groq/llama-3.1-8b-instant'."
        )
        self.model = model


class ConfigError(LLMGateError):
    """Missing or invalid configuration (e.g. absent API key env var)."""


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class StreamingNotSupported(LLMGateError):
    """Streaming is not yet implemented in this version of llmgate."""

    def __init__(self) -> None:
        super().__init__(
            "Streaming (stream=True) is not yet supported in llmgate v0.1. "
            "Track progress at https://github.com/thatAverageGuy/llm-gate."
        )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class EmbeddingsNotSupported(LLMGateError):
    """The requested provider does not offer an embeddings API."""

    def __init__(self, provider: str) -> None:
        super().__init__(
            f"Provider '{provider}' does not support embeddings. "
            "Use OpenAI, Gemini, Azure, Cohere, Mistral, Ollama, or Bedrock.",
            provider=provider,
        )
