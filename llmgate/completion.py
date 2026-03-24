"""
llmgate.completion
~~~~~~~~~~~~~~~~~~
Main entry points for llmgate.

Public API:
    completion(model, messages, **kwargs)   -> CompletionResponse
    acompletion(model, messages, **kwargs)  -> CompletionResponse (async)

Provider resolution order:
1. If ``provider`` kwarg is given, use that provider by name.
2. Otherwise, iterate registered providers and find one whose ``supports()``
   returns True for the given model string.
3. Raise ``ModelNotFoundError`` if no provider matched.

Provider instances are cached on the ``ProviderRegistry`` so SDK clients aren't
re-created for every call.
"""
from __future__ import annotations

from typing import Any

from llmgate.base import BaseProvider
from llmgate.exceptions import ConfigError, ModelNotFoundError, StreamingNotSupported
from llmgate.providers.anthropic import AnthropicProvider
from llmgate.providers.gemini import GeminiProvider
from llmgate.providers.groq import GroqProvider
from llmgate.providers.openai import OpenAIProvider
from llmgate.types import CompletionRequest, CompletionResponse, Message


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Ordered list of provider *classes* used for model-prefix routing.
_PROVIDER_CLASSES: list[type[BaseProvider]] = [
    OpenAIProvider,
    GeminiProvider,
    AnthropicProvider,
    GroqProvider,
]

#: Map of provider name -> class for explicit ``provider=`` override.
_PROVIDER_NAME_MAP: dict[str, type[BaseProvider]] = {
    cls.name: cls for cls in _PROVIDER_CLASSES  # type: ignore[attr-defined]
}

#: Cache of instantiated providers keyed by (provider_name, api_key or None)
#: so we reuse SDK clients across calls.
_provider_cache: dict[tuple[str, str | None], BaseProvider] = {}


def _get_provider(
    model: str,
    provider_name: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> BaseProvider:
    """Resolve and (lazily) instantiate a provider."""
    if provider_name:
        cls = _PROVIDER_NAME_MAP.get(provider_name.lower())
        if cls is None:
            raise ConfigError(
                f"Unknown provider '{provider_name}'. "
                f"Available: {list(_PROVIDER_NAME_MAP)}"
            )
    else:
        cls = next((c for c in _PROVIDER_CLASSES if c.supports(model)), None)  # type: ignore[attr-defined]
        if cls is None:
            raise ModelNotFoundError(model)

    cache_key = (cls.name, api_key)  # type: ignore[attr-defined]
    if cache_key not in _provider_cache:
        _provider_cache[cache_key] = cls(api_key=api_key, **kwargs)  # type: ignore[call-arg]
    return _provider_cache[cache_key]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_messages(messages: list[dict[str, str] | Message]) -> list[Message]:
    return [
        Message(**m) if isinstance(m, dict) else m
        for m in messages
    ]


def _build_request(
    model: str,
    messages: list[dict[str, str] | Message],
    stream: bool,
    kwargs: dict[str, Any],
) -> CompletionRequest:
    known_keys = {"max_tokens", "temperature", "top_p", "extra_kwargs"}
    extra = {k: v for k, v in kwargs.items() if k not in known_keys}
    return CompletionRequest(
        model=model,
        messages=_normalise_messages(messages),
        max_tokens=kwargs.get("max_tokens"),
        temperature=kwargs.get("temperature"),
        top_p=kwargs.get("top_p"),
        stream=stream,
        extra_kwargs=extra,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def completion(
    model: str,
    messages: list[dict[str, str] | Message],
    *,
    provider: str | None = None,
    api_key: str | None = None,
    stream: bool = False,
    **kwargs: Any,
) -> CompletionResponse:
    """
    Perform a synchronous chat completion.

    Args:
        model:    Model name. Use ``groq/`` prefix for Groq models.
        messages: List of message dicts (``{"role": ..., "content": ...}``)
                  or :class:`~llmgate.types.Message` instances.
        provider: Force a specific provider by name (``"openai"``, ``"gemini"``,
                  ``"anthropic"``, ``"groq"``). Optional — auto-detected otherwise.
        api_key:  Override the API key for this call. Defaults to the relevant
                  environment variable.
        stream:   Not yet supported — raises ``StreamingNotSupported``.
        **kwargs: Extra parameters forwarded to the provider
                  (``max_tokens``, ``temperature``, ``top_p``, etc.).

    Returns:
        :class:`~llmgate.types.CompletionResponse`

    Raises:
        :class:`~llmgate.exceptions.ModelNotFoundError`: Unknown model.
        :class:`~llmgate.exceptions.StreamingNotSupported`: ``stream=True``.
        :class:`~llmgate.exceptions.ProviderError`: Provider returned an error.
    """
    if stream:
        raise StreamingNotSupported()

    provider_inst = _get_provider(model, provider, api_key)
    request = _build_request(model, messages, stream, kwargs)
    return provider_inst.complete(request)


async def acompletion(
    model: str,
    messages: list[dict[str, str] | Message],
    *,
    provider: str | None = None,
    api_key: str | None = None,
    stream: bool = False,
    **kwargs: Any,
) -> CompletionResponse:
    """
    Perform an asynchronous chat completion.

    Same signature as :func:`completion` but returns a coroutine.
    """
    if stream:
        raise StreamingNotSupported()

    provider_inst = _get_provider(model, provider, api_key)
    request = _build_request(model, messages, stream, kwargs)
    return await provider_inst.acomplete(request)
