"""
llmgate.completion
~~~~~~~~~~~~~~~~~~
Main entry points for llmgate.

Public API:
    completion(model, messages, **kwargs)   -> CompletionResponse  (stream=False)
                                           -> Iterator[StreamChunk] (stream=True)
    acompletion(model, messages, **kwargs)  -> CompletionResponse  (stream=False, async)
                                           -> AsyncIterator[StreamChunk] (stream=True, async)

Provider resolution order:
1. If ``provider`` kwarg is given, use that provider by name.
2. Otherwise, iterate registered providers and find one whose ``supports()``
   returns True for the given model string.
3. Raise ``ModelNotFoundError`` if no provider matched.

Provider instances are cached on the ``ProviderRegistry`` so SDK clients aren't
re-created for every call.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Literal, Union, overload

if TYPE_CHECKING:
    from llmgate.middleware.base import BaseMiddleware

from llmgate.base import BaseProvider
from llmgate.exceptions import ConfigError, ModelNotFoundError
from llmgate.providers.anthropic import AnthropicProvider
from llmgate.providers.gemini import GeminiProvider
from llmgate.providers.groq import GroqProvider
from llmgate.providers.openai import OpenAIProvider
from llmgate.types import (
    CompletionRequest, CompletionResponse, Message, StreamChunk, ToolDefinition,
)


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


# Alias used by LLMGate client
_get_or_create_provider = _get_provider


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
    known_keys = {"max_tokens", "temperature", "top_p", "tools", "tool_choice",
                  "extra_kwargs", "middleware", "provider", "api_key"}
    extra = {k: v for k, v in kwargs.items() if k not in known_keys}
    # Normalise tools: accept list of ToolDefinition or plain dicts
    raw_tools = kwargs.get("tools")
    tools: list[ToolDefinition] | None = None
    if raw_tools is not None:
        tools = [
            ToolDefinition(**t) if isinstance(t, dict) else t
            for t in raw_tools
        ]
    return CompletionRequest(
        model=model,
        messages=_normalise_messages(messages),
        max_tokens=kwargs.get("max_tokens"),
        temperature=kwargs.get("temperature"),
        top_p=kwargs.get("top_p"),
        stream=stream,
        tools=tools,
        tool_choice=kwargs.get("tool_choice"),
        extra_kwargs=extra,
    )


# ---------------------------------------------------------------------------
# Public API — completion()
# ---------------------------------------------------------------------------

_MsgList = list[Union[dict[str, str], Message]]


@overload
def completion(
    model: str,
    messages: _MsgList,
    *,
    provider: str | None = ...,
    api_key: str | None = ...,
    stream: Literal[True],
    middleware: list[BaseMiddleware] | None = ...,
    **kwargs: Any,
) -> Iterator[StreamChunk]: ...


@overload
def completion(
    model: str,
    messages: _MsgList,
    *,
    provider: str | None = ...,
    api_key: str | None = ...,
    stream: Literal[False] = ...,
    middleware: list[BaseMiddleware] | None = ...,
    **kwargs: Any,
) -> CompletionResponse: ...


def completion(
    model: str,
    messages: _MsgList,
    *,
    provider: str | None = None,
    api_key: str | None = None,
    stream: bool = False,
    middleware: list[BaseMiddleware] | None = None,
    **kwargs: Any,
) -> CompletionResponse | Iterator[StreamChunk]:
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
        stream:   If True, returns an ``Iterator[StreamChunk]`` instead of a
                  ``CompletionResponse``. Iterate the result to receive deltas.
        **kwargs: Extra parameters forwarded to the provider
                  (``max_tokens``, ``temperature``, ``top_p``, etc.).

    Returns:
        :class:`~llmgate.types.CompletionResponse` when ``stream=False`` (default).
        ``Iterator[StreamChunk]`` when ``stream=True``.

    Raises:
        :class:`~llmgate.exceptions.ModelNotFoundError`: Unknown model.
        :class:`~llmgate.exceptions.ProviderError`: Provider returned an error.
    """
    provider_inst = _get_provider(model, provider, api_key)
    request = _build_request(model, messages, stream, kwargs)

    if middleware:
        from llmgate.gate import _build_sync_chain  # noqa: PLC0415
        def _inner(req: CompletionRequest) -> CompletionResponse:
            return provider_inst.complete(req)
        if stream:
            _mw_chain = _inner  # streaming ignores middleware for now
            return provider_inst.stream(request)
        chain = _build_sync_chain(middleware, _inner)
        return chain(request)

    if stream:
        return provider_inst.stream(request)
    return provider_inst.complete(request)


# ---------------------------------------------------------------------------
# Public API — acompletion()
# ---------------------------------------------------------------------------


@overload
async def acompletion(
    model: str,
    messages: _MsgList,
    *,
    provider: str | None = ...,
    api_key: str | None = ...,
    stream: Literal[True],
    middleware: list[BaseMiddleware] | None = ...,
    **kwargs: Any,
) -> AsyncIterator[StreamChunk]: ...


@overload
async def acompletion(
    model: str,
    messages: _MsgList,
    *,
    provider: str | None = ...,
    api_key: str | None = ...,
    stream: Literal[False] = ...,
    middleware: list[BaseMiddleware] | None = ...,
    **kwargs: Any,
) -> CompletionResponse: ...


async def acompletion(
    model: str,
    messages: _MsgList,
    *,
    provider: str | None = None,
    api_key: str | None = None,
    stream: bool = False,
    middleware: list[BaseMiddleware] | None = None,
    **kwargs: Any,
) -> CompletionResponse | AsyncIterator[StreamChunk]:
    """
    Perform an asynchronous chat completion.

    Same signature as :func:`completion` but returns a coroutine.
    When ``stream=True``, returns an ``AsyncIterator[StreamChunk]``.
    """
    provider_inst = _get_provider(model, provider, api_key)
    request = _build_request(model, messages, stream, kwargs)

    if middleware:
        from llmgate.gate import _build_async_chain  # noqa: PLC0415
        async def _inner(req: CompletionRequest) -> CompletionResponse:
            return await provider_inst.acomplete(req)
        if stream:
            return provider_inst.astream(request)
        chain = _build_async_chain(middleware, _inner)
        return await chain(request)

    if stream:
        return provider_inst.astream(request)
    return await provider_inst.acomplete(request)
