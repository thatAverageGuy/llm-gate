"""
llmgate.completion
~~~~~~~~~~~~~~~~~~
Main entry points for llmgate.

Public API:
    completion(model, messages, **kwargs)   -> CompletionResponse  (stream=False)
                                           -> Iterator[StreamChunk] (stream=True)
    acompletion(model, messages, **kwargs)  -> CompletionResponse  (stream=False, async)
                                           -> AsyncIterator[StreamChunk] (stream=True, async)

``model`` accepts either a single string *or* a list of strings.  When a list
is passed, the library tries each model in order and falls back automatically
on ``RateLimitError``, ``ProviderAPIError``, and ``AuthError`` (configurable
via the ``fallback_on`` parameter).  The first successful response is returned
with ``CompletionResponse.fallback_attempts`` populated.

Provider resolution order (single-model path):
1. If ``provider`` kwarg is given, use that provider by name.
2. Otherwise, iterate registered providers and find one whose ``supports()``
   returns True for the given model string.
3. Raise ``ModelNotFoundError`` if no provider matched.

Provider instances are cached on the ``ProviderRegistry`` so SDK clients aren't
re-created for every call.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Literal, TypeVar, Union, overload

if TYPE_CHECKING:
    from llmgate.middleware.base import BaseMiddleware

from llmgate.base import BaseProvider
from llmgate.exceptions import AuthError, ConfigError, ModelNotFoundError, ProviderAPIError, RateLimitError


# Core providers (always available — hard deps)
from llmgate.providers.anthropic import AnthropicProvider
from llmgate.providers.gemini import GeminiProvider
from llmgate.providers.groq import GroqProvider
from llmgate.providers.openai import OpenAIProvider

from llmgate.types import (
    CompletionRequest, CompletionResponse, Message, StreamChunk, ToolDefinition,
)

T = TypeVar("T")

#: Default exception types that trigger a fallback to the next model.
_DEFAULT_FALLBACK_ON: tuple[type[Exception], ...] = (
    RateLimitError,
    ProviderAPIError,
    AuthError,
)


# ---------------------------------------------------------------------------
# Registry — optional providers loaded lazily
# ---------------------------------------------------------------------------

#: (routing_prefix, module_path, class_name) for optional providers.
#: Loaded lazily so a missing SDK never breaks `import llmgate`.
_OPTIONAL_PROVIDERS: list[tuple[str, str, str]] = [
    ("mistral/", "llmgate.providers.mistral", "MistralProvider"),
    ("cohere/",  "llmgate.providers.cohere",  "CohereProvider"),
    ("azure/",   "llmgate.providers.azure",   "AzureOpenAIProvider"),
    ("bedrock/", "llmgate.providers.bedrock", "BedrockProvider"),
    ("ollama/",  "llmgate.providers.ollama",  "OllamaProvider"),
]

_optional_provider_cache: dict[str, type[BaseProvider]] = {}


def _get_optional_provider_class(prefix: str) -> type[BaseProvider] | None:
    """Lazily import and return an optional provider class by routing prefix."""
    if prefix in _optional_provider_cache:
        return _optional_provider_cache[prefix]
    for p, module_path, class_name in _OPTIONAL_PROVIDERS:
        if p == prefix:
            import importlib  # noqa: PLC0415
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            _optional_provider_cache[prefix] = cls
            return cls
    return None


#: Core provider classes (eager) — always installed
_CORE_PROVIDER_CLASSES: list[type[BaseProvider]] = [
    OpenAIProvider,
    GeminiProvider,
    AnthropicProvider,
    GroqProvider,
]

#: Map of provider name → class for explicit ``provider=`` override (core providers only at load time).
_PROVIDER_NAME_MAP: dict[str, type[BaseProvider]] = {
    cls.name: cls for cls in _CORE_PROVIDER_CLASSES  # type: ignore[attr-defined]
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
        # Try core map first, then optional providers by name
        cls = _PROVIDER_NAME_MAP.get(provider_name.lower())
        if cls is None:
            # Try finding the optional provider by name
            for _prefix, _module, _cls_name in _OPTIONAL_PROVIDERS:
                _provider_name = _cls_name.lower().replace("provider", "").replace("openai", "azure").rstrip()
                # Match by the known name keys
                pass
            # Build up full name map including optional ones on demand
            for _pfx, _mod, _clsname in _OPTIONAL_PROVIDERS:
                import importlib  # noqa: PLC0415
                try:
                    _mod_obj = importlib.import_module(_mod)
                    _opt_cls = getattr(_mod_obj, _clsname)
                    if _opt_cls.name == provider_name.lower():  # type: ignore[attr-defined]
                        cls = _opt_cls
                        break
                except ImportError:
                    continue
        if cls is None:
            available = list(_PROVIDER_NAME_MAP) + [p for _, _, p in _OPTIONAL_PROVIDERS]
            raise ConfigError(
                f"Unknown provider '{provider_name}'. "
                f"Available: {available}"
            )
    else:
        # Auto-detect: try core providers first
        cls = next((c for c in _CORE_PROVIDER_CLASSES if c.supports(model)), None)  # type: ignore[attr-defined]
        if cls is None:
            # Try optional providers by prefix match
            for prefix, module_path, class_name in _OPTIONAL_PROVIDERS:
                if model.startswith(prefix):
                    cls = _get_optional_provider_class(prefix)
                    break
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
                  "extra_kwargs", "middleware", "provider", "api_key", "response_format",
                  "stream_fallback_mode", "stream_resume_prompt"}
    extra = {k: v for k, v in kwargs.items() if k not in known_keys}
    # Normalise tools: accept list of ToolDefinition or plain dicts
    raw_tools = kwargs.get("tools")
    tools: list[ToolDefinition] | None = None
    if raw_tools is not None:
        tools = [
            ToolDefinition(**t) if isinstance(t, dict) else t
            for t in raw_tools
        ]
    response_format = kwargs.get("response_format")
    if response_format is not None and stream:
        raise ValueError(
            "stream=True and response_format cannot be used together. "
            "Structured outputs require a complete response."
        )
    return CompletionRequest(
        model=model,
        messages=_normalise_messages(messages),
        max_tokens=kwargs.get("max_tokens"),
        temperature=kwargs.get("temperature"),
        top_p=kwargs.get("top_p"),
        stream=stream,
        tools=tools,
        tool_choice=kwargs.get("tool_choice"),
        response_format=response_format,
        extra_kwargs=extra,
    )


# ---------------------------------------------------------------------------
# Public API — completion()
# ---------------------------------------------------------------------------

_MsgList = list[Union[dict[str, str], Message]]
_ModelArg = Union[str, list[str]]


@overload
def completion(
    model: _ModelArg,
    messages: _MsgList,
    *,
    provider: str | None = ...,
    api_key: str | None = ...,
    stream: Literal[True],
    middleware: list[BaseMiddleware] | None = ...,
    fallback_on: tuple[type[Exception], ...] = ...,
    **kwargs: Any,
) -> Iterator[StreamChunk]: ...


@overload
def completion(
    model: _ModelArg,
    messages: _MsgList,
    *,
    provider: str | None = ...,
    api_key: str | None = ...,
    stream: Literal[False] = ...,
    middleware: list[BaseMiddleware] | None = ...,
    fallback_on: tuple[type[Exception], ...] = ...,
    **kwargs: Any,
) -> CompletionResponse: ...


def completion(
    model: _ModelArg,
    messages: _MsgList,
    *,
    provider: str | None = None,
    api_key: str | None = None,
    stream: bool = False,
    middleware: list[BaseMiddleware] | None = None,
    fallback_on: tuple[type[Exception], ...] = _DEFAULT_FALLBACK_ON,
    stream_fallback_mode: str = "restart",
    stream_resume_prompt: str | None = None,
    **kwargs: Any,
) -> CompletionResponse | Iterator[StreamChunk]:
    """
    Perform a synchronous chat completion.

    Args:
        model:                Model name string *or* a list of model strings for
                              automatic fallback routing.  When a list is given,
                              each model is tried in order; the first success is
                              returned.
        messages:             List of message dicts or :class:`~llmgate.types.Message`
                              instances.
        provider:             Force a specific provider by name.
                              Ignored when *model* is a list.
        api_key:              Override the API key for this call.
        stream:               If ``True``, returns an ``Iterator[StreamChunk]``.
        middleware:           Middleware stack applied to every attempt.
        fallback_on:          Exception types that trigger fallback to the next model.
                              Defaults to ``(RateLimitError, ProviderAPIError, AuthError)``.
                              Only used when *model* is a list.
        stream_fallback_mode: Strategy used when a stream fails mid-way during
                              multi-model fallback.  One of:

                              * ``"restart"`` *(default)* — start fresh with the
                                next model; no partial text carried forward.
                              * ``"prefill"`` — append partial text as an assistant
                                message so the fallback model continues from that
                                point.  Auto-downgrades to ``"user_turn"`` for
                                providers that don't support prefill.
                              * ``"user_turn"`` — wrap partial text in a user
                                continuation message; works universally.

                              Only used when *model* is a list and ``stream=True``.
        stream_resume_prompt: Custom continuation message injected in
                              ``"user_turn"`` mode.  Defaults to
                              ``"Continue from exactly where you left off."``.  Pass
                              ``None`` to use the default.
        **kwargs:             Extra parameters forwarded to the provider
                              (``max_tokens``, ``temperature``, ``top_p``, etc.).

    Returns:
        :class:`~llmgate.types.CompletionResponse` when ``stream=False``.
        ``Iterator[StreamChunk]`` when ``stream=True``.

    Raises:
        :class:`~llmgate.exceptions.AllProvidersFailedError`: All models failed.
        :class:`~llmgate.exceptions.ModelNotFoundError`: Unknown model string.
        :class:`~llmgate.exceptions.ProviderError`: Provider returned an error.
    """
    # ---- Multi-model fallback path ----
    if isinstance(model, list):
        if stream:
            from llmgate.fallback import _try_models_stream_sync  # noqa: PLC0415
            return _try_models_stream_sync(
                model,
                messages,
                fallback_on=fallback_on,
                middleware=middleware,
                mode=stream_fallback_mode,
                stream_resume_prompt=stream_resume_prompt,
                provider=provider,
                api_key=api_key,
                **kwargs,
            )
        from llmgate.fallback import _try_models_sync  # noqa: PLC0415
        return _try_models_sync(
            model,
            messages,
            fallback_on=fallback_on,
            middleware=middleware,
            provider=provider,
            api_key=api_key,
            **kwargs,
        )

    # ---- Single-model path (unchanged behaviour) ----
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
    model: _ModelArg,
    messages: _MsgList,
    *,
    provider: str | None = ...,
    api_key: str | None = ...,
    stream: Literal[True],
    middleware: list[BaseMiddleware] | None = ...,
    fallback_on: tuple[type[Exception], ...] = ...,
    **kwargs: Any,
) -> AsyncIterator[StreamChunk]: ...


@overload
async def acompletion(
    model: _ModelArg,
    messages: _MsgList,
    *,
    provider: str | None = ...,
    api_key: str | None = ...,
    stream: Literal[False] = ...,
    middleware: list[BaseMiddleware] | None = ...,
    fallback_on: tuple[type[Exception], ...] = ...,
    **kwargs: Any,
) -> CompletionResponse: ...


async def acompletion(
    model: _ModelArg,
    messages: _MsgList,
    *,
    provider: str | None = None,
    api_key: str | None = None,
    stream: bool = False,
    middleware: list[BaseMiddleware] | None = None,
    fallback_on: tuple[type[Exception], ...] = _DEFAULT_FALLBACK_ON,
    stream_fallback_mode: str = "restart",
    stream_resume_prompt: str | None = None,
    **kwargs: Any,
) -> CompletionResponse | AsyncIterator[StreamChunk]:
    """
    Perform an asynchronous chat completion.

    Same signature as :func:`completion` including ``stream_fallback_mode`` and
    ``stream_resume_prompt``.  When ``stream=True`` and *model* is a list,
    returns an ``AsyncIterator[StreamChunk]`` with fallback support.
    """
    # ---- Multi-model fallback path ----
    if isinstance(model, list):
        if stream:
            from llmgate.fallback import _try_models_stream_async  # noqa: PLC0415
            return _try_models_stream_async(
                model,
                messages,
                fallback_on=fallback_on,
                middleware=middleware,
                mode=stream_fallback_mode,
                stream_resume_prompt=stream_resume_prompt,
                provider=provider,
                api_key=api_key,
                **kwargs,
            )
        from llmgate.fallback import _try_models_async  # noqa: PLC0415
        return await _try_models_async(
            model,
            messages,
            fallback_on=fallback_on,
            middleware=middleware,
            provider=provider,
            api_key=api_key,
            **kwargs,
        )

    # ---- Single-model path (unchanged behaviour) ----
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


# ---------------------------------------------------------------------------
# Convenience: parse() / aparse() — return the Pydantic model directly
# ---------------------------------------------------------------------------


def parse(
    model: str,
    messages: _MsgList,
    *,
    response_format: type[T],
    provider: str | None = None,
    api_key: str | None = None,
    middleware: list | None = None,
    **kwargs: Any,
) -> T:
    """
    Structured-output shorthand — returns a validated Pydantic model instance.

    Equivalent to::

        resp = completion(model, messages, response_format=MyModel, **kwargs)
        return resp.parsed  # type: MyModel

    Raises ``ValueError`` if ``resp.parsed`` is ``None`` (should never happen
    for a well-behaved provider, but guards against edge cases).
    """
    resp = completion(
        model,
        messages,
        response_format=response_format,
        provider=provider,
        api_key=api_key,
        middleware=middleware,
        **kwargs,
    )
    if resp.parsed is None:  # type: ignore[union-attr]
        raise ValueError(
            f"Provider returned no structured output. "
            f"Raw text: {resp.text!r}"  # type: ignore[union-attr]
        )
    return resp.parsed  # type: ignore[return-value,union-attr]


async def aparse(
    model: str,
    messages: _MsgList,
    *,
    response_format: type[T],
    provider: str | None = None,
    api_key: str | None = None,
    middleware: list | None = None,
    **kwargs: Any,
) -> T:
    """Async version of :func:`parse`."""
    resp = await acompletion(
        model,
        messages,
        response_format=response_format,
        provider=provider,
        api_key=api_key,
        middleware=middleware,
        **kwargs,
    )
    if resp.parsed is None:  # type: ignore[union-attr]
        raise ValueError(
            f"Provider returned no structured output. "
            f"Raw text: {resp.text!r}"  # type: ignore[union-attr]
        )
    return resp.parsed  # type: ignore[return-value,union-attr]
