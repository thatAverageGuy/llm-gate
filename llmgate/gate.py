"""
llmgate.gate
~~~~~~~~~~~~
``LLMGate`` — a stateful client that bundles a middleware stack with the
provider registry.

Usage::

    from llmgate import LLMGate
    from llmgate.middleware import RetryMiddleware, LoggingMiddleware, CacheMiddleware

    gate = LLMGate(middleware=[
        RetryMiddleware(max_retries=3),
        LoggingMiddleware(level="INFO"),
        CacheMiddleware(ttl=300),
    ])

    # Sync
    resp = gate.completion("groq/llama-3.1-8b-instant", messages)

    # Async
    resp = await gate.acompletion("gemini-2.5-flash-lite", messages)

    # Streaming
    for chunk in gate.stream("groq/llama-3.1-8b-instant", messages):
        print(chunk.delta, end="", flush=True)
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

from llmgate.completion import (
    _build_request,
    _get_or_create_provider,
    _normalise_messages,
)
from llmgate.middleware.base import AsyncNext, BaseMiddleware, SyncNext
from llmgate.types import CompletionRequest, CompletionResponse, Message, StreamChunk


def _build_sync_chain(
    middlewares: list[BaseMiddleware],
    inner: SyncNext,
) -> SyncNext:
    """Compose a list of middlewares around ``inner`` (right-to-left: first item is outermost)."""
    chain = inner
    for mw in reversed(middlewares):
        # Capture mw and next_call in closure
        _mw, _next = mw, chain
        chain = lambda req, _mw=_mw, _next=_next: _mw.handle(req, _next)  # noqa: E731
    return chain


def _build_async_chain(
    middlewares: list[BaseMiddleware],
    inner: AsyncNext,
) -> AsyncNext:
    chain = inner
    for mw in reversed(middlewares):
        _mw, _next = mw, chain
        chain = lambda req, _mw=_mw, _next=_next: _mw.ahandle(req, _next)  # noqa: E731
    return chain


class LLMGate:
    """
    A configured LLM gateway with a fixed middleware stack.

    The middleware list is applied **left to right**; the leftmost
    middleware is the outermost wrapper (executed first on the way in,
    last on the way out).  Recommended order::

        [RetryMiddleware, LoggingMiddleware, CacheMiddleware]

    Args:
        middleware: Ordered list of ``BaseMiddleware`` instances.
        **provider_defaults: Default kwargs forwarded to every call
                             (e.g. ``temperature=0.0``).
    """

    def __init__(
        self,
        middleware: list[BaseMiddleware] | None = None,
        **provider_defaults: Any,
    ) -> None:
        self._middleware: list[BaseMiddleware] = middleware or []
        self._defaults = provider_defaults

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any] | Message],
        **kwargs: Any,
    ) -> CompletionResponse:
        """Sync completion with middleware stack applied."""
        merged = {**self._defaults, **kwargs}
        request = _build_request(model, messages, stream=False, kwargs=merged)
        provider = _get_or_create_provider(model, merged.get("provider"))

        def _inner(req: CompletionRequest) -> CompletionResponse:
            return provider.complete(req)

        chain = _build_sync_chain(self._middleware, _inner)
        return chain(request)

    def stream(
        self,
        model: str,
        messages: list[dict[str, Any] | Message],
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Sync streaming with stream_handle middleware applied."""
        merged = {**self._defaults, **kwargs}
        request = _build_request(model, messages, stream=True, kwargs=merged)
        provider = _get_or_create_provider(model, merged.get("provider"))

        def _inner(req: CompletionRequest) -> Iterator[StreamChunk]:
            return provider.stream(req)

        # Build stream chain
        chain = _inner
        for mw in reversed(self._middleware):
            _mw, _next = mw, chain
            chain = lambda req, _mw=_mw, _next=_next: _mw.stream_handle(req, _next)  # noqa: E731

        return chain(request)

    # ------------------------------------------------------------------
    # Async
    # ------------------------------------------------------------------

    async def acompletion(
        self,
        model: str,
        messages: list[dict[str, Any] | Message],
        **kwargs: Any,
    ) -> CompletionResponse:
        """Async completion with middleware stack applied."""
        merged = {**self._defaults, **kwargs}
        request = _build_request(model, messages, stream=False, kwargs=merged)
        provider = _get_or_create_provider(model, merged.get("provider"))

        async def _inner(req: CompletionRequest) -> CompletionResponse:
            return await provider.acomplete(req)

        chain = _build_async_chain(self._middleware, _inner)
        return await chain(request)

    async def astream(
        self,
        model: str,
        messages: list[dict[str, Any] | Message],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Async streaming with astream_handle middleware applied."""
        merged = {**self._defaults, **kwargs}
        request = _build_request(model, messages, stream=True, kwargs=merged)
        provider = _get_or_create_provider(model, merged.get("provider"))

        async def _inner(req: CompletionRequest) -> AsyncIterator[StreamChunk]:
            return provider.astream(req)

        chain = _inner
        for mw in reversed(self._middleware):
            _mw, _next = mw, chain
            chain = lambda req, _mw=_mw, _next=_next: _mw.astream_handle(req, _next)  # noqa: E731

        async for chunk in await chain(request):
            yield chunk
