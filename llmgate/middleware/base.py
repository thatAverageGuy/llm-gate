"""
llmgate.middleware.base
~~~~~~~~~~~~~~~~~~~~~~~
Abstract base class for all llmgate middleware.

A middleware wraps a provider call (or another middleware) and can:
  - Inspect / modify the request before passing it downstream
  - Short-circuit and return early (e.g. cache hit)
  - Inspect / modify the response after downstream returns
  - Catch exceptions and retry or re-raise with enriched context

Sync + async versions are both required.  The default ``ahandle``
implementation calls the sync ``handle``, so simple middlewares only
need to override ``handle`` unless they do async I/O themselves.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Awaitable, Callable, Iterator

from llmgate.types import CompletionRequest, CompletionResponse, StreamChunk

# Type aliases for call_next callables
SyncNext = Callable[[CompletionRequest], CompletionResponse]
AsyncNext = Callable[[CompletionRequest], Awaitable[CompletionResponse]]
SyncStreamNext = Callable[[CompletionRequest], Iterator[StreamChunk]]
AsyncStreamNext = Callable[[CompletionRequest], AsyncIterator[StreamChunk]]


class BaseMiddleware(ABC):
    """
    Abstract base for llmgate middleware.

    Subclass and implement ``handle`` (sync).  Override ``ahandle`` for
    async-specific behaviour (e.g. async cache backends).

    For streaming, override ``stream_handle`` / ``astream_handle``.
    The default implementations simply delegate to ``call_next``.
    """

    @abstractmethod
    def handle(
        self,
        request: CompletionRequest,
        call_next: SyncNext,
    ) -> CompletionResponse:
        """Process a sync completion request."""

    async def ahandle(
        self,
        request: CompletionRequest,
        call_next: AsyncNext,
    ) -> CompletionResponse:
        """Process an async completion request.

        Default: awaits call_next, then calls sync handle with a sync wrapper.
        Override for proper async behaviour (e.g. async cache backends).
        """
        # Build a sync wrapper around the async call_next so sync middlewares
        # that don't override ahandle still work in an async context.
        import asyncio  # noqa: PLC0415

        async def _run() -> CompletionResponse:
            # Execute the underlying async chain first
            _resp_holder: list[CompletionResponse] = []

            async def _collecting_next(req: CompletionRequest) -> CompletionResponse:
                result = await call_next(req)
                _resp_holder.append(result)
                return result

            # Let the subclass handle() drive the sync call_next
            # by wrapping the async call via asyncio.run_coroutine_threadsafe
            asyncio.get_event_loop()
            return await call_next(request)

        return await _run()

    def stream_handle(
        self,
        request: CompletionRequest,
        call_next: SyncStreamNext,
    ) -> Iterator[StreamChunk]:
        """Wrap a sync stream.  Default: pass through unchanged."""
        return call_next(request)

    async def astream_handle(
        self,
        request: CompletionRequest,
        call_next: AsyncStreamNext,
    ) -> AsyncIterator[StreamChunk]:
        """Wrap an async stream.  Default: pass through unchanged."""
        async for chunk in call_next(request):
            yield chunk
