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
    _DEFAULT_FALLBACK_ON,
    _build_request,
    _get_or_create_provider,
)
from llmgate.embeddings import aembed as _aembed_fn
from llmgate.embeddings import embed as _embed_fn
from llmgate.middleware.base import (
    AsyncEmbedNext,
    AsyncNext,
    BaseMiddleware,
    SyncEmbedNext,
    SyncNext,
)
from llmgate.types import (
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    Message,
    StreamChunk,
)


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


def _build_stream_sync_chain(middlewares: list[BaseMiddleware], inner: Any) -> Any:
    """Compose ``stream_handle`` middlewares around ``inner`` (right-to-left)."""
    chain = inner
    for mw in reversed(middlewares):
        _mw, _next = mw, chain
        chain = lambda req, _mw=_mw, _next=_next: _mw.stream_handle(req, _next)  # noqa: E731
    return chain


def _build_stream_async_chain(middlewares: list[BaseMiddleware], inner: Any) -> Any:
    """Compose ``astream_handle`` middlewares around ``inner`` (right-to-left)."""
    chain = inner
    for mw in reversed(middlewares):
        _mw, _next = mw, chain
        chain = lambda req, _mw=_mw, _next=_next: _mw.astream_handle(req, _next)  # noqa: E731
    return chain


def _build_embed_sync_chain(
    middlewares: list[BaseMiddleware],
    inner: SyncEmbedNext,
) -> SyncEmbedNext:
    chain = inner
    for mw in reversed(middlewares):
        _mw, _next = mw, chain
        chain = lambda req, _mw=_mw, _next=_next: _mw.embed_handle(req, _next)  # noqa: E731
    return chain


def _build_embed_async_chain(
    middlewares: list[BaseMiddleware],
    inner: AsyncEmbedNext,
) -> AsyncEmbedNext:
    chain = inner
    for mw in reversed(middlewares):
        _mw, _next = mw, chain
        chain = lambda req, _mw=_mw, _next=_next: _mw.aembed_handle(req, _next)  # noqa: E731
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
        fallback_chain: list[str] | None = None,
        fallback_on: tuple[type[Exception], ...] = _DEFAULT_FALLBACK_ON,
        stream_fallback_mode: str = "restart",
        stream_resume_prompt: str | None = None,
        **provider_defaults: Any,
    ) -> None:
        self._middleware: list[BaseMiddleware] = middleware or []
        self._defaults = provider_defaults
        self._fallback_chain = fallback_chain
        self._fallback_on = fallback_on
        self._stream_fallback_mode = stream_fallback_mode
        self._stream_resume_prompt = stream_resume_prompt

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def completion(
        self,
        model: str | list[str] | None = None,
        messages: list[dict[str, Any] | Message] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Sync completion with middleware stack applied.

        When ``fallback_chain`` was set on this gate, *model* is optional and
        the chain overrides any single model string passed here.
        """
        if messages is None:
            raise ValueError("messages is required")

        merged = {**self._defaults, **kwargs}
        effective_models: list[str] | None = self._fallback_chain

        if effective_models:
            # Full fallback loop — middleware applied per-candidate
            from llmgate.fallback import _try_models_sync  # noqa: PLC0415

            return _try_models_sync(
                effective_models,
                messages,
                fallback_on=self._fallback_on,
                middleware=self._middleware or None,
                **merged,
            )

        # Single-model path
        single_model = (
            model
            if isinstance(model, str)
            else (model[0] if isinstance(model, list) else None)
        )
        if single_model is None:
            raise ValueError(
                "Provide a model string or set fallback_chain on the gate."
            )
        request = _build_request(single_model, messages, stream=False, kwargs=merged)
        provider = _get_or_create_provider(single_model, merged.get("provider"))

        def _inner(req: CompletionRequest) -> CompletionResponse:
            return provider.complete(req)

        chain = _build_sync_chain(self._middleware, _inner)
        return chain(request)

    def stream(
        self,
        model: str | list[str],
        messages: list[dict[str, Any] | Message],
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Sync streaming with fallback support.

        When ``fallback_chain`` is set on this gate *or* ``model`` is a list,
        streaming fallback is applied using ``stream_fallback_mode``.
        """
        merged = {**self._defaults, **kwargs}
        effective_models = self._fallback_chain or (
            model if isinstance(model, list) else [model]
        )

        if len(effective_models) > 1:
            from llmgate.fallback import _try_models_stream_sync  # noqa: PLC0415

            return _try_models_stream_sync(
                effective_models,
                messages,
                fallback_on=self._fallback_on,
                middleware=self._middleware or None,
                mode=self._stream_fallback_mode,
                stream_resume_prompt=self._stream_resume_prompt,
                **merged,
            )

        # Single-model path
        single_model = effective_models[0]
        request = _build_request(single_model, messages, stream=True, kwargs=merged)
        provider = _get_or_create_provider(single_model, merged.get("provider"))

        def _inner(req: CompletionRequest) -> Iterator[StreamChunk]:
            return provider.stream(req)

        chain = (
            _build_stream_sync_chain(self._middleware, _inner)
            if self._middleware
            else _inner
        )
        return chain(request)

    # ------------------------------------------------------------------
    # Async
    # ------------------------------------------------------------------

    async def acompletion(
        self,
        model: str | list[str] | None = None,
        messages: list[dict[str, Any] | Message] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Async completion with middleware stack applied.

        When ``fallback_chain`` was set on this gate, *model* is optional and
        the chain overrides any single model string passed here.
        """
        if messages is None:
            raise ValueError("messages is required")

        merged = {**self._defaults, **kwargs}
        effective_models: list[str] | None = self._fallback_chain

        if effective_models:
            from llmgate.fallback import _try_models_async  # noqa: PLC0415

            return await _try_models_async(
                effective_models,
                messages,
                fallback_on=self._fallback_on,
                middleware=self._middleware or None,
                **merged,
            )

        single_model = (
            model
            if isinstance(model, str)
            else (model[0] if isinstance(model, list) else None)
        )
        if single_model is None:
            raise ValueError(
                "Provide a model string or set fallback_chain on the gate."
            )
        request = _build_request(single_model, messages, stream=False, kwargs=merged)
        provider = _get_or_create_provider(single_model, merged.get("provider"))

        async def _inner(req: CompletionRequest) -> CompletionResponse:
            return await provider.acomplete(req)

        chain = _build_async_chain(self._middleware, _inner)
        return await chain(request)

    async def astream(
        self,
        model: str | list[str],
        messages: list[dict[str, Any] | Message],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Async streaming with fallback support.

        When ``fallback_chain`` is set on this gate *or* ``model`` is a list,
        streaming fallback is applied using ``stream_fallback_mode``.
        """
        merged = {**self._defaults, **kwargs}
        effective_models = self._fallback_chain or (
            model if isinstance(model, list) else [model]
        )

        if len(effective_models) > 1:
            from llmgate.fallback import _try_models_stream_async  # noqa: PLC0415

            async for chunk in _try_models_stream_async(
                effective_models,
                messages,
                fallback_on=self._fallback_on,
                middleware=self._middleware or None,
                mode=self._stream_fallback_mode,
                stream_resume_prompt=self._stream_resume_prompt,
                **merged,
            ):
                yield chunk
            return

        # Single-model path
        single_model = effective_models[0]
        request = _build_request(single_model, messages, stream=True, kwargs=merged)
        provider = _get_or_create_provider(single_model, merged.get("provider"))

        async def _inner(req: CompletionRequest) -> AsyncIterator[StreamChunk]:
            return provider.astream(req)

        chain = (
            _build_stream_async_chain(self._middleware, _inner)
            if self._middleware
            else _inner
        )
        async for chunk in await chain(request):
            yield chunk

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(
        self,
        model: str,
        input: str | list[str],  # noqa: A002
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Generate embeddings using the configured defaults and middleware stack."""
        merged = {**self._defaults, **kwargs}
        api_key = merged.pop("api_key", None)

        request = EmbeddingRequest(
            model=model,
            input=input,
            dimensions=merged.pop("dimensions", None),
            task_type=merged.pop("task_type", None),
            title=merged.pop("title", None),
            input_type=merged.pop("input_type", None),
            truncate=merged.pop("truncate", None),
            encoding_format=merged.pop("encoding_format", None),
            user=merged.pop("user", None),
            extra_kwargs=merged,
        )

        def _inner(req: EmbeddingRequest) -> EmbeddingResponse:
            return _embed_fn(
                model=req.model,
                input=req.input,
                api_key=api_key,
                dimensions=req.dimensions,
                task_type=req.task_type,
                title=req.title,
                input_type=req.input_type,
                truncate=req.truncate,
                encoding_format=req.encoding_format,
                user=req.user,
                **req.extra_kwargs,
            )

        chain = (
            _build_embed_sync_chain(self._middleware, _inner)
            if self._middleware
            else _inner
        )
        return chain(request)

    async def aembed(
        self,
        model: str,
        input: str | list[str],  # noqa: A002
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Async version of :meth:`embed`."""
        merged = {**self._defaults, **kwargs}
        api_key = merged.pop("api_key", None)

        request = EmbeddingRequest(
            model=model,
            input=input,
            dimensions=merged.pop("dimensions", None),
            task_type=merged.pop("task_type", None),
            title=merged.pop("title", None),
            input_type=merged.pop("input_type", None),
            truncate=merged.pop("truncate", None),
            encoding_format=merged.pop("encoding_format", None),
            user=merged.pop("user", None),
            extra_kwargs=merged,
        )

        async def _inner(req: EmbeddingRequest) -> EmbeddingResponse:
            return await _aembed_fn(
                model=req.model,
                input=req.input,
                api_key=api_key,
                dimensions=req.dimensions,
                task_type=req.task_type,
                title=req.title,
                input_type=req.input_type,
                truncate=req.truncate,
                encoding_format=req.encoding_format,
                user=req.user,
                **req.extra_kwargs,
            )

        chain = (
            _build_embed_async_chain(self._middleware, _inner)
            if self._middleware
            else _inner
        )
        return await chain(request)

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def batch(
        self,
        requests: list[Any],
        *,
        max_concurrency: int = 5,
        fail_fast: bool = False,
        timeout: float | None = None,
    ) -> Any:
        """Execute a batch of completion requests using this gate's middleware.

        Wraps :func:`~llmgate.batch.batch`, injecting the gate's middleware
        into every individual request.

        Args:
            requests:        List of :class:`~llmgate.types.CompletionRequest`
                             objects or plain dicts.
            max_concurrency: Max requests in flight simultaneously.
            fail_fast:       Raise on the first error instead of collecting.
            timeout:         Per-request timeout in seconds.

        Returns:
            :class:`~llmgate.types.BatchResult`
        """
        from llmgate.batch import batch as _batch  # noqa: PLC0415

        return _batch(
            requests,
            max_concurrency=max_concurrency,
            fail_fast=fail_fast,
            middleware=self._middleware or None,
            timeout=timeout,
        )

    async def abatch(
        self,
        requests: list[Any],
        *,
        max_concurrency: int = 5,
        fail_fast: bool = False,
        timeout: float | None = None,
    ) -> Any:
        """Async batch of completion requests using this gate's middleware.

        Wraps :func:`~llmgate.batch.abatch`, injecting the gate's middleware
        into every individual request.

        Args:
            requests:        List of :class:`~llmgate.types.CompletionRequest`
                             objects or plain dicts.
            max_concurrency: Max requests in flight simultaneously.
            fail_fast:       Raise on the first error instead of collecting.
            timeout:         Per-request timeout in seconds.

        Returns:
            :class:`~llmgate.types.BatchResult`
        """
        from llmgate.batch import abatch as _abatch  # noqa: PLC0415

        return await _abatch(
            requests,
            max_concurrency=max_concurrency,
            fail_fast=fail_fast,
            middleware=self._middleware or None,
            timeout=timeout,
        )
