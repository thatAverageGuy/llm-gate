"""
llmgate.fallback
~~~~~~~~~~~~~~~~
Core fallback / routing logic for multi-model completion chains.

Both sync and async variants share the same strategy:

1. Iterate the model list in order.
2. For each model, run the full middleware-wrapped completion call so that
   middlewares like ``RetryMiddleware`` still apply per-model.
3. On success, stamp ``CompletionResponse.fallback_attempts`` with the list of
   models that were tried before this one and return.
4. On an error that is in ``fallback_on``, record it and advance to the next
   model.
5. After exhausting all models, raise ``AllProvidersFailedError`` with the
   collected ``(model, exception)`` pairs.

Streaming fallback
------------------
``_try_models_stream_sync``  / ``_try_models_stream_async`` support three
``stream_fallback_mode`` strategies:

* ``"restart"``    — start fresh with the next model (default, always safe).
* ``"prefill"``    — append buffered partial text as an assistant message so
                     the fallback model continues from that point.  Auto-
                     degrades to ``"user_turn"`` when the target provider does
                     not support prefill (``supports_prefill = False``).
* ``"user_turn"``  — wrap the partial text in a user continuation message.
                     Works universally across all providers.

A ``UserWarning`` is emitted on every fallback event.

Public helpers
--------------
``_try_models_sync``          — called by ``completion()`` and ``LLMGate.completion()``
``_try_models_async``         — called by ``acompletion()`` and ``LLMGate.acompletion()``
``_try_models_stream_sync``   — called by ``completion(..., stream=True)`` and
                                ``LLMGate.stream()``
``_try_models_stream_async``  — called by ``acompletion(..., stream=True)`` and
                                ``LLMGate.astream()``
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator

from llmgate.exceptions import AllProvidersFailedError
from llmgate.types import CompletionRequest, CompletionResponse, Message, StreamChunk

# These are imported at module level so they can be patched in tests.
# fallback.py is itself only imported lazily (inside completion.py function
# bodies), so there is no circular import at package load time.
from llmgate.completion import _build_request, _get_provider  # noqa: PLC0415
from llmgate.gate import _build_sync_chain, _build_async_chain  # noqa: PLC0415
from llmgate.gate import _build_stream_sync_chain, _build_stream_async_chain  # noqa: PLC0415

if TYPE_CHECKING:
    from llmgate.middleware.base import BaseMiddleware

# Keys that are consumed by the fallback layer and must not be forwarded to
# _build_request (which would treat them as unknown extra_kwargs).
_FALLBACK_KEYS = frozenset(
    {"fallback_on", "stream_fallback_mode", "stream_resume_prompt"}
)


# ---------------------------------------------------------------------------
# Non-streaming fallback helpers (unchanged behaviour)
# ---------------------------------------------------------------------------


def _try_models_sync(
    models: list[str],
    messages: list[Any],
    fallback_on: tuple[type[Exception], ...],
    middleware: list[BaseMiddleware] | None,
    **kwargs: Any,
) -> CompletionResponse:
    """Try each model in *models* sequentially (sync), falling back on *fallback_on* errors.

    Args:
        models:      Ordered list of model strings to attempt.
        messages:    The message list, forwarded verbatim to each attempt.
        fallback_on: Tuple of exception types that should trigger fallback.
                     Other exceptions propagate immediately.
        middleware:  Optional middleware stack applied to every attempt.
        **kwargs:    Forwarded to ``_build_request`` (temperature, max_tokens, etc.).

    Returns:
        The first successful :class:`~llmgate.types.CompletionResponse`, with
        ``fallback_attempts`` populated with the models that failed before it.

    Raises:
        AllProvidersFailedError: All models raised an error in *fallback_on*.
        Exception: Any exception *not* in *fallback_on* is re-raised immediately.
    """
    errors: list[tuple[str, Exception]] = []
    attempted: list[str] = []

    # Strip out fallback-specific kwargs that _build_request doesn't know about
    build_kwargs = {k: v for k, v in kwargs.items() if k not in _FALLBACK_KEYS}

    for model in models:
        try:
            provider = _get_provider(
                model,
                build_kwargs.get("provider"),
                build_kwargs.get("api_key"),
            )
            request = _build_request(model, messages, stream=False, kwargs=build_kwargs)

            if middleware:

                def _inner(req: CompletionRequest, _p=provider) -> CompletionResponse:
                    return _p.complete(req)

                chain = _build_sync_chain(middleware, _inner)
                resp = chain(request)
            else:
                resp = provider.complete(request)

            # Annotate with fallback metadata and return
            return resp.model_copy(update={"fallback_attempts": attempted.copy()})

        except fallback_on as exc:  # type: ignore[misc]
            errors.append((model, exc))
            attempted.append(model)
        # Any other exception propagates immediately (not caught here)

    raise AllProvidersFailedError(errors)


async def _try_models_async(
    models: list[str],
    messages: list[Any],
    fallback_on: tuple[type[Exception], ...],
    middleware: list[BaseMiddleware] | None,
    **kwargs: Any,
) -> CompletionResponse:
    """Async variant of :func:`_try_models_sync`."""
    errors: list[tuple[str, Exception]] = []
    attempted: list[str] = []

    build_kwargs = {k: v for k, v in kwargs.items() if k not in _FALLBACK_KEYS}

    for model in models:
        try:
            provider = _get_provider(
                model,
                build_kwargs.get("provider"),
                build_kwargs.get("api_key"),
            )
            request = _build_request(model, messages, stream=False, kwargs=build_kwargs)

            if middleware:

                async def _inner(
                    req: CompletionRequest, _p=provider
                ) -> CompletionResponse:
                    return await _p.acomplete(req)

                chain = _build_async_chain(middleware, _inner)
                resp = await chain(request)
            else:
                resp = await provider.acomplete(request)

            return resp.model_copy(update={"fallback_attempts": attempted.copy()})

        except fallback_on as exc:  # type: ignore[misc]
            errors.append((model, exc))
            attempted.append(model)

    raise AllProvidersFailedError(errors)


# ---------------------------------------------------------------------------
# Stream-resume helper
# ---------------------------------------------------------------------------


def _build_resume_messages(
    original_messages: list[Any],
    partial: str,
    mode: str,
    stream_resume_prompt: str | None,
    fallback_model: str,
    build_kwargs: dict[str, Any],
) -> list[Any]:
    """Return the message list to send to the fallback model.

    For ``"restart"`` mode (or when there is no partial text yet) the original
    messages are returned unchanged.  For ``"prefill"`` and ``"user_turn"``
    modes the buffered partial text is injected so the fallback model can
    continue the response seamlessly.
    """
    if not partial or mode == "restart":
        return original_messages

    if mode == "prefill":
        provider = _get_provider(
            fallback_model,
            build_kwargs.get("provider"),
            build_kwargs.get("api_key"),
        )
        if not getattr(provider, "supports_prefill", True):
            warnings.warn(
                f"llmgate: provider '{provider.name}' does not support assistant prefill "
                f"(model={fallback_model!r}). Automatically using 'user_turn' continuation.",
                stacklevel=6,
            )
            # fall through to user_turn block below
        else:
            return list(original_messages) + [
                Message(role="assistant", content=partial)
            ]

    # "user_turn" (or prefill that was downgraded because provider doesn't support it)
    prompt = stream_resume_prompt or "Continue from exactly where you left off."
    return list(original_messages) + [
        Message(role="assistant", content=partial),
        Message(role="user", content=prompt),
    ]


# ---------------------------------------------------------------------------
# Streaming fallback helpers
# ---------------------------------------------------------------------------


def _try_models_stream_sync(
    models: list[str],
    messages: list[Any],
    fallback_on: tuple[type[Exception], ...],
    middleware: list[BaseMiddleware] | None,
    mode: str = "restart",
    stream_resume_prompt: str | None = None,
    **kwargs: Any,
) -> Iterator[StreamChunk]:
    """Try each model in *models* sequentially (sync streaming).

    Buffers yielded chunks so that on a mid-stream failure the next model can
    resume from where the previous one left off (depending on *mode*).

    Args:
        models:               Ordered list of model strings to attempt.
        messages:             Original message list.
        fallback_on:          Exception types that trigger fallback.
        middleware:           Middleware stack applied per attempt.
        mode:                 ``"restart"`` | ``"prefill"`` | ``"user_turn"``.
        stream_resume_prompt: User message injected in ``"user_turn"`` mode.
        **kwargs:             Forwarded to ``_build_request``.

    Yields:
        :class:`~llmgate.types.StreamChunk` — first chunk of each model run
        has ``fallback_attempts`` and ``resumed_from_partial`` populated.

    Raises:
        AllProvidersFailedError: All models failed with errors in *fallback_on*.
    """
    errors: list[tuple[str, Exception]] = []
    attempted: list[str] = []
    accumulated_partial: str = ""

    build_kwargs = {k: v for k, v in kwargs.items() if k not in _FALLBACK_KEYS}

    for idx, model in enumerate(models):
        # In restart mode we never inject partial text, so resumed_from_partial=False
        is_resume = bool(accumulated_partial) and mode != "restart"
        run_messages = (
            _build_resume_messages(
                list(messages),
                accumulated_partial,
                mode,
                stream_resume_prompt,
                model,
                build_kwargs,
            )
            if is_resume
            else list(messages)
        )

        provider = _get_provider(
            model,
            build_kwargs.get("provider"),
            build_kwargs.get("api_key"),
        )
        request = _build_request(model, run_messages, stream=True, kwargs=build_kwargs)

        if middleware:

            def _inner(req: CompletionRequest, _p=provider) -> Iterator[StreamChunk]:
                return _p.stream(req)

            stream_iter: Iterator[StreamChunk] = _build_stream_sync_chain(
                middleware, _inner
            )(request)
        else:
            stream_iter = provider.stream(request)

        first_chunk = True
        try:
            for chunk in stream_iter:
                if first_chunk:
                    chunk = chunk.model_copy(
                        update={
                            "fallback_attempts": attempted.copy(),
                            "resumed_from_partial": is_resume,
                        }
                    )
                    first_chunk = False
                accumulated_partial += chunk.delta
                yield chunk
            return  # stream completed — done

        except fallback_on as exc:  # type: ignore[misc]
            errors.append((model, exc))
            attempted.append(model)
            remaining = models[idx + 1 :]
            warnings.warn(
                f"llmgate: stream from {model!r} failed with "
                f"{type(exc).__name__}: {exc}. "
                + (
                    f"Falling back to {remaining[0]!r} (stream_fallback_mode={mode!r})."
                    if remaining
                    else "No more fallback models."
                ),
                stacklevel=3,
            )
            # accumulated_partial carries the text already yielded to the caller

    raise AllProvidersFailedError(errors)


async def _try_models_stream_async(
    models: list[str],
    messages: list[Any],
    fallback_on: tuple[type[Exception], ...],
    middleware: list[BaseMiddleware] | None,
    mode: str = "restart",
    stream_resume_prompt: str | None = None,
    **kwargs: Any,
) -> AsyncIterator[StreamChunk]:
    """Async variant of :func:`_try_models_stream_sync`."""
    errors: list[tuple[str, Exception]] = []
    attempted: list[str] = []
    accumulated_partial: str = ""

    build_kwargs = {k: v for k, v in kwargs.items() if k not in _FALLBACK_KEYS}

    for idx, model in enumerate(models):
        # In restart mode we never inject partial text, so resumed_from_partial=False
        is_resume = bool(accumulated_partial) and mode != "restart"
        run_messages = (
            _build_resume_messages(
                list(messages),
                accumulated_partial,
                mode,
                stream_resume_prompt,
                model,
                build_kwargs,
            )
            if is_resume
            else list(messages)
        )

        provider = _get_provider(
            model,
            build_kwargs.get("provider"),
            build_kwargs.get("api_key"),
        )
        request = _build_request(model, run_messages, stream=True, kwargs=build_kwargs)

        if middleware:

            async def _inner(
                req: CompletionRequest, _p=provider
            ) -> AsyncIterator[StreamChunk]:
                return _p.astream(req)

            astream_iter: AsyncIterator[StreamChunk] = await _build_stream_async_chain(
                middleware, _inner
            )(request)
        else:
            # provider.astream() returns an AsyncIterator — do NOT await it
            astream_iter = provider.astream(request)

        first_chunk = True
        try:
            async for chunk in astream_iter:
                if first_chunk:
                    chunk = chunk.model_copy(
                        update={
                            "fallback_attempts": attempted.copy(),
                            "resumed_from_partial": is_resume,
                        }
                    )
                    first_chunk = False
                accumulated_partial += chunk.delta
                yield chunk
            return

        except fallback_on as exc:  # type: ignore[misc]
            errors.append((model, exc))
            attempted.append(model)
            remaining = models[idx + 1 :]
            warnings.warn(
                f"llmgate: stream from {model!r} failed with "
                f"{type(exc).__name__}: {exc}. "
                + (
                    f"Falling back to {remaining[0]!r} (stream_fallback_mode={mode!r})."
                    if remaining
                    else "No more fallback models."
                ),
                stacklevel=3,
            )

    raise AllProvidersFailedError(errors)
