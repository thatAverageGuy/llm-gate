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

Public helpers
--------------
``_try_models_sync``   — called by ``completion()`` and ``LLMGate.completion()``
``_try_models_async``  — called by ``acompletion()`` and ``LLMGate.acompletion()``
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmgate.exceptions import AllProvidersFailedError
from llmgate.types import CompletionRequest, CompletionResponse

# These are imported at module level so they can be patched in tests.
# fallback.py is itself only imported lazily (inside completion.py function
# bodies), so there is no circular import at package load time.
from llmgate.completion import _build_request, _get_provider  # noqa: PLC0415
from llmgate.gate import _build_sync_chain, _build_async_chain  # noqa: PLC0415

if TYPE_CHECKING:
    from llmgate.middleware.base import BaseMiddleware


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
    build_kwargs = {k: v for k, v in kwargs.items() if k != "fallback_on"}

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
    """Async variant of :func:`_try_models_sync`.

    Args:
        models:      Ordered list of model strings to attempt.
        messages:    The message list, forwarded verbatim to each attempt.
        fallback_on: Tuple of exception types that should trigger fallback.
        middleware:  Optional middleware stack applied to every attempt.
        **kwargs:    Forwarded to ``_build_request``.

    Returns:
        The first successful :class:`~llmgate.types.CompletionResponse`.

    Raises:
        AllProvidersFailedError: All models raised an error in *fallback_on*.
    """
    errors: list[tuple[str, Exception]] = []
    attempted: list[str] = []

    build_kwargs = {k: v for k, v in kwargs.items() if k != "fallback_on"}

    for model in models:
        try:
            provider = _get_provider(
                model,
                build_kwargs.get("provider"),
                build_kwargs.get("api_key"),
            )
            request = _build_request(model, messages, stream=False, kwargs=build_kwargs)

            if middleware:
                async def _inner(req: CompletionRequest, _p=provider) -> CompletionResponse:
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
