"""
llmgate.middleware.fallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~
``FallbackMiddleware`` — try the primary model first, then fall back to
alternative models when a transient error occurs.

Usage::

    from llmgate import LLMGate
    from llmgate.middleware import RetryMiddleware, FallbackMiddleware

    gate = LLMGate(middleware=[
        RetryMiddleware(max_retries=2),          # retries the primary model
        FallbackMiddleware(                       # then falls back if still failing
            models=["groq/llama-3.1-8b-instant", "gemini-2.5-flash-lite"],
        ),
    ])
    resp = gate.completion("gpt-4o-mini", messages)

How it works
------------
``FallbackMiddleware`` sits around the inner middleware chain.  It calls
``call_next(request)`` for the primary model.  If that raises an error whose
type is in ``fallback_on``, it directly calls each fallback model's provider
in sequence (bypassing the middleware chain to avoid recursion).  Any other
exception type propagates immediately.

.. note::

    For the recommended layering (retry, *then* fall back), place
    ``RetryMiddleware`` **before** ``FallbackMiddleware`` in the list passed to
    ``LLMGate`` so that retries are exhausted on the primary model before
    fallback kicks in.

    Fallback models do *not* run through the middleware stack — this is a
    deliberate v0.6 simplification to avoid recursive middleware chains.
    Use ``LLMGate(fallback_chain=[...])`` or ``completion(model=[...])``
    if you need full middleware coverage on every candidate.
"""
from __future__ import annotations


from llmgate.exceptions import AllProvidersFailedError, AuthError, ProviderAPIError, RateLimitError
from llmgate.middleware.base import AsyncNext, BaseMiddleware, SyncNext
from llmgate.types import CompletionRequest, CompletionResponse

# Module-level import so tests can patch llmgate.middleware.fallback._get_provider
from llmgate.completion import _get_provider  # noqa: PLC0415

#: Default exception types that trigger a fallback attempt.
_DEFAULT_FALLBACK_ON: tuple[type[Exception], ...] = (
    RateLimitError,
    ProviderAPIError,
    AuthError,
)


class FallbackMiddleware(BaseMiddleware):
    """Try the primary model, then fall back to *models* on transient errors.

    Args:
        models:      Ordered list of fallback model strings to try when the
                     primary model (passed to ``completion()``) fails.
        fallback_on: Tuple of exception types that trigger fallback.
                     Defaults to ``(RateLimitError, ProviderAPIError, AuthError)``.

    Example::

        from llmgate import LLMGate
        from llmgate.middleware import FallbackMiddleware

        gate = LLMGate(middleware=[
            FallbackMiddleware(
                models=["groq/llama-3.1-8b-instant", "gemini-2.5-flash-lite"],
            ),
        ])
        resp = gate.completion("gpt-4o-mini", messages)
        print(resp.text)
        print(resp.fallback_attempts)  # ["gpt-4o-mini"] if primary failed
    """

    def __init__(
        self,
        models: list[str],
        fallback_on: tuple[type[Exception], ...] = _DEFAULT_FALLBACK_ON,
    ) -> None:
        if not models:
            raise ValueError("FallbackMiddleware requires at least one fallback model.")
        self.models = models
        self.fallback_on = fallback_on

    def _call_provider(self, model: str, request: CompletionRequest) -> CompletionResponse:
        """Directly call a provider, bypassing the middleware chain."""
        provider = _get_provider(model, None, None)
        # Create a new request with the fallback model name
        fallback_req = request.model_copy(update={"model": model})
        return provider.complete(fallback_req)

    async def _acall_provider(self, model: str, request: CompletionRequest) -> CompletionResponse:
        """Async: directly call a provider, bypassing the middleware chain."""
        provider = _get_provider(model, None, None)
        fallback_req = request.model_copy(update={"model": model})
        return await provider.acomplete(fallback_req)

    def handle(
        self,
        request: CompletionRequest,
        call_next: SyncNext,
    ) -> CompletionResponse:
        primary_model = request.model
        errors: list[tuple[str, Exception]] = []

        # --- Try primary model (via full middleware chain) ---
        try:
            resp = call_next(request)
            # Primary succeeded — no fallback_attempts to record
            return resp
        except self.fallback_on as exc:  # type: ignore[misc]
            errors.append((primary_model, exc))

        # --- Try fallback models (direct provider calls) ---
        attempted = [primary_model]
        for model in self.models:
            try:
                resp = self._call_provider(model, request)
                return resp.model_copy(update={"fallback_attempts": attempted.copy()})
            except self.fallback_on as exc:  # type: ignore[misc]
                errors.append((model, exc))
                attempted.append(model)

        raise AllProvidersFailedError(errors)

    async def ahandle(
        self,
        request: CompletionRequest,
        call_next: AsyncNext,
    ) -> CompletionResponse:
        primary_model = request.model
        errors: list[tuple[str, Exception]] = []

        try:
            resp = await call_next(request)
            return resp
        except self.fallback_on as exc:  # type: ignore[misc]
            errors.append((primary_model, exc))

        attempted = [primary_model]
        for model in self.models:
            try:
                resp = await self._acall_provider(model, request)
                return resp.model_copy(update={"fallback_attempts": attempted.copy()})
            except self.fallback_on as exc:  # type: ignore[misc]
                errors.append((model, exc))
                attempted.append(model)

        raise AllProvidersFailedError(errors)
