"""
llmgate.middleware.retry
~~~~~~~~~~~~~~~~~~~~~~~~
Exponential backoff retry middleware.

Retries on ``RateLimitError`` (always) and optionally on
``ProviderAPIError`` (configurable, default: True).

Backoff formula::

    wait = backoff_factor * (2 ** attempt) + uniform(0, jitter_max)

If the exception carries a ``retry_after`` attribute (provided by some
providers on 429 responses), that value takes precedence.

Usage::

    from llmgate.middleware import RetryMiddleware

    gate = LLMGate(middleware=[RetryMiddleware(max_retries=3, backoff_factor=1.0)])
"""
from __future__ import annotations

import asyncio
import random
import time

from llmgate.exceptions import ProviderAPIError, RateLimitError
from llmgate.middleware.base import AsyncNext, BaseMiddleware, SyncNext
from llmgate.types import CompletionRequest, CompletionResponse


class RetryMiddleware(BaseMiddleware):
    """
    Retry with exponential backoff on transient provider errors.

    Args:
        max_retries:             Maximum number of retry attempts (not including
                                 the original call). Default: 3.
        backoff_factor:          Base multiplier for wait time. Default: 1.0.
        jitter_max:              Max seconds of random jitter added to each wait.
                                 Default: 0.5.
        retry_on_provider_errors: Also retry on ``ProviderAPIError`` (5xx).
                                 Default: True.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        jitter_max: float = 0.5,
        retry_on_provider_errors: bool = True,
    ) -> None:
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.jitter_max = jitter_max
        self.retry_on_provider_errors = retry_on_provider_errors

    def _should_retry(self, exc: Exception) -> bool:
        if isinstance(exc, RateLimitError):
            return True
        if self.retry_on_provider_errors and isinstance(exc, ProviderAPIError):
            status = getattr(exc, "status_code", None)
            # Only retry 5xx errors; 4xx (except 429) are non-transient
            if status is None or status >= 500:
                return True
        return False

    def _wait_time(self, attempt: int, exc: Exception) -> float:
        # Respect Retry-After if the provider sent one
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None:
            try:
                return float(retry_after)
            except (TypeError, ValueError):
                pass
        return self.backoff_factor * (2 ** attempt) + random.uniform(0, self.jitter_max)

    def handle(
        self,
        request: CompletionRequest,
        call_next: SyncNext,
    ) -> CompletionResponse:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return call_next(request)
            except Exception as exc:
                if not self._should_retry(exc) or attempt >= self.max_retries:
                    raise
                last_exc = exc
                wait = self._wait_time(attempt, exc)
                time.sleep(wait)
        # Unreachable, but satisfies type checkers
        raise last_exc  # type: ignore[misc]

    async def ahandle(
        self,
        request: CompletionRequest,
        call_next: AsyncNext,
    ) -> CompletionResponse:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await call_next(request)
            except Exception as exc:
                if not self._should_retry(exc) or attempt >= self.max_retries:
                    raise
                last_exc = exc
                wait = self._wait_time(attempt, exc)
                await asyncio.sleep(wait)
        raise last_exc  # type: ignore[misc]
