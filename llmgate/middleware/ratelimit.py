"""
llmgate.middleware.ratelimit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Client-side token-bucket rate limiter.

Raises ``RateLimitError`` (before the request is sent) when the
configured rate would be exceeded.  Intended to protect against
accidentally hammering a provider quota.

For recovering from provider 429s, use ``RetryMiddleware`` instead (or
stack both).

Usage::

    from llmgate.middleware import RateLimitMiddleware

    # Allow at most 20 requests per minute
    gate = LLMGate(middleware=[RateLimitMiddleware(requests_per_minute=20)])
"""
from __future__ import annotations

import asyncio
import threading
import time

from llmgate.exceptions import RateLimitError
from llmgate.middleware.base import AsyncNext, BaseMiddleware, SyncNext
from llmgate.types import CompletionRequest, CompletionResponse


class RateLimitMiddleware(BaseMiddleware):
    """
    Token-bucket client-side rate limiter.

    Args:
        requests_per_minute: Maximum requests allowed per 60-second window.
                             Default: 60.
        burst:               Max tokens allowed to accumulate (controls burst size).
                             Default: equal to ``requests_per_minute``.
        raise_on_limit:      If True, raise ``RateLimitError`` when rate exceeded.
                             If False, block until a token is available. Default: True.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst: int | None = None,
        raise_on_limit: bool = True,
    ) -> None:
        self._rate = requests_per_minute / 60.0      # tokens per second
        self._capacity = float(burst or requests_per_minute)
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._raise_on_limit = raise_on_limit

    def _refill(self) -> None:
        """Add tokens proportional to elapsed time (up to capacity)."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def _try_consume(self) -> float:
        """Consume one token.  Returns 0.0 on success or seconds-to-wait on failure."""
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0
            return (1.0 - self._tokens) / self._rate

    def handle(
        self,
        request: CompletionRequest,
        call_next: SyncNext,
    ) -> CompletionResponse:
        wait = self._try_consume()
        if wait > 0:
            if self._raise_on_limit:
                raise RateLimitError(
                    f"Client-side rate limit exceeded. Try again in {wait:.1f}s.",
                    provider="llmgate",
                )
            time.sleep(wait)
            with self._lock:
                self._tokens -= 1.0  # consume after waiting
        return call_next(request)

    async def ahandle(
        self,
        request: CompletionRequest,
        call_next: AsyncNext,
    ) -> CompletionResponse:
        wait = self._try_consume()
        if wait > 0:
            if self._raise_on_limit:
                raise RateLimitError(
                    f"Client-side rate limit exceeded. Try again in {wait:.1f}s.",
                    provider="llmgate",
                )
            await asyncio.sleep(wait)
            with self._lock:
                self._tokens -= 1.0
        return await call_next(request)
