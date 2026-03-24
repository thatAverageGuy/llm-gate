"""
llmgate.middleware.cache
~~~~~~~~~~~~~~~~~~~~~~~~
In-memory TTL + LRU cache middleware.

Cache key is derived from: model, messages, max_tokens, temperature, top_p.
Streaming and tool-call requests are never cached.

Usage::

    from llmgate.middleware import CacheMiddleware

    gate = LLMGate(middleware=[CacheMiddleware(ttl=300, maxsize=256)])
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any

from llmgate.middleware.base import AsyncNext, BaseMiddleware, SyncNext
from llmgate.types import CompletionRequest, CompletionResponse


class CacheMiddleware(BaseMiddleware):
    """
    Transparent in-memory response cache with TTL and LRU eviction.

    Args:
        ttl:     Time-to-live in seconds.  Default: 300 (5 minutes).
        maxsize: Maximum number of cached entries.  Default: 128.

    Cache misses on:
        - ``stream=True`` requests
        - Requests with ``tools`` set (non-deterministic by nature)
    """

    def __init__(self, ttl: float = 300.0, maxsize: int = 128) -> None:
        self.ttl = ttl
        self.maxsize = maxsize
        # OrderedDict used as an LRU store: key → (response, expiry_ts)
        self._store: OrderedDict[str, tuple[CompletionResponse, float]] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Cache key
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(request: CompletionRequest) -> str:
        """Stable SHA-256 key from the deterministic parts of the request."""
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [m.to_dict() for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Store helpers
    # ------------------------------------------------------------------

    def _get(self, key: str) -> CompletionResponse | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            resp, expiry = entry
            if time.monotonic() > expiry:
                del self._store[key]
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            return resp

    def _set(self, key: str, resp: CompletionResponse) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (resp, time.monotonic() + self.ttl)
            # LRU eviction
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def _should_cache(self, request: CompletionRequest) -> bool:
        # Never cache streams or tool-calling requests
        return not request.stream and not request.tools

    # ------------------------------------------------------------------
    # Middleware interface
    # ------------------------------------------------------------------

    def handle(
        self,
        request: CompletionRequest,
        call_next: SyncNext,
    ) -> CompletionResponse:
        if not self._should_cache(request):
            return call_next(request)

        key = self._cache_key(request)
        cached = self._get(key)
        if cached is not None:
            return cached

        resp = call_next(request)
        self._set(key, resp)
        return resp

    async def ahandle(
        self,
        request: CompletionRequest,
        call_next: AsyncNext,
    ) -> CompletionResponse:
        if not self._should_cache(request):
            return await call_next(request)

        key = self._cache_key(request)
        cached = self._get(key)
        if cached is not None:
            return cached

        resp = await call_next(request)
        self._set(key, resp)
        return resp

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of valid (non-expired) cache entries."""
        return len(self._store)

    def clear(self) -> None:
        """Evict all cached entries."""
        with self._lock:
            self._store.clear()
