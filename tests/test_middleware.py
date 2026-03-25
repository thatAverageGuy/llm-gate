"""
Tests for the middleware system — all mocked, no real API calls.
"""
from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from llmgate.exceptions import ProviderAPIError, RateLimitError
from llmgate.middleware import (
    CacheMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
    RetryMiddleware,
)
from llmgate.middleware.base import BaseMiddleware
from llmgate.types import (
    Choice, CompletionRequest, CompletionResponse, Message, TokenUsage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _req(model: str = "gpt-4o-mini") -> CompletionRequest:
    return CompletionRequest(
        model=model,
        messages=[Message(role="user", content="hello")],
    )


def _resp(model: str = "gpt-4o-mini", text: str = "Hi!") -> CompletionResponse:
    return CompletionResponse(
        id="test-id",
        model=model,
        provider="openai",
        choices=[Choice(index=0, message=Message(role="assistant", content=text), finish_reason="stop")],
        usage=TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )


# ---------------------------------------------------------------------------
# BaseMiddleware
# ---------------------------------------------------------------------------


class _ConcreteMiddleware(BaseMiddleware):
    """Simple pass-through middleware for testing."""
    def handle(self, request, call_next):
        return call_next(request)


class TestBaseMiddleware:
    def test_handle_called(self):
        mw = _ConcreteMiddleware()
        req = _req()
        expected = _resp()
        result = mw.handle(req, lambda r: expected)
        assert result is expected

    @pytest.mark.asyncio
    async def test_ahandle_defaults_to_passthrough(self):
        """Default ahandle delegates to call_next without blocking."""
        class SyncMW(BaseMiddleware):
            def handle(self, request, call_next):
                return call_next(request)
        mw = SyncMW()
        req = _req()
        expected = _resp()
        async def _async_next(_r): return expected
        result = await mw.ahandle(req, _async_next)
        assert result is expected

    def test_stream_handle_passthrough(self):
        mw = _ConcreteMiddleware()
        req = _req()
        sentinel = iter([])
        result = mw.stream_handle(req, lambda r: sentinel)
        assert result is sentinel


# ---------------------------------------------------------------------------
# LoggingMiddleware
# ---------------------------------------------------------------------------


class TestLoggingMiddleware:
    def test_logs_on_success(self, caplog):
        mw = LoggingMiddleware(level="DEBUG")
        req = _req()
        resp = _resp()
        with caplog.at_level(logging.DEBUG, logger="llmgate"):
            result = mw.handle(req, lambda r: resp)
        assert result is resp
        messages = [r.message for r in caplog.records]
        assert any("request" in m for m in messages)
        assert any("response" in m for m in messages)

    def test_logs_error_and_reraises(self, caplog):
        mw = LoggingMiddleware(level="DEBUG")
        req = _req()
        def _raise(_r):
            raise ProviderAPIError("boom", provider="openai")
        with caplog.at_level(logging.DEBUG, logger="llmgate"):
            with pytest.raises(ProviderAPIError):
                mw.handle(req, _raise)
        assert any("error" in r.message.lower() for r in caplog.records)

    def test_mask_content(self, caplog):
        mw = LoggingMiddleware(level="DEBUG", mask_content=True)
        req = _req()
        with caplog.at_level(logging.DEBUG, logger="llmgate"):
            mw.handle(req, lambda r: _resp())
        # The raw content should not appear in any log record's extra
        all_text = " ".join(str(r.__dict__) for r in caplog.records)
        assert "hello" not in all_text

    @pytest.mark.asyncio
    async def test_async_logs(self, caplog):
        mw = LoggingMiddleware(level="DEBUG")
        req = _req()
        resp = _resp()
        async def _next(_r): return resp
        with caplog.at_level(logging.DEBUG, logger="llmgate"):
            result = await mw.ahandle(req, _next)
        assert result is resp
        assert any("response" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# RetryMiddleware
# ---------------------------------------------------------------------------


class TestRetryMiddleware:
    def test_no_retry_on_success(self):
        mw = RetryMiddleware(max_retries=3)
        calls = []
        def _next(r):
            calls.append(1)
            return _resp()
        mw.handle(_req(), _next)
        assert len(calls) == 1

    def test_retries_on_rate_limit(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda _: None)
        mw = RetryMiddleware(max_retries=2, backoff_factor=0.0)
        calls = []
        def _next(r):
            calls.append(1)
            if len(calls) < 3:
                raise RateLimitError("429", provider="groq")
            return _resp()
        result = mw.handle(_req(), _next)
        assert len(calls) == 3
        assert result.text == "Hi!"

    def test_raises_after_max_retries(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda _: None)
        mw = RetryMiddleware(max_retries=2, backoff_factor=0.0)
        def _next(r):
            raise RateLimitError("429", provider="groq")
        with pytest.raises(RateLimitError):
            mw.handle(_req(), _next)

    def test_no_retry_on_4xx(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda _: None)
        mw = RetryMiddleware(max_retries=3, backoff_factor=0.0)
        calls = []
        def _next(r):
            calls.append(1)
            raise ProviderAPIError("400 bad request", provider="openai", status_code=400)
        with pytest.raises(ProviderAPIError):
            mw.handle(_req(), _next)
        assert len(calls) == 1  # no retries for 400

    def test_retry_on_5xx(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda _: None)
        mw = RetryMiddleware(max_retries=2, backoff_factor=0.0, retry_on_provider_errors=True)
        calls = []
        def _next(r):
            calls.append(1)
            if len(calls) < 2:
                raise ProviderAPIError("500", provider="openai", status_code=500)
            return _resp()
        result = mw.handle(_req(), _next)
        assert len(calls) == 2
        assert result.text == "Hi!"

    @pytest.mark.asyncio
    async def test_async_retries(self):
        mw = RetryMiddleware(max_retries=2, backoff_factor=0.0)
        calls = []
        async def _next(r):
            calls.append(1)
            if len(calls) < 2:
                raise RateLimitError("429", provider="groq")
            return _resp()
        await mw.ahandle(_req(), _next)
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# CacheMiddleware
# ---------------------------------------------------------------------------


class TestCacheMiddleware:
    def test_cache_hit_skips_provider(self):
        cache = CacheMiddleware(ttl=60)
        calls = []
        def _next(r):
            calls.append(1)
            return _resp()
        # First call — miss
        r1 = cache.handle(_req(), _next)
        # Second call — hit
        r2 = cache.handle(_req(), _next)
        assert len(calls) == 1
        assert r1 is r2

    def test_cache_miss_after_ttl(self, monkeypatch):
        cache = CacheMiddleware(ttl=0.01)
        calls = []
        def _next(r):
            calls.append(1)
            return _resp()
        cache.handle(_req(), _next)
        time.sleep(0.02)  # expire
        cache.handle(_req(), _next)
        assert len(calls) == 2

    def test_stream_not_cached(self):
        cache = CacheMiddleware(ttl=60)
        calls = []
        req = CompletionRequest(
            model="gpt-4o-mini",
            messages=[Message(role="user", content="hello")],
            stream=True,
        )
        def _next(r):
            calls.append(1)
            return _resp()
        cache.handle(req, _next)
        cache.handle(req, _next)
        assert len(calls) == 2  # not cached

    def test_tools_not_cached(self):
        from llmgate.types import FunctionDefinition, ToolDefinition
        cache = CacheMiddleware(ttl=60)
        calls = []
        req = CompletionRequest(
            model="gpt-4o-mini",
            messages=[Message(role="user", content="hello")],
            tools=[ToolDefinition(function=FunctionDefinition(
                name="f", description="d", parameters={}
            ))],
        )
        def _next(r):
            calls.append(1)
            return _resp()
        cache.handle(req, _next)
        cache.handle(req, _next)
        assert len(calls) == 2  # not cached

    def test_maxsize_eviction(self):
        cache = CacheMiddleware(ttl=60, maxsize=2)
        models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        call_count = []
        def _next(r):
            call_count.append(1)
            return _resp(model=r.model)
        for m in models:
            cache.handle(_req(m), _next)
        assert cache.size == 2  # oldest evicted

    def test_clear(self):
        cache = CacheMiddleware(ttl=60)
        calls = []
        def _next(r):
            calls.append(1)
            return _resp()
        cache.handle(_req(), _next)
        cache.clear()
        cache.handle(_req(), _next)
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_async_cache_hit(self):
        cache = CacheMiddleware(ttl=60)
        calls = []
        async def _next(r):
            calls.append(1)
            return _resp()
        r1 = await cache.ahandle(_req(), _next)
        r2 = await cache.ahandle(_req(), _next)
        assert len(calls) == 1
        assert r1 is r2


# ---------------------------------------------------------------------------
# RateLimitMiddleware
# ---------------------------------------------------------------------------


class TestRateLimitMiddleware:
    def test_allows_within_limit(self):
        mw = RateLimitMiddleware(requests_per_minute=600, raise_on_limit=True)
        result = mw.handle(_req(), lambda r: _resp())
        assert result.text == "Hi!"

    def test_raises_when_exceeded(self):
        # Very low rate — immediately exhausted
        mw = RateLimitMiddleware(requests_per_minute=1, burst=1, raise_on_limit=True)
        mw.handle(_req(), lambda r: _resp())  # consume the 1 token
        with pytest.raises(RateLimitError, match="Client-side rate limit exceeded"):
            mw.handle(_req(), lambda r: _resp())

    @pytest.mark.asyncio
    async def test_async_raises_when_exceeded(self):
        mw = RateLimitMiddleware(requests_per_minute=1, burst=1, raise_on_limit=True)
        async def _next(r): return _resp()
        await mw.ahandle(_req(), _next)
        with pytest.raises(RateLimitError):
            await mw.ahandle(_req(), _next)


# ---------------------------------------------------------------------------
# Middleware chain composition (via LLMGate)
# ---------------------------------------------------------------------------


class TestMiddlewareChain:
    def test_chain_order(self):
        """Middlewares execute in correct left-to-right order."""
        from llmgate.gate import _build_sync_chain

        order = []

        class TraceA(BaseMiddleware):
            def handle(self, req, call_next):
                order.append("A-before")
                r = call_next(req)
                order.append("A-after")
                return r

        class TraceB(BaseMiddleware):
            def handle(self, req, call_next):
                order.append("B-before")
                r = call_next(req)
                order.append("B-after")
                return r

        chain = _build_sync_chain([TraceA(), TraceB()], lambda r: _resp())
        chain(_req())
        assert order == ["A-before", "B-before", "B-after", "A-after"]

    def test_gate_completion_with_middleware(self):
        """LLMGate.completion runs middleware and reaches provider."""
        from llmgate.gate import LLMGate
        touched = []

        class TouchMW(BaseMiddleware):
            def handle(self, req, call_next):
                touched.append(True)
                return call_next(req)

        mock_provider = MagicMock()
        mock_provider.complete.return_value = _resp()

        gate = LLMGate(middleware=[TouchMW()])
        # _get_or_create_provider is imported into gate.py from completion.py;
        # patch it at the gate module level so provider instantiation is bypassed.
        with patch("llmgate.gate._get_or_create_provider", return_value=mock_provider):
            resp = gate.completion("gpt-4o-mini", [{"role": "user", "content": "hi"}])

        assert touched == [True]
        assert resp.text == "Hi!"
        mock_provider.complete.assert_called_once()
