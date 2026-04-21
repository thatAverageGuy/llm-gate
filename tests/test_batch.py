"""
tests/test_batch.py
~~~~~~~~~~~~~~~~~~~
Unit tests for llmgate.batch — batch() and abatch().

All tests are fully mocked; zero real API calls are made.
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

# llmgate/__init__.py exports a function named `batch`, which shadows the
# `llmgate.batch` submodule attribute on the package object.  Using
# sys.modules gives us the actual module so patch.object targets the right
# namespace.
import llmgate.batch  # noqa: F401  — ensure module is registered in sys.modules
_batch_mod = sys.modules["llmgate.batch"]

from llmgate.batch import abatch, batch  # noqa: E402
from llmgate.exceptions import RateLimitError  # noqa: E402
from llmgate.types import (  # noqa: E402
    BatchResult,
    Choice,
    CompletionRequest,
    CompletionResponse,
    Message,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(text: str = "hello", tokens: int = 10, provider: str = "openai") -> CompletionResponse:
    return CompletionResponse(
        id="resp-123",
        model="gpt-4o-mini",
        provider=provider,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=TokenUsage(prompt_tokens=5, completion_tokens=tokens - 5, total_tokens=tokens),
    )


def _make_requests(n: int = 3) -> list[CompletionRequest]:
    return [
        CompletionRequest(
            model="gpt-4o-mini",
            messages=[Message(role="user", content=f"msg {i}")],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Sync: batch()
# ---------------------------------------------------------------------------


class TestBatch:
    def test_successful_batch(self) -> None:
        """All requests succeed — results list is fully populated."""
        responses = [_make_response(f"r{i}", tokens=10 + i) for i in range(3)]
        call_count = 0

        def fake_completion(model: str, messages: Any, **kwargs: Any) -> CompletionResponse:
            nonlocal call_count
            resp = responses[call_count]
            call_count += 1
            return resp

        with patch.object(_batch_mod, "completion", side_effect=fake_completion):
            result = batch(_make_requests(3))

        assert isinstance(result, BatchResult)
        assert result.successful == 3
        assert result.failed == 0
        assert len(result.errors) == 0
        assert all(r is not None for r in result.results)

    def test_token_aggregation(self) -> None:
        """total_tokens sums usage from all successful responses."""
        responses = [_make_response(tokens=20), _make_response(tokens=30), _make_response(tokens=50)]
        idx = 0

        def fake_completion(model: str, messages: Any, **kwargs: Any) -> CompletionResponse:
            nonlocal idx
            r = responses[idx]
            idx += 1
            return r

        with patch.object(_batch_mod, "completion", side_effect=fake_completion):
            result = batch(_make_requests(3))

        assert result.total_tokens == 100

    def test_partial_failure(self) -> None:
        """Some requests fail — errors collected, successful results preserved."""
        def fake_completion(model: str, messages: Any, **kwargs: Any) -> CompletionResponse:
            content = messages[0].content
            if content == "msg 1":
                raise RateLimitError("quota exceeded", provider="openai")
            return _make_response(content)

        with patch.object(_batch_mod, "completion", side_effect=fake_completion):
            result = batch(_make_requests(3))

        assert result.successful == 2
        assert result.failed == 1
        assert len(result.errors) == 1
        err = result.errors[0]
        assert err.index == 1
        assert err.error_type == "RateLimitError"
        assert result.results[1] is None

    def test_fail_fast_raises(self) -> None:
        """fail_fast=True raises on the first error."""
        def fake_completion(model: str, messages: Any, **kwargs: Any) -> CompletionResponse:
            raise RateLimitError("quota", provider="openai")

        with patch.object(_batch_mod, "completion", side_effect=fake_completion):
            with pytest.raises(RateLimitError):
                batch(_make_requests(3), fail_fast=True)

    def test_order_preservation(self) -> None:
        """Results are in the same order as the input requests."""
        responses = [_make_response(f"result-{i}") for i in range(5)]
        idx = 0
        lock = __import__("threading").Lock()

        def fake_completion(model: str, messages: Any, **kwargs: Any) -> CompletionResponse:
            nonlocal idx
            with lock:
                content = messages[0].content
            # Find which index this is
            for i, r in enumerate(responses):
                if f"result-{i}" not in content:
                    pass
            # Return based on message content order
            msg_idx = int(content.split()[-1])
            return responses[msg_idx]

        with patch.object(_batch_mod, "completion", side_effect=fake_completion):
            result = batch(_make_requests(5), max_concurrency=5)

        for i, r in enumerate(result.results):
            assert r is not None
            assert r.text == f"result-{i}"

    def test_dict_requests_normalised(self) -> None:
        """Plain dicts are accepted and normalised to CompletionRequest objects."""
        dict_requests = [
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]},
        ]

        with patch.object(_batch_mod, "completion", return_value=_make_response()):
            result = batch(dict_requests)

        assert result.successful == 2

    def test_concurrency_limit(self) -> None:
        """max_concurrency is respected — active thread count never exceeds it."""
        import threading

        active = 0
        peak = 0
        lock = threading.Lock()

        def fake_completion(model: str, messages: Any, **kwargs: Any) -> CompletionResponse:
            nonlocal active, peak
            with lock:
                active += 1
                if active > peak:
                    peak = active
            import time
            time.sleep(0.02)
            with lock:
                active -= 1
            return _make_response()

        with patch.object(_batch_mod, "completion", side_effect=fake_completion):
            batch(_make_requests(10), max_concurrency=3)

        assert peak <= 3

    def test_timeout_raises_batch_timeout_error(self) -> None:
        """Requests that exceed timeout are recorded as BatchTimeoutError."""
        import time

        def slow_completion(model: str, messages: Any, **kwargs: Any) -> CompletionResponse:
            time.sleep(5)
            return _make_response()

        with patch.object(_batch_mod, "completion", side_effect=slow_completion):
            result = batch(_make_requests(2), timeout=0.05)

        assert result.failed == 2
        for err in result.errors:
            assert err.error_type == "BatchTimeoutError"

    def test_success_rate(self) -> None:
        """success_rate property returns correct fraction."""
        call_n = 0

        def fake_completion(model: str, messages: Any, **kwargs: Any) -> CompletionResponse:
            nonlocal call_n
            call_n += 1
            if call_n % 2 == 0:
                raise RateLimitError("rate limit", provider="openai")
            return _make_response()

        with patch.object(_batch_mod, "completion", side_effect=fake_completion):
            result = batch(_make_requests(4))

        assert result.success_rate == pytest.approx(0.5)

    def test_empty_requests(self) -> None:
        """Empty input returns an empty BatchResult."""
        result = batch([])
        assert result.successful == 0
        assert result.failed == 0
        assert result.total_tokens == 0
        assert result.results == []


# ---------------------------------------------------------------------------
# Async: abatch()
# ---------------------------------------------------------------------------


class TestABatch:
    @pytest.mark.asyncio
    async def test_async_successful_batch(self) -> None:
        """All async requests succeed."""
        responses = [_make_response(f"r{i}") for i in range(3)]
        idx = 0

        async def fake_acompletion(model: str, messages: Any, **kwargs: Any) -> CompletionResponse:
            nonlocal idx
            r = responses[idx]
            idx += 1
            return r

        with patch.object(_batch_mod, "acompletion", side_effect=fake_acompletion):
            result = await abatch(_make_requests(3))

        assert result.successful == 3
        assert result.failed == 0

    @pytest.mark.asyncio
    async def test_async_partial_failure(self) -> None:
        """Failed async requests are collected in errors."""
        async def fake_acompletion(model: str, messages: Any, **kwargs: Any) -> CompletionResponse:
            if messages[0].content == "msg 0":
                raise RateLimitError("quota", provider="openai")
            return _make_response()

        with patch.object(_batch_mod, "acompletion", side_effect=fake_acompletion):
            result = await abatch(_make_requests(3))

        assert result.successful == 2
        assert result.failed == 1
        assert result.errors[0].index == 0

    @pytest.mark.asyncio
    async def test_async_fail_fast(self) -> None:
        """fail_fast=True raises immediately on first async error."""
        async def fake_acompletion(model: str, messages: Any, **kwargs: Any) -> CompletionResponse:
            raise RateLimitError("quota", provider="openai")

        with patch.object(_batch_mod, "acompletion", side_effect=fake_acompletion):
            with pytest.raises(RateLimitError):
                await abatch(_make_requests(3), fail_fast=True)

    @pytest.mark.asyncio
    async def test_async_concurrency_limit(self) -> None:
        """Semaphore caps concurrent async requests."""
        active = 0
        peak = 0

        async def fake_acompletion(model: str, messages: Any, **kwargs: Any) -> CompletionResponse:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return _make_response()

        with patch.object(_batch_mod, "acompletion", side_effect=fake_acompletion):
            await abatch(_make_requests(10), max_concurrency=3)

        assert peak <= 3

    @pytest.mark.asyncio
    async def test_async_timeout(self) -> None:
        """Async requests that exceed timeout are recorded as BatchTimeoutError."""
        async def slow_acompletion(model: str, messages: Any, **kwargs: Any) -> CompletionResponse:
            await asyncio.sleep(10)
            return _make_response()

        with patch.object(_batch_mod, "acompletion", side_effect=slow_acompletion):
            result = await abatch(_make_requests(2), timeout=0.05)

        assert result.failed == 2
        for err in result.errors:
            assert err.error_type == "BatchTimeoutError"

    @pytest.mark.asyncio
    async def test_async_token_aggregation(self) -> None:
        """total_tokens aggregated correctly in async path."""
        responses = [_make_response(tokens=t) for t in [10, 20, 30]]
        idx = 0

        async def fake_acompletion(model: str, messages: Any, **kwargs: Any) -> CompletionResponse:
            nonlocal idx
            r = responses[idx]
            idx += 1
            return r

        with patch.object(_batch_mod, "acompletion", side_effect=fake_acompletion):
            result = await abatch(_make_requests(3))

        assert result.total_tokens == 60

    @pytest.mark.asyncio
    async def test_async_order_preservation(self) -> None:
        """Async results maintain input order despite concurrent execution."""
        async def fake_acompletion(model: str, messages: Any, **kwargs: Any) -> CompletionResponse:
            # Simulate variable latency
            msg_idx = int(messages[0].content.split()[-1])
            await asyncio.sleep(0.01 * (3 - msg_idx))  # earlier indices are slower
            return _make_response(f"result-{msg_idx}")

        with patch.object(_batch_mod, "acompletion", side_effect=fake_acompletion):
            result = await abatch(_make_requests(3), max_concurrency=3)

        for i, r in enumerate(result.results):
            assert r is not None
            assert r.text == f"result-{i}"

    @pytest.mark.asyncio
    async def test_async_dict_requests(self) -> None:
        """Plain dicts are accepted in the async path."""
        dict_requests = [
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        ]
        async_mock = AsyncMock(return_value=_make_response())

        with patch.object(_batch_mod, "acompletion", side_effect=async_mock):
            result = await abatch(dict_requests)

        assert result.successful == 1
