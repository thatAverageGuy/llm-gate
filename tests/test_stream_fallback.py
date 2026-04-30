"""
tests.test_stream_fallback
~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for streaming fallback (stream_fallback_mode: restart / prefill / user_turn).

All providers are fully mocked — no real API calls made.
"""

from __future__ import annotations

import warnings
from typing import Iterator, AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from llmgate.exceptions import AllProvidersFailedError, RateLimitError
from llmgate.types import CompletionRequest, StreamChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(delta: str, model: str = "model-a", provider: str = "prov") -> StreamChunk:
    return StreamChunk(id="c1", model=model, provider=provider, delta=delta)


def _make_stream(
    *chunks: StreamChunk, then_raise: Exception | None = None
) -> Iterator[StreamChunk]:
    """Return a generator that yields chunks, then optionally raises."""

    def _gen():
        for c in chunks:
            yield c
        if then_raise:
            raise then_raise

    return _gen()


async def _async_gen(
    *chunks: StreamChunk, then_raise: Exception | None = None
) -> AsyncIterator[StreamChunk]:
    for c in chunks:
        yield c
    if then_raise:
        raise then_raise


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MESSAGES = [{"role": "user", "content": "Hello"}]


# ---------------------------------------------------------------------------
# _try_models_stream_sync — restart mode
# ---------------------------------------------------------------------------


class TestRestartModeSync:
    def _run(self, models, side_effects, **kw):
        """Patch providers so each model returns a stream or raises."""
        from llmgate.fallback import _try_models_stream_sync

        def fake_get_provider(model, *a, **kw_):
            prov = MagicMock()
            prov.name = model
            prov.supports_prefill = False
            prov.stream.side_effect = side_effects[model]
            return prov

        with (
            patch("llmgate.fallback._get_provider", side_effect=fake_get_provider),
            patch(
                "llmgate.fallback._build_request",
                return_value=MagicMock(spec=CompletionRequest),
            ),
        ):
            chunks = list(
                _try_models_stream_sync(
                    models,
                    MESSAGES,
                    fallback_on=(RateLimitError,),
                    middleware=None,
                    mode=kw.get("mode", "restart"),
                    stream_resume_prompt=kw.get("stream_resume_prompt"),
                )
            )
        return chunks

    def test_primary_succeeds_no_fallback(self):
        chunks = self._run(
            ["model-a", "model-b"],
            {
                "model-a": [_make_stream(_chunk("Hi", "model-a"))],
                "model-b": [_make_stream(_chunk("Bye", "model-b"))],
            },
        )
        assert len(chunks) == 1
        assert chunks[0].delta == "Hi"
        assert chunks[0].fallback_attempts == []
        assert chunks[0].resumed_from_partial is False

    def test_pre_first_chunk_failure_falls_back(self):
        """Primary raises before yielding any chunk; fallback starts fresh."""
        err = RateLimitError("rate limit", provider="model-a")

        def stream_a():
            raise err
            yield  # make it a generator

        chunks = self._run(
            ["model-a", "model-b"],
            {
                "model-a": [stream_a()],
                "model-b": [_make_stream(_chunk("From B", "model-b"))],
            },
        )
        assert len(chunks) == 1
        assert chunks[0].delta == "From B"
        assert chunks[0].fallback_attempts == ["model-a"]
        # No partial was accumulated before failure
        assert chunks[0].resumed_from_partial is False

    def test_mid_stream_failure_restart(self):
        """Primary yields 2 chunks then fails; fallback gets fresh messages."""
        err = RateLimitError("rate limit", provider="model-a")

        partial_chunks = [_chunk("Hello ", "model-a"), _chunk("World", "model-a")]
        stream_a = _make_stream(*partial_chunks, then_raise=err)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            chunks = self._run(
                ["model-a", "model-b"],
                {
                    "model-a": [stream_a],
                    "model-b": [_make_stream(_chunk("Fresh start", "model-b"))],
                },
                mode="restart",
            )

        assert chunks[0].delta == "Hello "
        assert chunks[1].delta == "World"
        assert chunks[2].delta == "Fresh start"
        assert chunks[2].fallback_attempts == ["model-a"]
        # restart mode — no prefill carry-over
        assert chunks[2].resumed_from_partial is False
        assert len(w) == 1
        assert "model-a" in str(w[0].message)

    def test_all_models_fail_raises(self):
        err = RateLimitError("rate limit", provider="x")

        def dead():
            raise err
            yield

        from llmgate.fallback import _try_models_stream_sync

        with (
            patch("llmgate.fallback._get_provider") as gp,
            patch(
                "llmgate.fallback._build_request",
                return_value=MagicMock(spec=CompletionRequest),
            ),
        ):
            prov = MagicMock()
            prov.name = "x"
            prov.supports_prefill = False
            prov.stream.side_effect = lambda _: dead()
            gp.return_value = prov

            with pytest.raises(AllProvidersFailedError):
                list(
                    _try_models_stream_sync(
                        ["m1", "m2"],
                        MESSAGES,
                        fallback_on=(RateLimitError,),
                        middleware=None,
                        mode="restart",
                        stream_resume_prompt=None,
                    )
                )

    def test_non_fallback_exception_propagates(self):
        """Errors NOT in fallback_on must propagate immediately."""
        from llmgate.fallback import _try_models_stream_sync

        def bad_stream():
            raise ValueError("not a fallback error")
            yield

        with (
            patch("llmgate.fallback._get_provider") as gp,
            patch(
                "llmgate.fallback._build_request",
                return_value=MagicMock(spec=CompletionRequest),
            ),
        ):
            prov = MagicMock()
            prov.name = "x"
            prov.stream.side_effect = lambda _: bad_stream()
            gp.return_value = prov

            with pytest.raises(ValueError, match="not a fallback error"):
                list(
                    _try_models_stream_sync(
                        ["m1", "m2"],
                        MESSAGES,
                        fallback_on=(RateLimitError,),
                        middleware=None,
                        mode="restart",
                        stream_resume_prompt=None,
                    )
                )


# ---------------------------------------------------------------------------
# _try_models_stream_sync — prefill mode
# ---------------------------------------------------------------------------


class TestPrefillModeSync:
    def test_prefill_sent_to_fallback(self):
        """Fallback model receives original messages + partial as assistant message."""
        from llmgate.fallback import _try_models_stream_sync

        err = RateLimitError("rl", provider="model-a")
        partial_chunks = [_chunk("The capital", "model-a")]
        stream_a = _make_stream(*partial_chunks, then_raise=err)

        captured_messages = {}

        def fake_build_request(model, messages, stream, kwargs):
            captured_messages[model] = list(messages)
            req = MagicMock(spec=CompletionRequest)
            req.stream = stream
            return req

        def fake_get_provider(model, *a, **kw):
            prov = MagicMock()
            prov.name = model
            # model-b supports prefill
            prov.supports_prefill = True
            if model == "model-a":
                prov.stream.return_value = stream_a
            else:
                prov.stream.return_value = _make_stream(_chunk(" is Paris.", "model-b"))
            return prov

        with (
            patch("llmgate.fallback._get_provider", side_effect=fake_get_provider),
            patch("llmgate.fallback._build_request", side_effect=fake_build_request),
        ):
            chunks = list(
                _try_models_stream_sync(
                    ["model-a", "model-b"],
                    MESSAGES,
                    fallback_on=(RateLimitError,),
                    middleware=None,
                    mode="prefill",
                    stream_resume_prompt=None,
                )
            )

        assert chunks[0].delta == "The capital"
        assert chunks[1].delta == " is Paris."
        assert chunks[1].resumed_from_partial is True
        assert chunks[1].fallback_attempts == ["model-a"]

        # model-b got the prefill message
        b_msgs = captured_messages["model-b"]
        assert b_msgs[-1].role == "assistant"
        assert b_msgs[-1].content == "The capital"

    def test_prefill_downgrades_to_user_turn_when_unsupported(self):
        """If fallback provider has supports_prefill=False, auto-downgrade to user_turn."""
        from llmgate.fallback import _try_models_stream_sync

        err = RateLimitError("rl", provider="model-a")
        stream_a = _make_stream(_chunk("Hello", "model-a"), then_raise=err)
        captured_messages = {}

        def fake_build_request(model, messages, stream, kwargs):
            captured_messages[model] = list(messages)
            req = MagicMock(spec=CompletionRequest)
            req.stream = stream
            return req

        def fake_get_provider(model, *a, **kw):
            prov = MagicMock()
            prov.name = model
            prov.supports_prefill = False  # both providers don't support prefill
            if model == "model-a":
                prov.stream.return_value = stream_a
            else:
                prov.stream.return_value = _make_stream(_chunk("cont", "model-b"))
            return prov

        with (
            patch("llmgate.fallback._get_provider", side_effect=fake_get_provider),
            patch("llmgate.fallback._build_request", side_effect=fake_build_request),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            list(
                _try_models_stream_sync(
                    ["model-a", "model-b"],
                    MESSAGES,
                    fallback_on=(RateLimitError,),
                    middleware=None,
                    mode="prefill",
                    stream_resume_prompt=None,
                )
            )

        # Should have warned about downgrade
        downgrade_warnings = [x for x in w if "user_turn" in str(x.message)]
        assert len(downgrade_warnings) >= 1

        # model-b gets user-turn style: assistant + user continuation message
        b_msgs = captured_messages["model-b"]
        assert b_msgs[-2].role == "assistant"
        assert b_msgs[-2].content == "Hello"
        assert b_msgs[-1].role == "user"
        assert "Continue" in b_msgs[-1].content


# ---------------------------------------------------------------------------
# _try_models_stream_sync — user_turn mode
# ---------------------------------------------------------------------------


class TestUserTurnModeSync:
    def test_user_turn_messages_sent(self):
        from llmgate.fallback import _try_models_stream_sync

        err = RateLimitError("rl", provider="model-a")
        stream_a = _make_stream(_chunk("Paris is", "model-a"), then_raise=err)
        captured = {}

        def fake_build_request(model, messages, stream, kwargs):
            captured[model] = list(messages)
            req = MagicMock(spec=CompletionRequest)
            req.stream = stream
            return req

        def fake_get_provider(model, *a, **kw):
            prov = MagicMock()
            prov.name = model
            prov.supports_prefill = False
            if model == "model-a":
                prov.stream.return_value = stream_a
            else:
                prov.stream.return_value = _make_stream(_chunk(" great.", "model-b"))
            return prov

        with (
            patch("llmgate.fallback._get_provider", side_effect=fake_get_provider),
            patch("llmgate.fallback._build_request", side_effect=fake_build_request),
        ):
            chunks = list(
                _try_models_stream_sync(
                    ["model-a", "model-b"],
                    MESSAGES,
                    fallback_on=(RateLimitError,),
                    middleware=None,
                    mode="user_turn",
                    stream_resume_prompt="Keep going.",
                )
            )

        b_msgs = captured["model-b"]
        assert b_msgs[-2].role == "assistant"
        assert b_msgs[-2].content == "Paris is"
        assert b_msgs[-1].role == "user"
        assert b_msgs[-1].content == "Keep going."
        assert chunks[-1].resumed_from_partial is True


# ---------------------------------------------------------------------------
# 3-model chain
# ---------------------------------------------------------------------------


class TestThreeModelChain:
    def test_chain_of_three(self):
        """model-1 fails before first chunk, model-2 fails mid-stream, model-3 completes."""
        from llmgate.fallback import _try_models_stream_sync

        err = RateLimitError("rl", provider="x")

        def dead():
            raise err
            yield

        stream_2 = _make_stream(_chunk("Part", "model-2"), then_raise=err)

        def fake_get_provider(model, *a, **kw):
            prov = MagicMock()
            prov.name = model
            prov.supports_prefill = True
            if model == "model-1":
                prov.stream.side_effect = lambda _: dead()
            elif model == "model-2":
                prov.stream.return_value = stream_2
            else:
                prov.stream.return_value = _make_stream(_chunk("ial done", "model-3"))
            return prov

        with (
            patch("llmgate.fallback._get_provider", side_effect=fake_get_provider),
            patch(
                "llmgate.fallback._build_request",
                return_value=MagicMock(spec=CompletionRequest),
            ),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            chunks = list(
                _try_models_stream_sync(
                    ["model-1", "model-2", "model-3"],
                    MESSAGES,
                    fallback_on=(RateLimitError,),
                    middleware=None,
                    mode="restart",
                    stream_resume_prompt=None,
                )
            )

        deltas = [c.delta for c in chunks]
        assert "Part" in deltas
        assert "ial done" in deltas
        assert len(w) == 2  # one warning per fallback event


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------


class TestAsyncVariants:
    @pytest.mark.asyncio
    async def test_async_pre_first_chunk_failure(self):
        from llmgate.fallback import _try_models_stream_async

        err = RateLimitError("rl", provider="model-a")

        async def dead_stream():
            raise err
            yield  # type: ignore[misc]  # unreachable but makes it an async generator

        async def ok_stream():
            yield _chunk("OK from B", "model-b")

        def fake_get_provider(model, *a, **kw):
            prov = MagicMock()
            prov.name = model
            prov.supports_prefill = False
            if model == "model-a":
                prov.astream.return_value = dead_stream()
            else:
                prov.astream.return_value = ok_stream()
            return prov

        with (
            patch("llmgate.fallback._get_provider", side_effect=fake_get_provider),
            patch(
                "llmgate.fallback._build_request",
                return_value=MagicMock(spec=CompletionRequest),
            ),
        ):
            chunks = []
            async for chunk in _try_models_stream_async(
                ["model-a", "model-b"],
                MESSAGES,
                fallback_on=(RateLimitError,),
                middleware=None,
                mode="restart",
                stream_resume_prompt=None,
            ):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].delta == "OK from B"
        assert chunks[0].fallback_attempts == ["model-a"]

    @pytest.mark.asyncio
    async def test_async_all_fail_raises(self):
        from llmgate.fallback import _try_models_stream_async

        err = RateLimitError("rl", provider="x")

        async def dead():
            raise err
            yield

        def fake_get_provider(model, *a, **kw):
            prov = MagicMock()
            prov.name = model
            prov.supports_prefill = False
            prov.astream.return_value = dead()
            return prov

        with (
            patch("llmgate.fallback._get_provider", side_effect=fake_get_provider),
            patch(
                "llmgate.fallback._build_request",
                return_value=MagicMock(spec=CompletionRequest),
            ),
        ):
            with pytest.raises(AllProvidersFailedError):
                async for _ in _try_models_stream_async(
                    ["m1", "m2"],
                    MESSAGES,
                    fallback_on=(RateLimitError,),
                    middleware=None,
                    mode="restart",
                    stream_resume_prompt=None,
                ):
                    pass


# ---------------------------------------------------------------------------
# Public API: completion() no longer raises ValueError for stream + list
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_completion_stream_list_no_longer_raises(self):
        """completion([...], stream=True) must not raise ValueError."""
        from llmgate.completion import completion

        def fake_stream():
            yield _chunk("hi")

        def fake_get_provider(model, *a, **kw):
            prov = MagicMock()
            prov.name = model
            prov.supports_prefill = False
            prov.stream.return_value = fake_stream()
            return prov

        with (
            patch("llmgate.fallback._get_provider", side_effect=fake_get_provider),
            patch(
                "llmgate.fallback._build_request",
                return_value=MagicMock(spec=CompletionRequest),
            ),
        ):
            it = completion(["gpt-4o-mini", "gemini-2.0-flash"], MESSAGES, stream=True)
            chunks = list(it)
        assert len(chunks) >= 1

    def test_stream_fallback_mode_default_is_restart(self):
        """Default stream_fallback_mode is 'restart'."""
        from llmgate.fallback import _try_models_stream_sync
        import inspect

        sig = inspect.signature(_try_models_stream_sync)
        assert sig.parameters["mode"].default == "restart"
