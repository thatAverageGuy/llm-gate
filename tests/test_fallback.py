"""
Tests for the Fallback / Routing feature (v0.6).

All provider calls are fully mocked — no live API keys required.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmgate import LLMGate, acompletion, completion
from llmgate.exceptions import (
    AllProvidersFailedError,
    AuthError,
    ModelNotFoundError,
    ProviderAPIError,
    RateLimitError,
)
from llmgate.middleware import FallbackMiddleware
from llmgate.types import Choice, CompletionResponse, Message, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_cache() -> None:
    sys.modules["llmgate.completion"]._provider_cache.clear()


def _fake_response(
    provider: str = "openai", model: str = "gpt-4o-mini"
) -> CompletionResponse:
    return CompletionResponse(
        id="fake-id",
        model=model,
        provider=provider,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content="hello from " + provider),
                finish_reason="stop",
            )
        ],
        usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


MESSAGES = [{"role": "user", "content": "Hello!"}]

# Patch targets: fallback.py and middleware/fallback.py expose these at module-level
_PATCH_GET_PROVIDER = "llmgate.fallback._get_provider"
_PATCH_BUILD_REQUEST = "llmgate.fallback._build_request"
_PATCH_MW_GET_PROVIDER = "llmgate.middleware.fallback._get_provider"


# ---------------------------------------------------------------------------
# AllProvidersFailedError
# ---------------------------------------------------------------------------


class TestAllProvidersFailedError:
    def test_message_includes_all_failures(self):
        err = AllProvidersFailedError(
            [
                ("gpt-4o-mini", RateLimitError("rate limited", provider="openai")),
                (
                    "groq/llama-3.1-8b-instant",
                    ProviderAPIError("server error", provider="groq"),
                ),
            ]
        )
        assert "gpt-4o-mini" in str(err)
        assert "groq/llama-3.1-8b-instant" in str(err)

    def test_errors_list_preserved(self):
        exc1 = RateLimitError("rate limited", provider="openai")
        exc2 = AuthError("bad key", provider="groq")
        err = AllProvidersFailedError([("model-a", exc1), ("model-b", exc2)])
        assert err.errors[0] == ("model-a", exc1)
        assert err.errors[1] == ("model-b", exc2)


# ---------------------------------------------------------------------------
# Single-string model — unchanged behaviour
# ---------------------------------------------------------------------------


class TestSingleModelUnchanged:
    def test_single_string_routes_normally(self):
        with (
            patch("llmgate.providers.openai.OpenAIProvider.complete") as mock,
            patch(
                "llmgate.providers.openai.OpenAIProvider.__init__", return_value=None
            ),
        ):
            mock.return_value = _fake_response("openai")
            _clear_cache()
            resp = completion("gpt-4o-mini", MESSAGES, api_key="k")
            assert resp.provider == "openai"
            assert resp.fallback_attempts == []

    def test_single_string_propagates_error(self):
        _clear_cache()
        with pytest.raises(ModelNotFoundError):
            completion("totally-unknown-model-xyz", MESSAGES)


# ---------------------------------------------------------------------------
# completion(model=[...]) — fallback list
# ---------------------------------------------------------------------------


class TestCompletionFallbackList:
    def _make_mock_request(self):
        req = MagicMock()
        req.model_copy = MagicMock(
            side_effect=lambda update=None, **_: _fake_response(
                "groq", "groq/llama-3.1-8b-instant"
            ).model_copy(update=update or {})
        )
        return req

    def test_first_model_succeeds_no_fallback(self):
        openai_mock = MagicMock()
        openai_mock.complete.return_value = _fake_response("openai")

        with (
            patch(_PATCH_GET_PROVIDER, return_value=openai_mock),
            patch(_PATCH_BUILD_REQUEST, return_value=MagicMock()),
        ):
            _clear_cache()
            resp = completion(["gpt-4o-mini", "groq/llama-3.1-8b-instant"], MESSAGES)
            assert resp.provider == "openai"
            assert resp.fallback_attempts == []

    def test_fallback_on_rate_limit(self):
        openai_mock = MagicMock()
        openai_mock.complete.side_effect = RateLimitError("rl", provider="openai")
        groq_mock = MagicMock()
        groq_mock.complete.return_value = _fake_response(
            "groq", "groq/llama-3.1-8b-instant"
        )

        with (
            patch(_PATCH_GET_PROVIDER, side_effect=[openai_mock, groq_mock]),
            patch(_PATCH_BUILD_REQUEST, return_value=MagicMock()),
        ):
            _clear_cache()
            resp = completion(["gpt-4o-mini", "groq/llama-3.1-8b-instant"], MESSAGES)
            assert resp.provider == "groq"
            assert resp.fallback_attempts == ["gpt-4o-mini"]

    def test_fallback_on_provider_api_error(self):
        openai_mock = MagicMock()
        openai_mock.complete.side_effect = ProviderAPIError("500", provider="openai")
        groq_mock = MagicMock()
        groq_mock.complete.return_value = _fake_response("groq")

        with (
            patch(_PATCH_GET_PROVIDER, side_effect=[openai_mock, groq_mock]),
            patch(_PATCH_BUILD_REQUEST, return_value=MagicMock()),
        ):
            _clear_cache()
            resp = completion(["gpt-4o-mini", "groq/llama-3.1-8b-instant"], MESSAGES)
            assert resp.fallback_attempts == ["gpt-4o-mini"]

    def test_fallback_on_auth_error(self):
        """Option B: AuthError also triggers fallback."""
        openai_mock = MagicMock()
        openai_mock.complete.side_effect = AuthError("bad key", provider="openai")
        groq_mock = MagicMock()
        groq_mock.complete.return_value = _fake_response("groq")

        with (
            patch(_PATCH_GET_PROVIDER, side_effect=[openai_mock, groq_mock]),
            patch(_PATCH_BUILD_REQUEST, return_value=MagicMock()),
        ):
            _clear_cache()
            resp = completion(["gpt-4o-mini", "groq/llama-3.1-8b-instant"], MESSAGES)
            assert resp.provider == "groq"
            assert resp.fallback_attempts == ["gpt-4o-mini"]

    def test_no_fallback_on_other_errors(self):
        """Non-fallback errors (e.g. ModelNotFoundError) propagate immediately."""
        openai_mock = MagicMock()
        openai_mock.complete.side_effect = ModelNotFoundError("bad-model")

        with (
            patch(_PATCH_GET_PROVIDER, return_value=openai_mock),
            patch(_PATCH_BUILD_REQUEST, return_value=MagicMock()),
        ):
            _clear_cache()
            with pytest.raises(ModelNotFoundError):
                completion(["gpt-4o-mini", "groq/llama-3.1-8b-instant"], MESSAGES)

    def test_all_fail_raises_all_providers_failed(self):
        failing = MagicMock()
        failing.complete.side_effect = RateLimitError("rl", provider="x")

        with (
            patch(_PATCH_GET_PROVIDER, return_value=failing),
            patch(_PATCH_BUILD_REQUEST, return_value=MagicMock()),
        ):
            _clear_cache()
            with pytest.raises(AllProvidersFailedError) as exc_info:
                completion(["gpt-4o-mini", "groq/llama-3.1-8b-instant"], MESSAGES)
            assert len(exc_info.value.errors) == 2
            assert exc_info.value.errors[0][0] == "gpt-4o-mini"
            assert exc_info.value.errors[1][0] == "groq/llama-3.1-8b-instant"

    def test_three_model_chain_second_succeeds(self):
        openai_mock = MagicMock()
        openai_mock.complete.side_effect = RateLimitError("rl", provider="openai")
        groq_mock = MagicMock()
        groq_mock.complete.return_value = _fake_response("groq")
        gemini_mock = MagicMock()
        gemini_mock.complete.return_value = _fake_response("gemini")

        with (
            patch(
                _PATCH_GET_PROVIDER, side_effect=[openai_mock, groq_mock, gemini_mock]
            ),
            patch(_PATCH_BUILD_REQUEST, return_value=MagicMock()),
        ):
            _clear_cache()
            resp = completion(
                ["gpt-4o-mini", "groq/llama-3.1-8b-instant", "gemini-2.5-flash-lite"],
                MESSAGES,
            )
            assert resp.provider == "groq"
            assert resp.fallback_attempts == ["gpt-4o-mini"]
            gemini_mock.complete.assert_not_called()

    def test_fallback_attempts_all_three_fail(self):
        failing = MagicMock()
        failing.complete.side_effect = RateLimitError("rl", provider="x")

        with (
            patch(_PATCH_GET_PROVIDER, return_value=failing),
            patch(_PATCH_BUILD_REQUEST, return_value=MagicMock()),
        ):
            _clear_cache()
            with pytest.raises(AllProvidersFailedError) as exc_info:
                completion(["a", "b", "c"], MESSAGES)
            assert len(exc_info.value.errors) == 3

    def test_stream_true_with_list_now_works(self):
        """stream=True + model list no longer raises ValueError; uses stream fallback."""
        from llmgate.types import StreamChunk

        def fake_stream():
            yield StreamChunk(
                id="c1", model="gpt-4o-mini", provider="openai", delta="hi"
            )

        prov = MagicMock()
        prov.name = "openai"
        prov.supports_prefill = False
        prov.stream.return_value = fake_stream()

        with (
            patch("llmgate.fallback._get_provider", return_value=prov),
            patch("llmgate.fallback._build_request", return_value=MagicMock()),
        ):
            _clear_cache()
            it = completion(
                ["gpt-4o-mini", "groq/llama-3.1-8b-instant"], MESSAGES, stream=True
            )
            chunks = list(it)
        assert len(chunks) >= 1
        assert chunks[0].delta == "hi"

    def test_custom_fallback_on_excludes_auth_error(self):
        """Custom fallback_on=(RateLimitError,) does NOT fall back on AuthError."""
        openai_mock = MagicMock()
        openai_mock.complete.side_effect = AuthError("bad key", provider="openai")

        with (
            patch(_PATCH_GET_PROVIDER, return_value=openai_mock),
            patch(_PATCH_BUILD_REQUEST, return_value=MagicMock()),
        ):
            _clear_cache()
            with pytest.raises(AuthError):
                completion(
                    ["gpt-4o-mini", "groq/llama-3.1-8b-instant"],
                    MESSAGES,
                    fallback_on=(RateLimitError,),
                )


# ---------------------------------------------------------------------------
# acompletion(model=[...]) — async fallback
# ---------------------------------------------------------------------------


class TestACompletionFallback:
    @pytest.mark.asyncio
    async def test_async_first_model_succeeds(self):
        openai_mock = MagicMock()
        openai_mock.acomplete = AsyncMock(return_value=_fake_response("openai"))

        with (
            patch(_PATCH_GET_PROVIDER, return_value=openai_mock),
            patch(_PATCH_BUILD_REQUEST, return_value=MagicMock()),
        ):
            _clear_cache()
            resp = await acompletion(
                ["gpt-4o-mini", "groq/llama-3.1-8b-instant"], MESSAGES
            )
            assert resp.provider == "openai"
            assert resp.fallback_attempts == []

    @pytest.mark.asyncio
    async def test_async_fallback_on_rate_limit(self):
        openai_mock = MagicMock()
        openai_mock.acomplete = AsyncMock(
            side_effect=RateLimitError("rl", provider="openai")
        )
        groq_mock = MagicMock()
        groq_mock.acomplete = AsyncMock(return_value=_fake_response("groq"))

        with (
            patch(_PATCH_GET_PROVIDER, side_effect=[openai_mock, groq_mock]),
            patch(_PATCH_BUILD_REQUEST, return_value=MagicMock()),
        ):
            _clear_cache()
            resp = await acompletion(
                ["gpt-4o-mini", "groq/llama-3.1-8b-instant"], MESSAGES
            )
            assert resp.provider == "groq"
            assert resp.fallback_attempts == ["gpt-4o-mini"]

    @pytest.mark.asyncio
    async def test_async_all_fail(self):
        failing = MagicMock()
        failing.acomplete = AsyncMock(side_effect=RateLimitError("rl", provider="x"))

        with (
            patch(_PATCH_GET_PROVIDER, return_value=failing),
            patch(_PATCH_BUILD_REQUEST, return_value=MagicMock()),
        ):
            _clear_cache()
            with pytest.raises(AllProvidersFailedError):
                await acompletion(
                    ["gpt-4o-mini", "groq/llama-3.1-8b-instant"], MESSAGES
                )

    @pytest.mark.asyncio
    async def test_async_stream_with_list_now_works(self):
        """acompletion stream=True + model list no longer raises ValueError."""
        from llmgate.types import StreamChunk

        async def fake_astream():
            yield StreamChunk(
                id="c1", model="gpt-4o-mini", provider="openai", delta="async hi"
            )

        prov = MagicMock()
        prov.name = "openai"
        prov.supports_prefill = False
        prov.astream.return_value = fake_astream()

        with (
            patch("llmgate.fallback._get_provider", return_value=prov),
            patch("llmgate.fallback._build_request", return_value=MagicMock()),
        ):
            _clear_cache()
            it = await acompletion(
                ["gpt-4o-mini", "groq/llama-3.1-8b-instant"], MESSAGES, stream=True
            )
            chunks = []
            async for chunk in it:
                chunks.append(chunk)
        assert len(chunks) >= 1
        assert chunks[0].delta == "async hi"


# ---------------------------------------------------------------------------
# LLMGate(fallback_chain=[...])
# ---------------------------------------------------------------------------


class TestLLMGateFallbackChain:
    def test_gate_with_chain_falls_back(self):
        openai_mock = MagicMock()
        openai_mock.complete.side_effect = RateLimitError("rl", provider="openai")
        groq_mock = MagicMock()
        groq_mock.complete.return_value = _fake_response("groq")

        gate = LLMGate(fallback_chain=["gpt-4o-mini", "groq/llama-3.1-8b-instant"])

        with (
            patch(_PATCH_GET_PROVIDER, side_effect=[openai_mock, groq_mock]),
            patch(_PATCH_BUILD_REQUEST, return_value=MagicMock()),
        ):
            _clear_cache()
            resp = gate.completion(messages=MESSAGES)
            assert resp.provider == "groq"
            assert resp.fallback_attempts == ["gpt-4o-mini"]

    def test_gate_without_chain_behaves_normally(self):
        gate = LLMGate()
        with (
            patch("llmgate.providers.openai.OpenAIProvider.complete") as mock,
            patch(
                "llmgate.providers.openai.OpenAIProvider.__init__", return_value=None
            ),
        ):
            mock.return_value = _fake_response("openai")
            _clear_cache()
            resp = gate.completion("gpt-4o-mini", MESSAGES, api_key="k")
            assert resp.provider == "openai"

    def test_gate_chain_all_fail_raises(self):
        failing = MagicMock()
        failing.complete.side_effect = RateLimitError("rl", provider="x")

        gate = LLMGate(fallback_chain=["gpt-4o-mini", "groq/llama-3.1-8b-instant"])

        with (
            patch(_PATCH_GET_PROVIDER, return_value=failing),
            patch(_PATCH_BUILD_REQUEST, return_value=MagicMock()),
        ):
            _clear_cache()
            with pytest.raises(AllProvidersFailedError):
                gate.completion(messages=MESSAGES)

    @pytest.mark.asyncio
    async def test_gate_async_fallback_chain(self):
        openai_mock = MagicMock()
        openai_mock.acomplete = AsyncMock(
            side_effect=RateLimitError("rl", provider="openai")
        )
        groq_mock = MagicMock()
        groq_mock.acomplete = AsyncMock(return_value=_fake_response("groq"))

        gate = LLMGate(fallback_chain=["gpt-4o-mini", "groq/llama-3.1-8b-instant"])

        with (
            patch(_PATCH_GET_PROVIDER, side_effect=[openai_mock, groq_mock]),
            patch(_PATCH_BUILD_REQUEST, return_value=MagicMock()),
        ):
            _clear_cache()
            resp = await gate.acompletion(messages=MESSAGES)
            assert resp.provider == "groq"
            assert resp.fallback_attempts == ["gpt-4o-mini"]

    def test_gate_no_model_no_chain_raises(self):
        gate = LLMGate()
        with pytest.raises(ValueError, match="Provide a model string"):
            gate.completion(messages=MESSAGES)


# ---------------------------------------------------------------------------
# FallbackMiddleware
# ---------------------------------------------------------------------------


class TestFallbackMiddleware:
    def test_primary_succeeds_no_fallback(self):
        mw = FallbackMiddleware(models=["groq/llama-3.1-8b-instant"])
        fake = _fake_response("openai")
        req = MagicMock()
        req.model = "gpt-4o-mini"
        resp = mw.handle(req, lambda r: fake)
        assert resp == fake

    def test_fallback_triggered_on_rate_limit(self):
        mw = FallbackMiddleware(models=["groq/llama-3.1-8b-instant"])
        groq_resp = _fake_response("groq")

        def _call_next(req):
            raise RateLimitError("rl", provider="openai")

        req = MagicMock()
        req.model = "gpt-4o-mini"
        # model_copy returns a MagicMock that looks like a fallback request
        fallback_req = MagicMock()
        fallback_req.model_copy = MagicMock(
            return_value=groq_resp.model_copy(
                update={"fallback_attempts": ["gpt-4o-mini"]}
            )
        )
        req.model_copy = MagicMock(return_value=fallback_req)

        groq_provider = MagicMock()
        groq_provider.complete.return_value = groq_resp

        with patch(_PATCH_MW_GET_PROVIDER, return_value=groq_provider):
            resp = mw.handle(req, _call_next)

        assert resp.fallback_attempts == ["gpt-4o-mini"]
        groq_provider.complete.assert_called_once()

    def test_all_fallbacks_fail_raises(self):
        mw = FallbackMiddleware(models=["groq/llama-3.1-8b-instant"])

        def _call_next(req):
            raise RateLimitError("rl", provider="openai")

        req = MagicMock()
        req.model = "gpt-4o-mini"
        req.model_copy = MagicMock(return_value=MagicMock())

        groq_provider = MagicMock()
        groq_provider.complete.side_effect = RateLimitError("rl", provider="groq")

        with patch(_PATCH_MW_GET_PROVIDER, return_value=groq_provider):
            with pytest.raises(AllProvidersFailedError) as exc_info:
                mw.handle(req, _call_next)
        assert len(exc_info.value.errors) == 2  # primary + 1 fallback

    def test_empty_models_raises(self):
        with pytest.raises(ValueError, match="at least one fallback model"):
            FallbackMiddleware(models=[])

    @pytest.mark.asyncio
    async def test_async_primary_succeeds(self):
        mw = FallbackMiddleware(models=["groq/llama-3.1-8b-instant"])
        fake = _fake_response("openai")

        async def _call_next(req):
            return fake

        req = MagicMock()
        req.model = "gpt-4o-mini"
        resp = await mw.ahandle(req, _call_next)
        assert resp == fake

    @pytest.mark.asyncio
    async def test_async_fallback_on_rate_limit(self):
        mw = FallbackMiddleware(models=["groq/llama-3.1-8b-instant"])
        groq_resp = _fake_response("groq")

        async def _call_next(req):
            raise RateLimitError("rl", provider="openai")

        req = MagicMock()
        req.model = "gpt-4o-mini"
        fallback_req = MagicMock()
        fallback_req.model_copy = MagicMock(
            return_value=groq_resp.model_copy(
                update={"fallback_attempts": ["gpt-4o-mini"]}
            )
        )
        req.model_copy = MagicMock(return_value=fallback_req)

        groq_provider = MagicMock()
        groq_provider.acomplete = AsyncMock(return_value=groq_resp)

        with patch(_PATCH_MW_GET_PROVIDER, return_value=groq_provider):
            resp = await mw.ahandle(req, _call_next)

        assert resp.fallback_attempts == ["gpt-4o-mini"]
