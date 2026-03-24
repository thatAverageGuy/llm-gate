"""Tests for the top-level completion() and acompletion() functions."""
from __future__ import annotations

import sys
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import llmgate.completion  # ensure module is loaded into sys.modules
from llmgate import acompletion, completion
from llmgate.exceptions import ModelNotFoundError
from llmgate.types import Choice, CompletionResponse, Message, StreamChunk, TokenUsage


def _clear_cache() -> None:
    """Clear the provider instance cache between tests."""
    sys.modules["llmgate.completion"]._provider_cache.clear()


# ---------------------------------------------------------------------------
# Helpers — build a fake CompletionResponse
# ---------------------------------------------------------------------------


def _fake_response(provider: str = "openai", model: str = "gpt-4o-mini") -> CompletionResponse:
    return CompletionResponse(
        id="fake-id",
        model=model,
        provider=provider,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content="test response"),
                finish_reason="stop",
            )
        ],
        usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------


class TestCompletionRouting:
    def test_routes_to_openai(self):
        with patch("llmgate.providers.openai.OpenAIProvider.complete") as mock_complete, \
             patch("llmgate.providers.openai.OpenAIProvider.__init__", return_value=None):
            mock_complete.return_value = _fake_response("openai")
            _clear_cache()
            resp = completion("gpt-4o-mini", [{"role": "user", "content": "hi"}],
                              api_key="test-key")
            assert resp.provider == "openai"

    def test_routes_to_anthropic(self):
        with patch("llmgate.providers.anthropic.AnthropicProvider.complete") as mock_complete, \
             patch("llmgate.providers.anthropic.AnthropicProvider.__init__", return_value=None):
            mock_complete.return_value = _fake_response("anthropic", "claude-3-5-sonnet-20241022")
            _clear_cache()
            resp = completion("claude-3-5-sonnet-20241022",
                              [{"role": "user", "content": "hi"}], api_key="test-key")
            assert resp.provider == "anthropic"

    def test_routes_to_gemini(self):
        with patch("llmgate.providers.gemini.GeminiProvider.complete") as mock_complete, \
             patch("llmgate.providers.gemini.GeminiProvider.__init__", return_value=None):
            mock_complete.return_value = _fake_response("gemini", "gemini-1.5-flash")
            _clear_cache()
            resp = completion("gemini-1.5-flash",
                              [{"role": "user", "content": "hi"}], api_key="test-key")
            assert resp.provider == "gemini"

    def test_routes_to_groq_with_prefix(self):
        with patch("llmgate.providers.groq.GroqProvider.complete") as mock_complete, \
             patch("llmgate.providers.groq.GroqProvider.__init__", return_value=None):
            mock_complete.return_value = _fake_response("groq", "groq/llama-3.1-8b-instant")
            _clear_cache()
            resp = completion("groq/llama-3.1-8b-instant",
                              [{"role": "user", "content": "hi"}], api_key="test-key")
            assert resp.provider == "groq"

    def test_unknown_model_raises(self):
        _clear_cache()
        with pytest.raises(ModelNotFoundError):
            completion("unknown-model-xyz", [{"role": "user", "content": "hi"}])

    def test_stream_raises_not_supported(self):
        """stream=True with no error should return an Iterator."""
        def _fake_stream():
            yield StreamChunk(id="c", model="gpt-4o", provider="openai", delta="hi")

        with patch("llmgate.providers.openai.OpenAIProvider.stream",
                   return_value=_fake_stream()) as mock_stream, \
             patch("llmgate.providers.openai.OpenAIProvider.__init__", return_value=None):
            _clear_cache()
            chunks = list(completion("gpt-4o", [{"role": "user", "content": "hi"}], stream=True))
            assert len(chunks) == 1
            assert isinstance(chunks[0], StreamChunk)
            assert chunks[0].delta == "hi"

    def test_explicit_provider_override(self):
        with patch("llmgate.providers.groq.GroqProvider.complete") as mock_complete, \
             patch("llmgate.providers.groq.GroqProvider.__init__", return_value=None):
            mock_complete.return_value = _fake_response("groq", "llama-3.1-8b-instant")
            _clear_cache()
            resp = completion("llama-3.1-8b-instant",
                              [{"role": "user", "content": "hi"}],
                              provider="groq", api_key="test-key")
            assert resp.provider == "groq"


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


class TestACompletion:
    @pytest.mark.asyncio
    async def test_async_routes_to_openai(self):
        with patch("llmgate.providers.openai.OpenAIProvider.acomplete", new_callable=AsyncMock) as mock_ac, \
             patch("llmgate.providers.openai.OpenAIProvider.__init__", return_value=None):
            mock_ac.return_value = _fake_response("openai")
            _clear_cache()
            resp = await acompletion("gpt-4o-mini",
                                     [{"role": "user", "content": "hi"}], api_key="key")
            assert resp.provider == "openai"

    @pytest.mark.asyncio
    async def test_async_stream_returns_iterator(self):
        """acompletion(stream=True) should return an AsyncIterator of StreamChunk."""
        async def _fake_astream():
            yield StreamChunk(id="c", model="gpt-4o", provider="openai", delta="hello")

        with patch("llmgate.providers.openai.OpenAIProvider.astream",
                   return_value=_fake_astream()) as mock_astream, \
             patch("llmgate.providers.openai.OpenAIProvider.__init__", return_value=None):
            _clear_cache()
            chunks = []
            async for chunk in await acompletion(
                "gpt-4o", [{"role": "user", "content": "hi"}], stream=True
            ):
                chunks.append(chunk)
            assert len(chunks) == 1
            assert chunks[0].delta == "hello"


# ---------------------------------------------------------------------------
# Message normalisation
# ---------------------------------------------------------------------------


class TestMessageNormalisation:
    def test_dict_messages_accepted(self):
        with patch("llmgate.providers.openai.OpenAIProvider.complete") as mock_complete, \
             patch("llmgate.providers.openai.OpenAIProvider.__init__", return_value=None):
            mock_complete.return_value = _fake_response("openai")
            _clear_cache()
            resp = completion(
                "gpt-4o-mini",
                [{"role": "user", "content": "hello"}],
                api_key="test-key",
            )
            assert resp is not None

    def test_message_objects_accepted(self):
        with patch("llmgate.providers.openai.OpenAIProvider.complete") as mock_complete, \
             patch("llmgate.providers.openai.OpenAIProvider.__init__", return_value=None):
            mock_complete.return_value = _fake_response("openai")
            _clear_cache()
            resp = completion(
                "gpt-4o-mini",
                [Message(role="user", content="hello")],
                api_key="test-key",
            )
            assert resp is not None
