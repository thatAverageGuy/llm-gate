"""Tests for the top-level completion() and acompletion() functions."""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import llmgate.completion  # ensure module is loaded into sys.modules
from llmgate import acompletion, completion
from llmgate.exceptions import ModelNotFoundError, StreamingNotSupported
from llmgate.types import Choice, CompletionResponse, Message, TokenUsage


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
        with pytest.raises(StreamingNotSupported):
            completion("gpt-4o", [{"role": "user", "content": "hi"}], stream=True)

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
    async def test_async_stream_raises(self):
        with pytest.raises(StreamingNotSupported):
            await acompletion("gpt-4o", [{"role": "user", "content": "hi"}], stream=True)


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
