"""Tests for individual provider request/response mapping (all mocked)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmgate.types import CompletionRequest, Message


def _make_request(model: str, messages=None) -> CompletionRequest:
    if messages is None:
        messages = [Message(role="user", content="hello")]
    return CompletionRequest(model=model, messages=messages)


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    def _make_raw(self):
        raw = SimpleNamespace(
            id="chatcmpl-xyz",
            choices=[
                SimpleNamespace(
                    index=0,
                    message=SimpleNamespace(role="assistant", content="Hi!"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        )
        return raw

    def test_complete_maps_response(self):
        with patch("llmgate.providers.openai.OpenAIProvider.__init__", return_value=None):
            from llmgate.providers.openai import OpenAIProvider
            provider = OpenAIProvider.__new__(OpenAIProvider)
            provider._openai = MagicMock()
            provider._client = MagicMock()
            provider._client.chat.completions.create.return_value = self._make_raw()

            req = _make_request("gpt-4o-mini")
            resp = provider.complete(req)

            assert resp.id == "chatcmpl-xyz"
            assert resp.provider == "openai"
            assert resp.text == "Hi!"
            assert resp.usage.total_tokens == 7

    @pytest.mark.asyncio
    async def test_acomplete_maps_response(self):
        with patch("llmgate.providers.openai.OpenAIProvider.__init__", return_value=None):
            from llmgate.providers.openai import OpenAIProvider
            provider = OpenAIProvider.__new__(OpenAIProvider)
            provider._openai = MagicMock()
            provider._async_client = MagicMock()
            provider._async_client.chat.completions.create = AsyncMock(
                return_value=self._make_raw()
            )

            req = _make_request("gpt-4o-mini")
            resp = await provider.acomplete(req)
            assert resp.text == "Hi!"


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    def _make_raw(self):
        return SimpleNamespace(
            id="msg-xyz",
            content=[SimpleNamespace(text="Hello from Claude")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=4),
        )

    def test_complete_system_extraction(self):
        """System messages should be extracted from the messages list."""
        with patch("llmgate.providers.anthropic.AnthropicProvider.__init__", return_value=None):
            from llmgate.providers.anthropic import AnthropicProvider
            provider = AnthropicProvider.__new__(AnthropicProvider)
            provider._anthropic = MagicMock()
            provider._client = MagicMock()
            provider._client.messages.create.return_value = self._make_raw()

            req = _make_request(
                "claude-3-5-sonnet-20241022",
                messages=[
                    Message(role="system", content="You are helpful."),
                    Message(role="user", content="hi"),
                ],
            )
            params = provider._build_params(req)
            assert "system" in params
            assert params["system"] == "You are helpful."
            # system msg should not appear in messages list
            assert not any(m.get("role") == "system" for m in params["messages"])

    def test_max_tokens_defaults_to_1024(self):
        with patch("llmgate.providers.anthropic.AnthropicProvider.__init__", return_value=None):
            from llmgate.providers.anthropic import AnthropicProvider
            provider = AnthropicProvider.__new__(AnthropicProvider)
            req = _make_request("claude-3-5-sonnet-20241022")
            params = provider._build_params(req)
            assert params["max_tokens"] == 1024

    def test_complete_maps_response(self):
        with patch("llmgate.providers.anthropic.AnthropicProvider.__init__", return_value=None):
            from llmgate.providers.anthropic import AnthropicProvider
            provider = AnthropicProvider.__new__(AnthropicProvider)
            provider._anthropic = MagicMock()
            provider._client = MagicMock()
            provider._client.messages.create.return_value = self._make_raw()

            req = _make_request("claude-3-5-sonnet-20241022")
            resp = provider.complete(req)
            assert resp.text == "Hello from Claude"
            assert resp.usage.prompt_tokens == 10


# ---------------------------------------------------------------------------
# Groq provider
# ---------------------------------------------------------------------------


class TestGroqProvider:
    def test_prefix_stripped(self):
        with patch("llmgate.providers.groq.GroqProvider.__init__", return_value=None):
            from llmgate.providers.groq import GroqProvider
            provider = GroqProvider.__new__(GroqProvider)
            req = _make_request("groq/llama-3.1-8b-instant")
            params = provider._build_params(req)
            assert params["model"] == "llama-3.1-8b-instant"

    def test_no_prefix_passes_through(self):
        with patch("llmgate.providers.groq.GroqProvider.__init__", return_value=None):
            from llmgate.providers.groq import GroqProvider
            provider = GroqProvider.__new__(GroqProvider)
            req = _make_request("groq/gemma2-9b-it")
            params = provider._build_params(req)
            assert params["model"] == "gemma2-9b-it"


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------


class TestGeminiProvider:
    def test_message_conversion(self):
        with patch("llmgate.providers.gemini.GeminiProvider.__init__", return_value=None):
            from llmgate.providers.gemini import GeminiProvider
            sys_instr, contents = GeminiProvider._to_gemini_contents([
                Message(role="system", content="Be concise."),
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there"),
                Message(role="user", content="How are you?"),
            ])
            assert sys_instr == "Be concise."
            assert contents[0]["role"] == "user"
            assert contents[1]["role"] == "model"  # assistant -> model
            assert contents[2]["role"] == "user"
