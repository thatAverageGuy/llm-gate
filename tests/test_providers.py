"""Tests for individual provider request/response mapping (all mocked)."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmgate.types import (
    CompletionRequest, FunctionDefinition, Message, ToolCall, ToolDefinition,
)


def _make_request(model: str, messages=None, **kwargs) -> CompletionRequest:
    if messages is None:
        messages = [Message(role="user", content="hello")]
    return CompletionRequest(model=model, messages=messages, **kwargs)


def _weather_tool() -> ToolDefinition:
    return ToolDefinition(function=FunctionDefinition(
        name="get_weather",
        description="Get current weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    ))


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    def _make_raw(self, tool_calls=None):
        msg = SimpleNamespace(role="assistant", content="Hi!", tool_calls=tool_calls)
        raw = SimpleNamespace(
            id="chatcmpl-xyz",
            choices=[
                SimpleNamespace(index=0, message=msg, finish_reason="stop")
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

    def test_tools_serialised_to_openai_format(self):
        with patch("llmgate.providers.openai.OpenAIProvider.__init__", return_value=None):
            from llmgate.providers.openai import OpenAIProvider
            provider = OpenAIProvider.__new__(OpenAIProvider)
            req = _make_request("gpt-4o-mini", tools=[_weather_tool()])
            params = provider._build_params(req)
            assert "tools" in params
            assert params["tools"][0]["type"] == "function"
            assert params["tools"][0]["function"]["name"] == "get_weather"
            assert "parameters" in params["tools"][0]["function"]

    def test_tool_call_response_mapped(self):
        with patch("llmgate.providers.openai.OpenAIProvider.__init__", return_value=None):
            from llmgate.providers.openai import OpenAIProvider
            provider = OpenAIProvider.__new__(OpenAIProvider)
            raw_tc = SimpleNamespace(
                id="call_abc",
                function=SimpleNamespace(name="get_weather", arguments='{"city": "London"}'),
            )
            raw = self._make_raw(tool_calls=[raw_tc])
            raw.choices[0].finish_reason = "tool_calls"
            raw.choices[0].message.content = None

            req = _make_request("gpt-4o-mini")
            resp = provider._map_response(raw, "gpt-4o-mini")
            assert len(resp.tool_calls) == 1
            assert resp.tool_calls[0].function == "get_weather"
            assert resp.tool_calls[0].arguments == {"city": "London"}


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    def _make_raw(self, content=None):
        if content is None:
            content = [SimpleNamespace(type="text", text="Hello from Claude")]
        return SimpleNamespace(
            id="msg-xyz",
            content=content,
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

    def test_tools_serialised_to_anthropic_format(self):
        with patch("llmgate.providers.anthropic.AnthropicProvider.__init__", return_value=None):
            from llmgate.providers.anthropic import AnthropicProvider
            provider = AnthropicProvider.__new__(AnthropicProvider)
            req = _make_request("claude-3-5-sonnet-20241022", tools=[_weather_tool()])
            params = provider._build_params(req)
            assert "tools" in params
            assert params["tools"][0]["name"] == "get_weather"
            # Anthropic uses input_schema, not parameters
            assert "input_schema" in params["tools"][0]
            assert "parameters" not in params["tools"][0]

    def test_tool_call_response_mapped(self):
        with patch("llmgate.providers.anthropic.AnthropicProvider.__init__", return_value=None):
            from llmgate.providers.anthropic import AnthropicProvider
            provider = AnthropicProvider.__new__(AnthropicProvider)
            # Anthropic returns tool_use content blocks
            tool_block = SimpleNamespace(
                type="tool_use", id="toolu_123", name="get_weather",
                input={"city": "London"},
            )
            raw = self._make_raw(content=[tool_block])
            raw.stop_reason = "tool_use"
            resp = provider._map_response(raw, "claude-3-5-sonnet-20241022")
            assert len(resp.tool_calls) == 1
            assert resp.tool_calls[0].function == "get_weather"
            assert resp.tool_calls[0].arguments == {"city": "London"}

    def test_tool_result_message_format(self):
        with patch("llmgate.providers.anthropic.AnthropicProvider.__init__", return_value=None):
            from llmgate.providers.anthropic import AnthropicProvider
            provider = AnthropicProvider.__new__(AnthropicProvider)
            msgs = [
                Message(role="user", content="What's the weather?"),
                Message(role="assistant", content=None, tool_calls=[
                    ToolCall(id="toolu_123", function="get_weather", arguments={"city": "London"})
                ]),
                Message(role="tool", tool_call_id="toolu_123", content='{"temp": "12°C"}'),
            ]
            _, built = AnthropicProvider._build_messages(msgs)
            # tool result should be a user message with tool_result block
            assert built[-1]["role"] == "user"
            assert built[-1]["content"][0]["type"] == "tool_result"
            assert built[-1]["content"][0]["tool_use_id"] == "toolu_123"


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

    def test_tools_serialised_to_openai_format(self):
        with patch("llmgate.providers.groq.GroqProvider.__init__", return_value=None):
            from llmgate.providers.groq import GroqProvider
            provider = GroqProvider.__new__(GroqProvider)
            req = _make_request("groq/llama-3.1-8b-instant", tools=[_weather_tool()])
            params = provider._build_params(req)
            assert "tools" in params
            assert params["tools"][0]["type"] == "function"
            assert params["tools"][0]["function"]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------


class TestGeminiProvider:
    def test_message_conversion(self):
        with patch("llmgate.providers.gemini.GeminiProvider.__init__", return_value=None):
            from llmgate.providers.gemini import GeminiProvider
            sys_instr, contents = GeminiProvider._to_genai_contents([
                Message(role="system", content="Be concise."),
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there"),
                Message(role="user", content="How are you?"),
            ])
            assert sys_instr == "Be concise."
            assert contents[0]["role"] == "user"
            assert contents[1]["role"] == "model"  # assistant -> model
            assert contents[2]["role"] == "user"

    def test_tools_serialised_to_gemini_format(self):
        with patch("llmgate.providers.gemini.GeminiProvider.__init__", return_value=None):
            from llmgate.providers.gemini import GeminiProvider
            provider = GeminiProvider.__new__(GeminiProvider)
            req = _make_request("gemini-2.5-flash-lite", tools=[_weather_tool()])
            config = provider._build_config(req)
            assert "tools" in config
            assert "function_declarations" in config["tools"][0]
            assert config["tools"][0]["function_declarations"][0]["name"] == "get_weather"

    def test_tool_result_message_format(self):
        with patch("llmgate.providers.gemini.GeminiProvider.__init__", return_value=None):
            from llmgate.providers.gemini import GeminiProvider
            msgs = [
                Message(role="user", content="What's the weather?"),
                Message(role="assistant", content=None, tool_calls=[
                    ToolCall(id="call-1", function="get_weather", arguments={"city": "London"})
                ]),
                Message(role="tool", name="get_weather", content='{"temp": "12°C"}'),
            ]
            _, contents = GeminiProvider._to_genai_contents(msgs)
            # Last item should be a user message with function_response part
            last = contents[-1]
            assert last["role"] == "user"
            assert "function_response" in last["parts"][0]
            assert last["parts"][0]["function_response"]["name"] == "get_weather"


