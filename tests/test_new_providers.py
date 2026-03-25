"""
Mocked unit tests for the 5 new optional providers.
No API calls are made — all SDK interactions are patched.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llmgate.exceptions import AuthError
from llmgate.types import (
    CompletionRequest, FunctionDefinition,
    Message, ToolDefinition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _req(model: str, content: str = "hello") -> CompletionRequest:
    return CompletionRequest(
        model=model,
        messages=[Message(role="user", content=content)],
    )


def _req_with_tools(model: str) -> CompletionRequest:
    return CompletionRequest(
        model=model,
        messages=[Message(role="user", content="What's the weather?")],
        tools=[ToolDefinition(function=FunctionDefinition(
            name="get_weather",
            description="Get weather for a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        ))],
    )


# --------------------------------------------------------------------------
# MistralProvider
# --------------------------------------------------------------------------

class TestMistralProvider:
    def _make_provider(self):
        from llmgate.providers.mistral import MistralProvider
        mock_client = MagicMock()
        p = MistralProvider.__new__(MistralProvider)
        p._client = mock_client
        return p, mock_client

    def test_model_prefix_stripped(self):
        from llmgate.providers.mistral import MistralProvider
        p = MistralProvider.__new__(MistralProvider)
        assert p._strip_prefix("mistral/mistral-large-latest") == "mistral-large-latest"

    def test_supports_prefix(self):
        from llmgate.providers.mistral import MistralProvider
        assert MistralProvider.supports("mistral/mistral-large-latest")
        assert not MistralProvider.supports("gpt-4o")

    def test_complete_maps_response(self):
        p, mock_client = self._make_provider()
        mock_raw = SimpleNamespace(
            id="mid-1",
            choices=[SimpleNamespace(
                index=0,
                message=SimpleNamespace(role="assistant", content="pong", tool_calls=None),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        mock_client.chat.complete.return_value = mock_raw
        resp = p.complete(_req("mistral/mistral-large-latest", "ping"))
        assert resp.text == "pong"
        assert resp.provider == "mistral"

    def test_tools_serialised_correctly(self):
        p, _ = self._make_provider()
        req = _req_with_tools("mistral/mistral-large-latest")
        params = p._build_params(req)
        assert "tools" in params
        assert params["tools"][0]["type"] == "function"
        assert params["tools"][0]["function"]["name"] == "get_weather"

    def test_auth_error_when_no_key(self, monkeypatch):
        pytest.importorskip("mistralai", reason="mistralai not installed")
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        from llmgate.providers.mistral import MistralProvider
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthError):
                MistralProvider(api_key=None)


# --------------------------------------------------------------------------
# CohereProvider
# --------------------------------------------------------------------------

class TestCohereProvider:
    def _make_provider(self):
        from llmgate.providers.cohere import CohereProvider
        p = CohereProvider.__new__(CohereProvider)
        mock_client = MagicMock()
        p._client = mock_client
        p._async_client = MagicMock()
        p._cohere = MagicMock()
        return p, mock_client

    def test_model_prefix_stripped(self):
        from llmgate.providers.cohere import CohereProvider
        p = CohereProvider.__new__(CohereProvider)
        assert p._strip_prefix("cohere/command-r-plus") == "command-r-plus"

    def test_supports_prefix(self):
        from llmgate.providers.cohere import CohereProvider
        assert CohereProvider.supports("cohere/command-r-plus")
        assert not CohereProvider.supports("gpt-4o")

    def test_messages_built_correctly(self):
        from llmgate.providers.cohere import CohereProvider
        p = CohereProvider.__new__(CohereProvider)
        req = _req("cohere/command-r-plus", "hello")
        params = p._build_params(req)
        assert params["model"] == "command-r-plus"
        assert any(m.get("role") == "user" for m in params["messages"])

    def test_tools_serialised_correctly(self):
        from llmgate.providers.cohere import CohereProvider
        p = CohereProvider.__new__(CohereProvider)
        req = _req_with_tools("cohere/command-r-plus")
        params = p._build_params(req)
        assert "tools" in params
        assert params["tools"][0]["function"]["name"] == "get_weather"

    def test_auth_error_when_no_key(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        from llmgate.providers.cohere import CohereProvider
        # cohere is a lazy import inside __init__; mock importlib
        mock_cohere = MagicMock()
        mock_cohere.ClientV2 = MagicMock()
        mock_cohere.AsyncClientV2 = MagicMock()
        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(AuthError):
                    CohereProvider(api_key=None)


# --------------------------------------------------------------------------
# AzureOpenAIProvider
# --------------------------------------------------------------------------

class TestAzureOpenAIProvider:
    def _make_provider(self):
        from llmgate.providers.azure import AzureOpenAIProvider
        p = AzureOpenAIProvider.__new__(AzureOpenAIProvider)
        p._client = MagicMock()
        p._async_client = MagicMock()
        p._openai = MagicMock()
        return p

    def test_model_prefix_stripped(self):
        from llmgate.providers.azure import AzureOpenAIProvider
        p = AzureOpenAIProvider.__new__(AzureOpenAIProvider)
        assert p._strip_prefix("azure/my-gpt4-deployment") == "my-gpt4-deployment"

    def test_supports_prefix(self):
        from llmgate.providers.azure import AzureOpenAIProvider
        assert AzureOpenAIProvider.supports("azure/my-deployment")
        assert not AzureOpenAIProvider.supports("gpt-4o")

    def test_params_built_correctly(self):
        p = self._make_provider()
        req = _req("azure/my-deployment", "hi")
        params = p._build_params(req)
        assert params["model"] == "my-deployment"
        assert params["messages"][0]["role"] == "user"

    def test_complete_maps_response(self):
        p = self._make_provider()
        mock_raw = SimpleNamespace(
            id="az-1",
            choices=[SimpleNamespace(
                index=0,
                message=SimpleNamespace(role="assistant", content="azure pong", tool_calls=None),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=3, total_tokens=7),
        )
        p._client.chat.completions.create.return_value = mock_raw
        resp = p.complete(_req("azure/gpt-4-deployment"))
        assert resp.text == "azure pong"
        assert resp.provider == "azure"

    def test_auth_error_when_no_key(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        from llmgate.providers.azure import AzureOpenAIProvider
        # openai is already installed; just unset env vars
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthError):
                AzureOpenAIProvider(api_key=None, azure_endpoint=None)


# --------------------------------------------------------------------------
# BedrockProvider
# --------------------------------------------------------------------------

class TestBedrockProvider:
    def _make_provider(self):
        from llmgate.providers.bedrock import BedrockProvider
        p = BedrockProvider.__new__(BedrockProvider)
        p._client = MagicMock()
        p._boto3 = MagicMock()
        p._region = "us-east-1"
        return p

    def test_model_prefix_stripped(self):
        from llmgate.providers.bedrock import BedrockProvider
        p = BedrockProvider.__new__(BedrockProvider)
        full = "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert p._strip_prefix(full) == "anthropic.claude-3-5-sonnet-20241022-v2:0"

    def test_supports_prefix(self):
        from llmgate.providers.bedrock import BedrockProvider
        assert BedrockProvider.supports("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
        assert not BedrockProvider.supports("gpt-4o")

    def test_converse_params_built_correctly(self):
        p = self._make_provider()
        req = _req("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0", "hello")
        params = p._build_converse_params(req)
        assert params["modelId"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert params["messages"][0]["role"] == "user"

    def test_system_prompt_extracted(self):
        from llmgate.providers.bedrock import BedrockProvider
        p = BedrockProvider.__new__(BedrockProvider)
        msgs = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="hello"),
        ]
        req = CompletionRequest(model="bedrock/claude", messages=msgs)
        params = p._build_converse_params(req)
        assert "system" in params
        assert params["system"][0]["text"] == "You are helpful."

    def test_tool_config_built(self):
        p = self._make_provider()
        req = _req_with_tools("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
        params = p._build_converse_params(req)
        assert "toolConfig" in params
        assert params["toolConfig"]["tools"][0]["toolSpec"]["name"] == "get_weather"

    def test_response_mapping(self):
        p = self._make_provider()
        mock_raw = {
            "output": {"message": {"content": [{"text": "bedrock pong"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
            "ResponseMetadata": {"RequestId": "br-1"},
        }
        p._client.converse.return_value = mock_raw
        resp = p.complete(_req("bedrock/claude"))
        assert resp.text == "bedrock pong"
        assert resp.provider == "bedrock"

    def test_tool_use_block_mapped(self):
        p = self._make_provider()
        mock_raw = {
            "output": {"message": {"content": [
                {"toolUse": {"toolUseId": "tu-1", "name": "get_weather", "input": {"city": "London"}}},
            ]}},
            "stopReason": "tool_use",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
            "ResponseMetadata": {"RequestId": "br-2"},
        }
        p._client.converse.return_value = mock_raw
        resp = p.complete(_req("bedrock/claude"))
        assert resp.tool_calls is not None
        assert resp.tool_calls[0].function == "get_weather"
        assert resp.tool_calls[0].arguments == {"city": "London"}


# --------------------------------------------------------------------------
# OllamaProvider
# --------------------------------------------------------------------------

class TestOllamaProvider:
    def _make_provider(self):
        from llmgate.providers.ollama import OllamaProvider
        p = OllamaProvider.__new__(OllamaProvider)
        p._client = MagicMock()
        p._async_client = MagicMock()
        p._ollama = MagicMock()
        return p

    def test_model_prefix_stripped(self):
        from llmgate.providers.ollama import OllamaProvider
        p = OllamaProvider.__new__(OllamaProvider)
        assert p._strip_prefix("ollama/llama3.2") == "llama3.2"

    def test_supports_prefix(self):
        from llmgate.providers.ollama import OllamaProvider
        assert OllamaProvider.supports("ollama/llama3.2")
        assert not OllamaProvider.supports("gpt-4o")

    def test_params_built_correctly(self):
        p = self._make_provider()
        req = _req("ollama/llama3.2", "hello")
        params = p._build_params(req)
        assert params["model"] == "llama3.2"
        assert params["messages"][0]["role"] == "user"

    def test_temperature_in_options(self):
        p = self._make_provider()
        req = CompletionRequest(
            model="ollama/llama3.2",
            messages=[Message(role="user", content="hi")],
            temperature=0.7,
            max_tokens=100,
        )
        params = p._build_params(req)
        assert "options" in params
        assert params["options"]["temperature"] == 0.7
        assert params["options"]["num_predict"] == 100

    def test_tools_serialised_correctly(self):
        p = self._make_provider()
        req = _req_with_tools("ollama/llama3.2")
        params = p._build_params(req)
        assert "tools" in params
        assert params["tools"][0]["type"] == "function"
        assert params["tools"][0]["function"]["name"] == "get_weather"

    def test_complete_maps_response(self):
        p = self._make_provider()
        mock_raw = SimpleNamespace(
            message=SimpleNamespace(role="assistant", content="ollama pong", tool_calls=None),
            done_reason="stop",
            prompt_eval_count=5,
            eval_count=3,
        )
        p._client.chat.return_value = mock_raw
        resp = p.complete(_req("ollama/llama3.2", "ping"))
        assert resp.text == "ollama pong"
        assert resp.provider == "ollama"

    def test_tool_call_mapped(self):
        p = self._make_provider()
        mock_tc = SimpleNamespace(
            function=SimpleNamespace(name="get_weather", arguments={"city": "NYC"}),
        )
        mock_raw = SimpleNamespace(
            message=SimpleNamespace(role="assistant", content=None, tool_calls=[mock_tc]),
            done_reason="tool_calls",
            prompt_eval_count=5,
            eval_count=3,
        )
        p._client.chat.return_value = mock_raw
        resp = p.complete(_req("ollama/llama3.2"))
        assert resp.tool_calls is not None
        assert resp.tool_calls[0].function == "get_weather"


# --------------------------------------------------------------------------
# Provider routing via completion() 
# --------------------------------------------------------------------------

class TestProviderRouting:
    def test_mistral_prefix_routed(self):
        from llmgate.completion import _OPTIONAL_PROVIDERS
        prefixes = [p for p, _, _ in _OPTIONAL_PROVIDERS]
        assert "mistral/" in prefixes

    def test_ollama_prefix_routed(self):
        from llmgate.completion import _OPTIONAL_PROVIDERS
        prefixes = [p for p, _, _ in _OPTIONAL_PROVIDERS]
        assert "ollama/" in prefixes

    def test_all_optional_prefixes_registered(self):
        from llmgate.completion import _OPTIONAL_PROVIDERS
        expected = {"mistral/", "cohere/", "azure/", "bedrock/", "ollama/"}
        actual = {p for p, _, _ in _OPTIONAL_PROVIDERS}
        assert actual == expected

    def test_unknown_prefix_raises(self):
        from llmgate.exceptions import ModelNotFoundError
        from llmgate.completion import _get_provider
        with pytest.raises(ModelNotFoundError):
            _get_provider("unknownprovider/some-model")
