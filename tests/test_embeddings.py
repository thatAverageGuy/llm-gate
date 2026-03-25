"""
Tests for the Embeddings API — all provider SDK calls are mocked.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmgate.embeddings import _route, embed
from llmgate.exceptions import EmbeddingsNotSupported
from llmgate.types import EmbeddingResponse

FAKE_VECTOR = [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class TestRoute:
    def test_openai_default(self):
        assert _route("text-embedding-3-small") == "openai"

    def test_openai_explicit(self):
        assert _route("openai/text-embedding-ada-002") == "openai"

    def test_gemini(self):
        assert _route("gemini/text-embedding-004") == "gemini"

    def test_azure(self):
        assert _route("azure/my-embedding") == "azure"

    def test_cohere(self):
        assert _route("cohere/embed-english-v3.0") == "cohere"

    def test_mistral(self):
        assert _route("mistral/mistral-embed") == "mistral"

    def test_ollama(self):
        assert _route("ollama/nomic-embed-text") == "ollama"

    def test_bedrock(self):
        assert _route("bedrock/amazon.titan-embed-text-v2:0") == "bedrock"

    def test_anthropic_raises(self):
        with pytest.raises(EmbeddingsNotSupported) as exc_info:
            embed("anthropic/claude-3", "hello")
        assert exc_info.value.provider == "anthropic"

    def test_groq_raises(self):
        with pytest.raises(EmbeddingsNotSupported) as exc_info:
            embed("groq/llama3", "hello")
        assert exc_info.value.provider == "groq"


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class TestOpenAIEmbeddings:
    def _make_raw(self):
        item = SimpleNamespace(index=0, embedding=FAKE_VECTOR)
        usage = SimpleNamespace(prompt_tokens=3, total_tokens=3)
        return SimpleNamespace(data=[item], usage=usage)

    def test_single_input(self):
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = self._make_raw()

        with patch("llmgate.embeddings.os"), \
             patch("llmgate.embeddings._embed_openai") as mock_fn:
            mock_fn.return_value = EmbeddingResponse(
                model="text-embedding-3-small",
                provider="openai",
                embeddings=[FAKE_VECTOR],
            )
            resp = embed("text-embedding-3-small", "hello")

        assert resp.provider == "openai"
        assert resp.embeddings == [FAKE_VECTOR]

    def test_batch_input_returns_multiple_vectors(self):
        two_vectors = [FAKE_VECTOR, [0.4, 0.5, 0.6]]
        with patch("llmgate.embeddings._embed_openai") as mock_fn:
            mock_fn.return_value = EmbeddingResponse(
                model="text-embedding-3-small",
                provider="openai",
                embeddings=two_vectors,
            )
            resp = embed("text-embedding-3-small", ["hello", "world"])

        assert len(resp.embeddings) == 2

    def test_dimensions_passed_through(self):
        """Verify dimensions kwarg is forwarded."""
        with patch("llmgate.embeddings._embed_openai") as mock_fn:
            mock_fn.return_value = EmbeddingResponse(
                model="text-embedding-3-small",
                provider="openai",
                embeddings=[FAKE_VECTOR],
            )
            embed("text-embedding-3-small", "hi", dimensions=256)
            call_args = mock_fn.call_args
            # dimensions goes into EmbeddingRequest
            assert call_args[0][0].dimensions == 256


# ---------------------------------------------------------------------------
# Gemini provider (mocked)
# ---------------------------------------------------------------------------

class TestGeminiEmbeddings:
    def test_embed_gemini(self):
        with patch("llmgate.embeddings._embed_gemini") as mock_fn:
            mock_fn.return_value = EmbeddingResponse(
                model="gemini/text-embedding-004",
                provider="gemini",
                embeddings=[FAKE_VECTOR],
            )
            resp = embed("gemini/text-embedding-004", "hello")

        assert resp.provider == "gemini"
        assert resp.embeddings[0] == FAKE_VECTOR


# ---------------------------------------------------------------------------
# Cohere provider (mocked)
# ---------------------------------------------------------------------------

class TestCohereEmbeddings:
    def test_embed_cohere(self):
        with patch("llmgate.embeddings._embed_cohere") as mock_fn:
            mock_fn.return_value = EmbeddingResponse(
                model="cohere/embed-english-v3.0",
                provider="cohere",
                embeddings=[FAKE_VECTOR],
            )
            resp = embed("cohere/embed-english-v3.0", "hello")

        assert resp.provider == "cohere"


# ---------------------------------------------------------------------------
# Mistral provider (mocked)
# ---------------------------------------------------------------------------

class TestMistralEmbeddings:
    def test_embed_mistral(self):
        with patch("llmgate.embeddings._embed_mistral") as mock_fn:
            mock_fn.return_value = EmbeddingResponse(
                model="mistral/mistral-embed",
                provider="mistral",
                embeddings=[FAKE_VECTOR],
            )
            resp = embed("mistral/mistral-embed", "hello")

        assert resp.provider == "mistral"


# ---------------------------------------------------------------------------
# Ollama provider (mocked)
# ---------------------------------------------------------------------------

class TestOllamaEmbeddings:
    def test_embed_ollama(self):
        with patch("llmgate.embeddings._embed_ollama") as mock_fn:
            mock_fn.return_value = EmbeddingResponse(
                model="ollama/nomic-embed-text",
                provider="ollama",
                embeddings=[FAKE_VECTOR],
            )
            resp = embed("ollama/nomic-embed-text", "hello")

        assert resp.provider == "ollama"


# ---------------------------------------------------------------------------
# Bedrock provider (mocked)
# ---------------------------------------------------------------------------

class TestBedrockEmbeddings:
    def test_embed_bedrock(self):
        with patch("llmgate.embeddings._embed_bedrock") as mock_fn:
            mock_fn.return_value = EmbeddingResponse(
                model="bedrock/amazon.titan-embed-text-v2:0",
                provider="bedrock",
                embeddings=[FAKE_VECTOR],
            )
            resp = embed("bedrock/amazon.titan-embed-text-v2:0", "hello")

        assert resp.provider == "bedrock"


# ---------------------------------------------------------------------------
# EmbeddingResponse model
# ---------------------------------------------------------------------------

class TestEmbeddingResponse:
    def test_always_list_of_lists(self):
        resp = EmbeddingResponse(
            model="text-embedding-3-small",
            provider="openai",
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
        )
        assert isinstance(resp.embeddings, list)
        assert isinstance(resp.embeddings[0], list)

    def test_raw_excluded_from_serialisation(self):
        resp = EmbeddingResponse(
            model="m",
            provider="openai",
            embeddings=[[0.1]],
            raw={"some": "object"},
        )
        serialised = resp.model_dump()
        assert "raw" not in serialised

    def test_usage_defaults_to_zero(self):
        resp = EmbeddingResponse(
            model="m",
            provider="openai",
            embeddings=[[0.1]],
        )
        assert resp.usage.total_tokens == 0


# ---------------------------------------------------------------------------
# LLMGate.embed()
# ---------------------------------------------------------------------------

class TestLLMGateEmbed:
    def test_gate_embed_delegates(self):
        from llmgate import LLMGate
        gate = LLMGate()
        expected = EmbeddingResponse(
            model="text-embedding-3-small",
            provider="openai",
            embeddings=[FAKE_VECTOR],
        )
        with patch("llmgate.gate._embed_fn", return_value=expected):
            resp = gate.embed("text-embedding-3-small", "hello")
        assert resp.provider == "openai"

    @pytest.mark.asyncio
    async def test_gate_aembed_delegates(self):
        from llmgate import LLMGate
        gate = LLMGate()
        expected = EmbeddingResponse(
            model="text-embedding-3-small",
            provider="openai",
            embeddings=[FAKE_VECTOR],
        )
        with patch("llmgate.gate._aembed_fn", new=AsyncMock(return_value=expected)):
            resp = await gate.aembed("text-embedding-3-small", "hello")
        assert resp.embeddings == [FAKE_VECTOR]
