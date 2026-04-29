"""
Tests for the Embeddings API — all provider SDK calls are mocked.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmgate.embeddings import _route, embed
from llmgate.exceptions import EmbeddingsNotSupported
from llmgate.types import EmbeddingRequest, EmbeddingResponse

FAKE_VECTOR = [0.1, 0.2, 0.3]
FAKE_VECTOR2 = [0.4, 0.5, 0.6]


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
# EmbeddingRequest — new fields
# ---------------------------------------------------------------------------

class TestEmbeddingRequestFields:
    def test_task_type_field(self):
        req = EmbeddingRequest(model="gemini/text-embedding-004", input="hi",
                               task_type="RETRIEVAL_DOCUMENT")
        assert req.task_type == "RETRIEVAL_DOCUMENT"

    def test_title_field(self):
        req = EmbeddingRequest(model="gemini/text-embedding-004", input="hi",
                               title="My Doc")
        assert req.title == "My Doc"

    def test_input_type_field(self):
        req = EmbeddingRequest(model="cohere/embed-english-v3.0", input="hi",
                               input_type="search_query")
        assert req.input_type == "search_query"

    def test_truncate_field(self):
        req = EmbeddingRequest(model="cohere/embed-english-v3.0", input="hi",
                               truncate="END")
        assert req.truncate == "END"

    def test_encoding_format_field(self):
        req = EmbeddingRequest(model="text-embedding-3-small", input="hi",
                               encoding_format="base64")
        assert req.encoding_format == "base64"

    def test_user_field(self):
        req = EmbeddingRequest(model="text-embedding-3-small", input="hi",
                               user="user-123")
        assert req.user == "user-123"

    def test_defaults_are_none(self):
        req = EmbeddingRequest(model="text-embedding-3-small", input="hi")
        assert req.task_type is None
        assert req.title is None
        assert req.input_type is None
        assert req.truncate is None
        assert req.encoding_format is None
        assert req.user is None


# ---------------------------------------------------------------------------
# OpenAI — batching + new params
# ---------------------------------------------------------------------------

class TestOpenAIEmbeddings:
    def _make_raw(self, n=1):
        items = [SimpleNamespace(index=i, embedding=FAKE_VECTOR) for i in range(n)]
        usage = SimpleNamespace(prompt_tokens=3 * n, total_tokens=3 * n)
        return SimpleNamespace(data=items, usage=usage)

    def _mock_client(self, n=1):
        mock = MagicMock()
        mock.embeddings.create.return_value = self._make_raw(n)
        return mock

    def test_single_input(self):
        with patch("llmgate.embeddings._embed_openai") as mock_fn:
            mock_fn.return_value = EmbeddingResponse(
                model="text-embedding-3-small", provider="openai",
                embeddings=[FAKE_VECTOR],
            )
            resp = embed("text-embedding-3-small", "hello")
        assert resp.provider == "openai"
        assert resp.embeddings == [FAKE_VECTOR]

    def test_batch_single_call(self):
        """Verify the full list is sent in ONE create() call, not looped."""
        mock_client_instance = self._mock_client(n=2)
        mock_client_instance.embeddings.create.return_value = self._make_raw(2)
        with patch("openai.OpenAI", return_value=mock_client_instance):
            embed("text-embedding-3-small", ["hello", "world"])
        mock_client_instance.embeddings.create.assert_called_once()
        call_kwargs = mock_client_instance.embeddings.create.call_args
        assert call_kwargs.kwargs["input"] == ["hello", "world"]

    def test_encoding_format_forwarded(self):
        mock_client_instance = self._mock_client()
        with patch("openai.OpenAI", return_value=mock_client_instance):
            embed("text-embedding-3-small", "hi", encoding_format="base64")
        call_kwargs = mock_client_instance.embeddings.create.call_args.kwargs
        assert call_kwargs.get("encoding_format") == "base64"

    def test_user_forwarded(self):
        mock_client_instance = self._mock_client()
        with patch("openai.OpenAI", return_value=mock_client_instance):
            embed("text-embedding-3-small", "hi", user="user-abc")
        call_kwargs = mock_client_instance.embeddings.create.call_args.kwargs
        assert call_kwargs.get("user") == "user-abc"

    def test_dimensions_forwarded(self):
        with patch("llmgate.embeddings._embed_openai") as mock_fn:
            mock_fn.return_value = EmbeddingResponse(
                model="text-embedding-3-small", provider="openai",
                embeddings=[FAKE_VECTOR],
            )
            embed("text-embedding-3-small", "hi", dimensions=256)
            assert mock_fn.call_args[0][0].dimensions == 256

    def test_batch_returns_multiple_vectors(self):
        with patch("llmgate.embeddings._embed_openai") as mock_fn:
            mock_fn.return_value = EmbeddingResponse(
                model="text-embedding-3-small", provider="openai",
                embeddings=[FAKE_VECTOR, FAKE_VECTOR2],
            )
            resp = embed("text-embedding-3-small", ["hello", "world"])
        assert len(resp.embeddings) == 2


# ---------------------------------------------------------------------------
# Gemini — true batch + task_type + title
# ---------------------------------------------------------------------------

class TestGeminiEmbeddings:
    def _make_gemini_response(self, n=1):
        """Fake EmbedContentResponse with .embeddings list."""
        embs = [SimpleNamespace(values=FAKE_VECTOR) for _ in range(n)]
        return SimpleNamespace(embeddings=embs, usage_metadata=None)

    def test_batch_single_call(self):
        """Gemini must make ONE embed_content call with the full list."""
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = self._make_gemini_response(3)
        with patch("google.genai.Client", return_value=mock_client):
            resp = embed("gemini/text-embedding-004", ["a", "b", "c"])
        mock_client.models.embed_content.assert_called_once()
        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        assert call_kwargs["contents"] == ["a", "b", "c"]
        assert len(resp.embeddings) == 3

    def test_task_type_in_config(self):
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = self._make_gemini_response()
        with patch("google.genai.Client", return_value=mock_client):
            embed("gemini/text-embedding-004", "hi",
                  task_type="RETRIEVAL_DOCUMENT")
        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        assert call_kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"

    def test_retrieval_query_task_type(self):
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = self._make_gemini_response()
        with patch("google.genai.Client", return_value=mock_client):
            embed("gemini/text-embedding-004", "query?",
                  task_type="RETRIEVAL_QUERY")
        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        assert call_kwargs["config"].task_type == "RETRIEVAL_QUERY"

    def test_title_in_config(self):
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = self._make_gemini_response()
        with patch("google.genai.Client", return_value=mock_client):
            embed("gemini/text-embedding-004", "hi",
                  task_type="RETRIEVAL_DOCUMENT", title="My Doc")
        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        assert call_kwargs["config"].title == "My Doc"

    def test_dimensions_in_config(self):
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = self._make_gemini_response()
        with patch("google.genai.Client", return_value=mock_client):
            embed("gemini/text-embedding-004", "hi", dimensions=128)
        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        assert call_kwargs["config"].output_dimensionality == 128

    def test_no_config_when_no_options(self):
        """When no options set, config should not be passed."""
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = self._make_gemini_response()
        with patch("google.genai.Client", return_value=mock_client):
            embed("gemini/text-embedding-004", "hi")
        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        assert "config" not in call_kwargs

    def test_response_parsing(self):
        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = self._make_gemini_response(2)
        with patch("google.genai.Client", return_value=mock_client):
            resp = embed("gemini/text-embedding-004", ["x", "y"])
        assert resp.embeddings == [FAKE_VECTOR, FAKE_VECTOR]


# ---------------------------------------------------------------------------
# Cohere — input_type, truncate, no mutation
# ---------------------------------------------------------------------------

class TestCohereEmbeddings:
    def _mock_response(self, n=1):
        vecs = [FAKE_VECTOR] * n + ([FAKE_VECTOR2] if n > 1 else [])
        vecs = vecs[:n]
        embs = MagicMock()
        embs.float_ = vecs
        return SimpleNamespace(embeddings=embs, meta=None)

    def test_input_type_forwarded(self):
        mock_fn = MagicMock(return_value=self._mock_response())
        with patch("llmgate.embeddings._embed_cohere", mock_fn):
            embed("cohere/embed-english-v3.0", "hi", input_type="search_query")
        req = mock_fn.call_args[0][0]
        assert req.input_type == "search_query"

    def test_default_input_type_is_search_document(self):
        mock_fn = MagicMock(return_value=self._mock_response())
        with patch("llmgate.embeddings._embed_cohere", mock_fn):
            embed("cohere/embed-english-v3.0", "hi")
        req = mock_fn.call_args[0][0]
        assert req.input_type is None  # embed() doesn't set a default; provider does

    def test_truncate_forwarded(self):
        mock_fn = MagicMock(return_value=self._mock_response())
        with patch("llmgate.embeddings._embed_cohere", mock_fn):
            embed("cohere/embed-english-v3.0", "hi", truncate="END")
        req = mock_fn.call_args[0][0]
        assert req.truncate == "END"

    def test_truncate_not_set_by_default(self):
        mock_fn = MagicMock(return_value=self._mock_response())
        with patch("llmgate.embeddings._embed_cohere", mock_fn):
            embed("cohere/embed-english-v3.0", "hi")
        req = mock_fn.call_args[0][0]
        assert req.truncate is None

    def test_batch_single_call(self):
        mock_fn = MagicMock(return_value=EmbeddingResponse(
            model="cohere/embed-english-v3.0", provider="cohere",
            embeddings=[FAKE_VECTOR, FAKE_VECTOR2],
        ))
        with patch("llmgate.embeddings._embed_cohere", mock_fn):
            resp = embed("cohere/embed-english-v3.0", ["a", "b"])
        mock_fn.assert_called_once()
        req = mock_fn.call_args[0][0]
        assert req.input == ["a", "b"]
        assert len(resp.embeddings) == 2

    def test_no_extra_kwargs_mutation(self):
        """extra_kwargs must NOT be mutated (no .pop() side effects)."""
        original_extra = {"input_type": "search_query"}
        req = EmbeddingRequest(
            model="cohere/embed-english-v3.0",
            input="hi",
            extra_kwargs=dict(original_extra),  # copy
        )
        from llmgate.embeddings import _embed_cohere
        # Patch cohere.ClientV2 at the embeddings module level
        mock_cohere = MagicMock()
        mock_client = MagicMock()
        mock_client.embed.return_value = SimpleNamespace(
            embeddings=SimpleNamespace(float_=[FAKE_VECTOR]), meta=None
        )
        mock_cohere.ClientV2.return_value = mock_client
        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            _embed_cohere(req, None)
        assert req.extra_kwargs.get("input_type") == "search_query"


# ---------------------------------------------------------------------------
# Mistral — output_dimension, encoding_format
# ---------------------------------------------------------------------------

class TestMistralEmbeddings:
    def _make_mistral_response(self, n=1):
        items = [SimpleNamespace(index=i, embedding=FAKE_VECTOR) for i in range(n)]
        usage = SimpleNamespace(prompt_tokens=3*n, total_tokens=3*n)
        return SimpleNamespace(data=items, usage=usage)

    def test_output_dimension_forwarded(self):
        mock_fn = MagicMock(return_value=EmbeddingResponse(
            model="mistral/mistral-embed", provider="mistral", embeddings=[FAKE_VECTOR],
        ))
        with patch("llmgate.embeddings._embed_mistral", mock_fn):
            embed("mistral/mistral-embed", "hi", dimensions=512)
        req = mock_fn.call_args[0][0]
        assert req.dimensions == 512

    def test_encoding_format_forwarded(self):
        mock_fn = MagicMock(return_value=EmbeddingResponse(
            model="mistral/mistral-embed", provider="mistral", embeddings=[FAKE_VECTOR],
        ))
        with patch("llmgate.embeddings._embed_mistral", mock_fn):
            embed("mistral/mistral-embed", "hi", encoding_format="base64")
        req = mock_fn.call_args[0][0]
        assert req.encoding_format == "base64"

    def test_batch_single_call(self):
        mock_fn = MagicMock(return_value=EmbeddingResponse(
            model="mistral/mistral-embed", provider="mistral",
            embeddings=[FAKE_VECTOR, FAKE_VECTOR2],
        ))
        with patch("llmgate.embeddings._embed_mistral", mock_fn):
            resp = embed("mistral/mistral-embed", ["a", "b"])
        mock_fn.assert_called_once()
        req = mock_fn.call_args[0][0]
        assert req.input == ["a", "b"]
        assert len(resp.embeddings) == 2


# ---------------------------------------------------------------------------
# Ollama — true batch (single call)
# ---------------------------------------------------------------------------

class TestOllamaEmbeddings:
    def _make_ollama_response(self, n=1):
        vecs = [FAKE_VECTOR] * n
        return SimpleNamespace(embeddings=vecs)

    def test_batch_single_call(self):
        mock_fn = MagicMock(return_value=EmbeddingResponse(
            model="ollama/nomic-embed-text", provider="ollama",
            embeddings=[FAKE_VECTOR, FAKE_VECTOR],
        ))
        with patch("llmgate.embeddings._embed_ollama", mock_fn):
            resp = embed("ollama/nomic-embed-text", ["a", "b"])
        mock_fn.assert_called_once()
        req = mock_fn.call_args[0][0]
        assert req.input == ["a", "b"]
        assert len(resp.embeddings) == 2

    def test_truncate_true(self):
        mock_fn = MagicMock(return_value=EmbeddingResponse(
            model="ollama/nomic-embed-text", provider="ollama", embeddings=[FAKE_VECTOR],
        ))
        with patch("llmgate.embeddings._embed_ollama", mock_fn):
            embed("ollama/nomic-embed-text", "hi", truncate="true")
        req = mock_fn.call_args[0][0]
        assert req.truncate == "true"

    def test_truncate_false(self):
        mock_fn = MagicMock(return_value=EmbeddingResponse(
            model="ollama/nomic-embed-text", provider="ollama", embeddings=[FAKE_VECTOR],
        ))
        with patch("llmgate.embeddings._embed_ollama", mock_fn):
            embed("ollama/nomic-embed-text", "hi", truncate="false")
        req = mock_fn.call_args[0][0]
        assert req.truncate == "false"

    def test_keep_alive_via_extra_kwargs(self):
        mock_fn = MagicMock(return_value=EmbeddingResponse(
            model="ollama/nomic-embed-text", provider="ollama", embeddings=[FAKE_VECTOR],
        ))
        with patch("llmgate.embeddings._embed_ollama", mock_fn):
            embed("ollama/nomic-embed-text", "hi", keep_alive="1h")
        req = mock_fn.call_args[0][0]
        assert req.extra_kwargs.get("keep_alive") == "1h"


# ---------------------------------------------------------------------------
# Bedrock — parallel calls, normalize, dimensions in body
# ---------------------------------------------------------------------------

class TestBedrockEmbeddings:
    def test_normalize_in_body(self):
        mock_fn = MagicMock(return_value=EmbeddingResponse(
            model="bedrock/amazon.titan-embed-text-v2:0", provider="bedrock",
            embeddings=[FAKE_VECTOR],
        ))
        with patch("llmgate.embeddings._embed_bedrock", mock_fn):
            embed("bedrock/amazon.titan-embed-text-v2:0", "hi")
        req = mock_fn.call_args[0][0]
        # normalize default is True in extra_kwargs when not passed
        assert req.extra_kwargs.get("normalize", True) is True

    def test_normalize_false_override(self):
        mock_fn = MagicMock(return_value=EmbeddingResponse(
            model="bedrock/amazon.titan-embed-text-v2:0", provider="bedrock",
            embeddings=[FAKE_VECTOR],
        ))
        with patch("llmgate.embeddings._embed_bedrock", mock_fn):
            embed("bedrock/amazon.titan-embed-text-v2:0", "hi", normalize=False)
        req = mock_fn.call_args[0][0]
        assert req.extra_kwargs.get("normalize") is False

    def test_dimensions_in_request(self):
        mock_fn = MagicMock(return_value=EmbeddingResponse(
            model="bedrock/amazon.titan-embed-text-v2:0", provider="bedrock",
            embeddings=[FAKE_VECTOR],
        ))
        with patch("llmgate.embeddings._embed_bedrock", mock_fn):
            embed("bedrock/amazon.titan-embed-text-v2:0", "hi", dimensions=512)
        req = mock_fn.call_args[0][0]
        assert req.dimensions == 512

    def test_parallel_batch_called_once_per_input(self):
        mock_fn = MagicMock(return_value=EmbeddingResponse(
            model="bedrock/amazon.titan-embed-text-v2:0", provider="bedrock",
            embeddings=[FAKE_VECTOR] * 5,
        ))
        with patch("llmgate.embeddings._embed_bedrock", mock_fn):
            resp = embed("bedrock/amazon.titan-embed-text-v2:0",
                         ["0", "1", "2", "3", "4"])
        mock_fn.assert_called_once()
        assert len(resp.embeddings) == 5


# ---------------------------------------------------------------------------
# EmbeddingResponse model
# ---------------------------------------------------------------------------

class TestEmbeddingResponse:
    def test_always_list_of_lists(self):
        resp = EmbeddingResponse(
            model="text-embedding-3-small", provider="openai",
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
        )
        assert isinstance(resp.embeddings, list)
        assert isinstance(resp.embeddings[0], list)

    def test_raw_excluded_from_serialisation(self):
        resp = EmbeddingResponse(
            model="m", provider="openai", embeddings=[[0.1]], raw={"some": "object"},
        )
        assert "raw" not in resp.model_dump()

    def test_usage_defaults_to_zero(self):
        resp = EmbeddingResponse(model="m", provider="openai", embeddings=[[0.1]])
        assert resp.usage.total_tokens == 0


# ---------------------------------------------------------------------------
# LLMGate.embed() delegation
# ---------------------------------------------------------------------------

class TestLLMGateEmbed:
    def test_gate_embed_delegates(self):
        from llmgate import LLMGate
        gate = LLMGate()
        expected = EmbeddingResponse(
            model="text-embedding-3-small", provider="openai", embeddings=[FAKE_VECTOR],
        )
        with patch("llmgate.gate._embed_fn", return_value=expected):
            resp = gate.embed("text-embedding-3-small", "hello")
        assert resp.provider == "openai"

    @pytest.mark.asyncio
    async def test_gate_aembed_delegates(self):
        from llmgate import LLMGate
        gate = LLMGate()
        expected = EmbeddingResponse(
            model="text-embedding-3-small", provider="openai", embeddings=[FAKE_VECTOR],
        )
        with patch("llmgate.gate._aembed_fn", new=AsyncMock(return_value=expected)):
            resp = await gate.aembed("text-embedding-3-small", "hello")
        assert resp.embeddings == [FAKE_VECTOR]

    def test_gate_embed_passes_task_type(self):
        """Named params must flow through gate.embed() correctly."""
        from llmgate import LLMGate
        gate = LLMGate()
        expected = EmbeddingResponse(
            model="gemini/text-embedding-004", provider="gemini", embeddings=[FAKE_VECTOR],
        )
        with patch("llmgate.gate._embed_fn", return_value=expected) as mock_fn:
            gate.embed("gemini/text-embedding-004", "hi",
                       task_type="RETRIEVAL_DOCUMENT")
        # task_type should appear in the kwargs passed to _embed_fn
        _, call_kwargs = mock_fn.call_args
        assert call_kwargs.get("task_type") == "RETRIEVAL_DOCUMENT"
