"""
Tests for structured output (Pydantic response_format).
All provider SDK calls are mocked.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from llmgate.structured import (
    build_schema_system_message,
    extract_json,
    get_json_schema,
    inject_schema_prompt,
    validate_parsed,
)
from llmgate.types import CompletionRequest, Message


# ---------------------------------------------------------------------------
# Schema model used in all tests
# ---------------------------------------------------------------------------

class Movie(BaseModel):
    title: str
    year: int
    rating: float = Field(ge=0.0, le=10.0)


MOVIE_JSON = '{"title": "Inception", "year": 2010, "rating": 8.8}'
MOVIE_FENCED = f"```json\n{MOVIE_JSON}\n```"
MOVIE_WITH_PREAMBLE = f"Here is the movie:\n{MOVIE_JSON}"


# ---------------------------------------------------------------------------
# llmgate.structured utilities
# ---------------------------------------------------------------------------

class TestGetJsonSchema:
    def test_returns_dict(self):
        schema = get_json_schema(Movie)
        assert isinstance(schema, dict)
        assert schema["title"] == "Movie"

    def test_contains_required_properties(self):
        schema = get_json_schema(Movie)
        props = schema.get("properties", {})
        assert "title" in props
        assert "year" in props
        assert "rating" in props


class TestExtractJson:
    def test_bare_json(self):
        result = extract_json(MOVIE_JSON)
        assert json.loads(result)["title"] == "Inception"

    def test_fenced_json(self):
        result = extract_json(MOVIE_FENCED)
        assert json.loads(result)["title"] == "Inception"

    def test_json_with_preamble(self):
        result = extract_json(MOVIE_WITH_PREAMBLE)
        assert json.loads(result)["title"] == "Inception"

    def test_no_json_returns_input(self):
        text = "no json here"
        result = extract_json(text)
        assert result == text


class TestValidateParsed:
    def test_bare_json_validates(self):
        movie = validate_parsed(MOVIE_JSON, Movie)
        assert isinstance(movie, Movie)
        assert movie.title == "Inception"
        assert movie.year == 2010

    def test_fenced_json_validates(self):
        movie = validate_parsed(MOVIE_FENCED, Movie)
        assert movie.rating == 8.8

    def test_json_with_preamble_validates(self):
        movie = validate_parsed(MOVIE_WITH_PREAMBLE, Movie)
        assert movie.title == "Inception"

    def test_invalid_json_raises(self):
        with pytest.raises(Exception):  # JSONDecodeError or ValidationError
            validate_parsed("not json at all!", Movie)

    def test_schema_mismatch_raises_validation_error(self):
        from pydantic import ValidationError
        bad = '{"title": "X", "year": "not_an_int", "rating": 5.0}'
        with pytest.raises(ValidationError):
            validate_parsed(bad, Movie)


class TestInjectSchemaPrompt:
    def test_prepends_system_message_if_none(self):
        msgs = [Message(role="user", content="hello")]
        result = inject_schema_prompt(msgs, Movie)
        assert result[0].role == "system"
        assert "Movie" in result[0].content or "JSON" in result[0].content
        assert result[1].role == "user"

    def test_merges_with_existing_system_message(self):
        msgs = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="hello"),
        ]
        result = inject_schema_prompt(msgs, Movie)
        assert result[0].role == "system"
        assert "You are helpful." in result[0].content
        assert len(result) == 2  # no extra message added

    def test_original_list_not_mutated(self):
        msgs = [Message(role="user", content="hello")]
        inject_schema_prompt(msgs, Movie)
        assert len(msgs) == 1  # original list unchanged


class TestBuildSchemaSystemMessage:
    def test_contains_schema_keys(self):
        prompt = build_schema_system_message(Movie)
        assert "title" in prompt
        assert "year" in prompt
        assert "JSON" in prompt


# ---------------------------------------------------------------------------
# Provider-level: _build_params and _map_response with response_format
# ---------------------------------------------------------------------------

def _movie_json_text():
    return MOVIE_JSON


class TestOpenAIStructuredOutput:
    def _make_provider(self):
        from llmgate.providers.openai import OpenAIProvider
        p = OpenAIProvider.__new__(OpenAIProvider)
        p._client = MagicMock()
        p._async_client = MagicMock()
        p._openai = MagicMock()
        return p

    def test_response_format_in_params(self):
        p = self._make_provider()
        req = CompletionRequest(
            model="gpt-4o-mini",
            messages=[Message(role="user", content="name a movie")],
            response_format=Movie,
        )
        params = p._build_params(req)
        assert "response_format" in params
        rf = params["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        assert "schema" in rf["json_schema"]

    def test_parsed_populated_in_map_response(self):
        p = self._make_provider()
        raw = SimpleNamespace(
            id="x",
            choices=[SimpleNamespace(
                index=0,
                message=SimpleNamespace(role="assistant", content=MOVIE_JSON, tool_calls=None),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        )
        resp = p._map_response(raw, "gpt-4o-mini", Movie)
        assert isinstance(resp.parsed, Movie)
        assert resp.parsed.title == "Inception"

    def test_parsed_none_without_response_format(self):
        p = self._make_provider()
        raw = SimpleNamespace(
            id="x",
            choices=[SimpleNamespace(
                index=0,
                message=SimpleNamespace(role="assistant", content="hello", tool_calls=None),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        )
        resp = p._map_response(raw, "gpt-4o-mini", None)
        assert resp.parsed is None


class TestGroqStructuredOutput:
    def _make_provider(self):
        from llmgate.providers.groq import GroqProvider
        p = GroqProvider.__new__(GroqProvider)
        p._client = MagicMock()
        p._async_client = MagicMock()
        return p

    def test_json_object_mode_in_params(self):
        p = self._make_provider()
        req = CompletionRequest(
            model="groq/llama-3.1-8b-instant",
            messages=[Message(role="user", content="name a movie")],
            response_format=Movie,
        )
        params = p._build_params(req)
        assert params.get("response_format") == {"type": "json_object"}

    def test_parsed_populated(self):
        p = self._make_provider()
        raw = SimpleNamespace(
            id="y",
            choices=[SimpleNamespace(
                index=0,
                message=SimpleNamespace(role="assistant", content=MOVIE_JSON, tool_calls=None),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        )
        resp = p._map_response(raw, "groq/llama-3.1-8b-instant", Movie)
        assert isinstance(resp.parsed, Movie)
        assert resp.parsed.year == 2010


class TestAnthropicStructuredOutput:
    def _make_provider(self):
        from llmgate.providers.anthropic import AnthropicProvider
        p = AnthropicProvider.__new__(AnthropicProvider)
        p._client = MagicMock()
        p._async_client = MagicMock()
        p._anthropic = MagicMock()
        return p

    def test_schema_injected_in_messages(self):
        p = self._make_provider()
        req = CompletionRequest(
            model="claude-3-5-haiku-20241022",
            messages=[Message(role="user", content="name a movie")],
            response_format=Movie,
        )
        params = p._build_params(req)
        # Should have a system prompt with JSON schema instruction
        assert "system" in params
        system_text = params["system"]
        assert "JSON" in system_text or "json" in system_text.lower()

    def test_parsed_populated(self):
        p = self._make_provider()
        mock_raw = SimpleNamespace(
            id="z",
            content=[SimpleNamespace(type="text", text=MOVIE_JSON)],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        )
        resp = p._map_response(mock_raw, "claude-3-5-haiku-20241022", Movie)
        assert isinstance(resp.parsed, Movie)
        assert resp.parsed.rating == 8.8


class TestOllamaStructuredOutput:
    def _make_provider(self):
        from llmgate.providers.ollama import OllamaProvider
        p = OllamaProvider.__new__(OllamaProvider)
        p._client = MagicMock()
        p._async_client = MagicMock()
        p._ollama = MagicMock()
        return p

    def test_format_schema_in_params(self):
        p = self._make_provider()
        req = CompletionRequest(
            model="ollama/llama3.2",
            messages=[Message(role="user", content="name a movie")],
            response_format=Movie,
        )
        params = p._build_params(req)
        assert "format" in params
        assert isinstance(params["format"], dict)
        assert "properties" in params["format"]

    def test_parsed_populated(self):
        p = self._make_provider()
        mock_raw = SimpleNamespace(
            message=SimpleNamespace(role="assistant", content=MOVIE_JSON, tool_calls=None),
            done_reason="stop",
            prompt_eval_count=5,
            eval_count=10,
        )
        resp = p._map_response(mock_raw, "ollama/llama3.2", Movie)
        assert isinstance(resp.parsed, Movie)
        assert resp.parsed.title == "Inception"


# ---------------------------------------------------------------------------
# completion() API: response_format passthrough and stream guard
# ---------------------------------------------------------------------------

class TestCompletionAPI:
    def test_stream_with_response_format_raises(self):
        from llmgate.completion import _build_request
        with pytest.raises(ValueError, match="stream=True"):
            _build_request(
                "gpt-4o-mini",
                [{"role": "user", "content": "hi"}],
                stream=True,
                kwargs={"response_format": Movie},
            )

    def test_response_format_passes_through(self):
        from llmgate.completion import _build_request
        req = _build_request(
            "gpt-4o-mini",
            [{"role": "user", "content": "hi"}],
            stream=False,
            kwargs={"response_format": Movie},
        )
        assert req.response_format is Movie

    def test_parse_returns_model_instance(self):
        from llmgate.completion import parse
        mock_resp = MagicMock()
        mock_resp.parsed = Movie(title="Test", year=2000, rating=7.0)
        with patch("llmgate.completion.completion", return_value=mock_resp):
            result = parse("gpt-4o-mini", [], response_format=Movie)
        assert isinstance(result, Movie)
        assert result.title == "Test"

    def test_parse_raises_if_parsed_is_none(self):
        from llmgate.completion import parse
        mock_resp = MagicMock()
        mock_resp.parsed = None
        mock_resp.text = "some text"
        with patch("llmgate.completion.completion", return_value=mock_resp):
            with pytest.raises(ValueError, match="no structured output"):
                parse("gpt-4o-mini", [], response_format=Movie)
