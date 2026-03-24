"""Tests for llmgate type models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from llmgate.types import (
    Choice,
    CompletionRequest,
    CompletionResponse,
    Message,
    StreamChunk,
    TokenUsage,
)


class TestMessage:
    def test_basic(self):
        m = Message(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"

    def test_to_dict(self):
        m = Message(role="assistant", content="hi")
        assert m.to_dict() == {"role": "assistant", "content": "hi"}

    def test_invalid_role(self):
        with pytest.raises(ValidationError):
            Message(role="unknown", content="x")


class TestCompletionRequest:
    def test_defaults(self):
        req = CompletionRequest(
            model="gpt-4o-mini",
            messages=[Message(role="user", content="hi")],
        )
        assert req.stream is False
        assert req.max_tokens is None
        assert req.extra_kwargs == {}

    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            CompletionRequest(
                model="gpt-4o",
                messages=[Message(role="user", content="hi")],
                temperature=3.0,  # > 2.0 is invalid
            )

    def test_full_params(self):
        req = CompletionRequest(
            model="claude-3-5-sonnet-20241022",
            messages=[Message(role="user", content="hi")],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
        )
        assert req.max_tokens == 512
        assert req.temperature == 0.7


class TestCompletionResponse:
    def _make_response(self):
        return CompletionResponse(
            id="test-id",
            model="gpt-4o-mini",
            provider="openai",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="Hello there!"),
                    finish_reason="stop",
                )
            ],
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    def test_text_shortcut(self):
        resp = self._make_response()
        assert resp.text == "Hello there!"

    def test_text_empty_choices(self):
        resp = CompletionResponse(
            id="x", model="gpt-4o", provider="openai", choices=[]
        )
        assert resp.text == ""

    def test_raw_excluded_from_serialisation(self):
        resp = self._make_response()
        resp.raw = object()  # type: ignore
        d = resp.model_dump()
        assert "raw" not in d


class TestStreamChunk:
    def test_stub_exists(self):
        chunk = StreamChunk(id="c1", model="gpt-4o", provider="openai", delta="Hello")
        assert chunk.delta == "Hello"
        assert chunk.index == 0
