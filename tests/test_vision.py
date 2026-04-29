"""
tests/test_vision.py
~~~~~~~~~~~~~~~~~~~~
Tests for the vision / multimodal feature (v0.5.0).

All external SDK calls are mocked.  We test:
  1. Type validation (ImageURL, ImageBytes, TextPart, ImagePart, Message)
  2. Vision normalizer helpers in llmgate/vision.py
  3. Per-provider message serialization with image content
  4. VisionNotSupported raised for Cohere
"""
from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest

from llmgate.exceptions import VisionNotSupported
from llmgate.types import ImageBytes, ImagePart, ImageURL, Message, TextPart
from llmgate import vision


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

JPEG_B64 = base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 10).decode()
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
PNG_B64 = base64.b64encode(PNG_BYTES).decode()


def _make_image_url_part(url: str, detail: str | None = None) -> ImagePart:
    return ImagePart(type="image_url", image_url=ImageURL(url=url, detail=detail))


def _make_image_bytes_part(data: str = JPEG_B64, mime: str = "image/jpeg") -> ImagePart:
    return ImagePart(type="image_bytes", image_bytes=ImageBytes(data=data, mime_type=mime))


def _make_text_part(text: str = "What's in this image?") -> TextPart:
    return TextPart(text=text)


# ---------------------------------------------------------------------------
# 1. Type Validation
# ---------------------------------------------------------------------------


class TestImageURL:
    def test_valid(self):
        img = ImageURL(url="https://example.com/img.jpg")
        assert img.url == "https://example.com/img.jpg"
        assert img.detail is None

    def test_with_detail(self):
        img = ImageURL(url="https://example.com/img.jpg", detail="high")
        assert img.detail == "high"

    def test_invalid_detail(self):
        with pytest.raises(Exception):
            ImageURL(url="https://x.com/img.jpg", detail="ultra")  # type: ignore


class TestImageBytes:
    def test_valid(self):
        ib = ImageBytes(data=JPEG_B64, mime_type="image/jpeg")
        assert ib.data == JPEG_B64
        assert ib.mime_type == "image/jpeg"


class TestTextPart:
    def test_valid(self):
        tp = TextPart(text="Hello")
        assert tp.type == "text"
        assert tp.text == "Hello"

    def test_default_type(self):
        assert TextPart(text="hi").type == "text"


class TestImagePart:
    def test_url_variant_valid(self):
        part = _make_image_url_part("https://example.com/img.jpg")
        assert part.type == "image_url"
        assert part.image_url is not None

    def test_bytes_variant_valid(self):
        part = _make_image_bytes_part()
        assert part.type == "image_bytes"
        assert part.image_bytes is not None

    def test_url_variant_missing_payload_raises(self):
        with pytest.raises(Exception, match="image_url is required"):
            ImagePart(type="image_url")

    def test_bytes_variant_missing_payload_raises(self):
        with pytest.raises(Exception, match="image_bytes is required"):
            ImagePart(type="image_bytes")


class TestMessage:
    def test_str_content_unchanged(self):
        msg = Message(role="user", content="Hello")
        assert msg.content == "Hello"
        assert msg.to_dict() == {"role": "user", "content": "Hello"}

    def test_none_content(self):
        msg = Message(role="assistant", content=None)
        assert msg.content is None

    def test_multipart_content(self):
        parts = [_make_text_part("Describe this"), _make_image_url_part("https://x.com/a.jpg")]
        msg = Message(role="user", content=parts)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_to_dict_with_str_content(self):
        msg = Message(role="user", content="hi")
        d = msg.to_dict()
        assert d["content"] == "hi"


# ---------------------------------------------------------------------------
# 2. Vision Normalizer
# ---------------------------------------------------------------------------


class TestToOpenAIContent:
    def test_str_passthrough(self):
        result = vision.to_openai_content("hello")
        assert result == "hello"

    def test_text_part(self):
        parts = [_make_text_part("Describe this")]
        result = vision.to_openai_content(parts)
        assert isinstance(result, list)
        assert result[0] == {"type": "text", "text": "Describe this"}

    def test_image_url_part(self):
        parts = [_make_image_url_part("https://example.com/img.jpg", detail="low")]
        result = vision.to_openai_content(parts)
        assert result[0]["type"] == "image_url"
        assert result[0]["image_url"]["url"] == "https://example.com/img.jpg"
        assert result[0]["image_url"]["detail"] == "low"

    def test_image_bytes_becomes_data_uri(self):
        parts = [_make_image_bytes_part(JPEG_B64, "image/jpeg")]
        result = vision.to_openai_content(parts)
        assert result[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_mixed_parts(self):
        parts = [
            _make_text_part("What's here?"),
            _make_image_url_part("https://x.com/img.jpg"),
        ]
        result = vision.to_openai_content(parts)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image_url"

    def test_include_detail_false_strips_detail(self):
        parts = [_make_image_url_part("https://x.com/img.jpg", detail="high")]
        result = vision.to_openai_content(parts, include_detail=False)
        assert "detail" not in result[0]["image_url"]


class TestToGroqContent:
    def test_strips_detail(self):
        parts = [_make_image_url_part("https://x.com/img.jpg", detail="high")]
        result = vision.to_groq_content(parts)
        assert "detail" not in result[0]["image_url"]


class TestToMistralContent:
    def test_str_passthrough(self):
        assert vision.to_mistral_content("hi") == "hi"

    def test_image_url_is_plain_string(self):
        parts = [_make_image_url_part("https://example.com/img.jpg")]
        result = vision.to_mistral_content(parts)
        assert result[0]["image_url"] == "https://example.com/img.jpg"
        # Must be a string, not a dict
        assert isinstance(result[0]["image_url"], str)

    def test_image_bytes_is_data_uri_string(self):
        parts = [_make_image_bytes_part(JPEG_B64, "image/jpeg")]
        result = vision.to_mistral_content(parts)
        assert result[0]["image_url"].startswith("data:image/jpeg;base64,")
        assert isinstance(result[0]["image_url"], str)


class TestToAnthropicContent:
    def test_str_passthrough(self):
        assert vision.to_anthropic_content("hi") == "hi"

    def test_text_part(self):
        parts = [_make_text_part("Hello")]
        result = vision.to_anthropic_content(parts)
        assert result == [{"type": "text", "text": "Hello"}]

    def test_image_url_becomes_url_source(self):
        parts = [_make_image_url_part("https://example.com/img.jpg")]
        result = vision.to_anthropic_content(parts)
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "url"
        assert result[0]["source"]["url"] == "https://example.com/img.jpg"

    def test_image_bytes_becomes_base64_source(self):
        parts = [_make_image_bytes_part(JPEG_B64, "image/jpeg")]
        result = vision.to_anthropic_content(parts)
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["media_type"] == "image/jpeg"
        assert result[0]["source"]["data"] == JPEG_B64

    def test_data_uri_decoded_to_base64_source(self):
        data_uri = f"data:image/jpeg;base64,{JPEG_B64}"
        parts = [_make_image_url_part(data_uri)]
        result = vision.to_anthropic_content(parts)
        assert result[0]["source"]["type"] == "base64"


class TestToGeminiParts:
    def test_str_returns_text_part(self):
        if not __import__("importlib").util.find_spec("google.genai"):
            pytest.skip("google-genai not installed")
        result = vision.to_gemini_parts("hello")
        assert len(result) == 1

    def test_text_part(self):
        if not __import__("importlib").util.find_spec("google.genai"):
            pytest.skip("google-genai not installed")
        parts = [_make_text_part("Describe")]
        result = vision.to_gemini_parts(parts)
        assert len(result) == 1


class TestToBedrockContent:
    def test_str_becomes_text_block(self):
        result = vision.to_bedrock_content("Hello")
        assert result == [{"text": "Hello"}]

    def test_text_part(self):
        result = vision.to_bedrock_content([_make_text_part("Hello")])
        assert result[0] == {"text": "Hello"}

    def test_image_bytes_block(self):
        result = vision.to_bedrock_content([_make_image_bytes_part(JPEG_B64, "image/jpeg")])
        block = result[0]
        assert "image" in block
        assert block["image"]["format"] == "jpeg"
        assert isinstance(block["image"]["source"]["bytes"], bytes)

    def test_unknown_mime_defaults_to_jpeg(self):
        result = vision.to_bedrock_content([_make_image_bytes_part(JPEG_B64, "image/unknown")])
        assert result[0]["image"]["format"] == "jpeg"


class TestToOllamaMessage:
    def test_str_content(self):
        msg = vision.to_ollama_message("user", "Hello")
        assert msg == {"role": "user", "content": "Hello"}

    def test_text_part_only(self):
        parts = [_make_text_part("Describe")]
        msg = vision.to_ollama_message("user", parts)
        assert msg["content"] == "Describe"
        assert "images" not in msg

    def test_image_bytes_extracted(self):
        parts = [_make_text_part("What?"), _make_image_bytes_part(JPEG_B64, "image/jpeg")]
        msg = vision.to_ollama_message("user", parts)
        assert msg["images"] == [JPEG_B64]
        assert msg["content"] == "What?"

    def test_multiple_images(self):
        parts = [_make_image_bytes_part(JPEG_B64), _make_image_bytes_part(PNG_B64, "image/png")]
        msg = vision.to_ollama_message("user", parts)
        assert len(msg["images"]) == 2


# ---------------------------------------------------------------------------
# 3. Provider Vision Tests (mocked)
# ---------------------------------------------------------------------------


def _make_openai_response(text: str = "An image.") -> MagicMock:
    msg = MagicMock()
    msg.content = text
    msg.role = "assistant"
    msg.tool_calls = None
    choice = MagicMock()
    choice.index = 0
    choice.message = msg
    choice.finish_reason = "stop"
    raw = MagicMock()
    raw.id = "chatcmpl-test"
    raw.choices = [choice]
    raw.usage.prompt_tokens = 10
    raw.usage.completion_tokens = 5
    raw.usage.total_tokens = 15
    return raw


class TestOpenAIVision:
    def test_url_image(self):
        from llmgate.providers.openai import OpenAIProvider
        with patch("openai.OpenAI") as mock_cls, patch("openai.AsyncOpenAI"):
            client = mock_cls.return_value
            client.chat.completions.create.return_value = _make_openai_response()
            provider = OpenAIProvider.__new__(OpenAIProvider)
            provider._openai = MagicMock()
            provider._client = client
            provider._async_client = MagicMock()
            provider.name = "openai"

            from llmgate.types import CompletionRequest
            parts = [_make_text_part("What's in this image?"), _make_image_url_part("https://x.com/img.jpg")]
            req = CompletionRequest(model="gpt-4o-mini", messages=[
                Message(role="user", content=parts)
            ])
            resp = provider.complete(req)
            assert resp.text == "An image."

            call_args = client.chat.completions.create.call_args
            messages = call_args[1]["messages"] if call_args[1] else call_args[0][0]
            msg_content = messages[0]["content"]
            assert isinstance(msg_content, list)
            assert any(p.get("type") == "image_url" for p in msg_content)

    def test_base64_image(self):
        from llmgate.providers.openai import OpenAIProvider
        with patch("openai.OpenAI") as mock_cls, patch("openai.AsyncOpenAI"):
            client = mock_cls.return_value
            client.chat.completions.create.return_value = _make_openai_response()
            provider = OpenAIProvider.__new__(OpenAIProvider)
            provider._openai = MagicMock()
            provider._client = client
            provider._async_client = MagicMock()
            provider.name = "openai"

            from llmgate.types import CompletionRequest
            parts = [_make_image_bytes_part(JPEG_B64, "image/jpeg"), _make_text_part("Describe")]
            req = CompletionRequest(model="gpt-4o-mini", messages=[
                Message(role="user", content=parts)
            ])
            provider.complete(req)
            call_args = client.chat.completions.create.call_args
            messages = call_args[1]["messages"] if call_args[1] else call_args[0][0]
            url = messages[0]["content"][0]["image_url"]["url"]
            assert url.startswith("data:image/jpeg;base64,")

    def test_str_content_unaffected(self):
        """Regression: plain text messages still work."""
        from llmgate.providers.openai import OpenAIProvider
        with patch("openai.OpenAI") as mock_cls, patch("openai.AsyncOpenAI"):
            client = mock_cls.return_value
            client.chat.completions.create.return_value = _make_openai_response("Hello!")
            provider = OpenAIProvider.__new__(OpenAIProvider)
            provider._openai = MagicMock()
            provider._client = client
            provider._async_client = MagicMock()
            provider.name = "openai"

            from llmgate.types import CompletionRequest
            req = CompletionRequest(model="gpt-4o-mini", messages=[
                Message(role="user", content="Hello!")
            ])
            resp = provider.complete(req)
            assert resp.text == "Hello!"


class TestAnthropicVision:
    def _make_response(self, text: str = "An image.") -> MagicMock:
        raw = MagicMock()
        block = MagicMock()
        block.type = "text"
        block.text = text
        raw.content = [block]
        raw.id = "msg-test"
        raw.stop_reason = "end_turn"
        raw.usage.input_tokens = 10
        raw.usage.output_tokens = 5
        return raw

    def test_url_image(self):
        from llmgate.providers.anthropic import AnthropicProvider
        with patch("anthropic.Anthropic") as mock_cls, patch("anthropic.AsyncAnthropic"):
            client = mock_cls.return_value
            client.messages.create.return_value = self._make_response()
            provider = AnthropicProvider.__new__(AnthropicProvider)
            provider._anthropic = MagicMock()
            provider._client = client
            provider._async_client = MagicMock()
            provider.name = "anthropic"

            from llmgate.types import CompletionRequest
            parts = [_make_image_url_part("https://x.com/img.jpg"), _make_text_part("Describe")]
            req = CompletionRequest(model="claude-opus-4-7", messages=[
                Message(role="user", content=parts)
            ])
            resp = provider.complete(req)
            assert resp.text == "An image."

            call_args = client.messages.create.call_args
            msgs = call_args[1]["messages"]
            content = msgs[0]["content"]
            assert any(b.get("type") == "image" for b in content)

    def test_base64_image(self):
        from llmgate.providers.anthropic import AnthropicProvider
        with patch("anthropic.Anthropic") as mock_cls, patch("anthropic.AsyncAnthropic"):
            client = mock_cls.return_value
            client.messages.create.return_value = self._make_response()
            provider = AnthropicProvider.__new__(AnthropicProvider)
            provider._anthropic = MagicMock()
            provider._client = client
            provider._async_client = MagicMock()
            provider.name = "anthropic"

            from llmgate.types import CompletionRequest
            parts = [_make_image_bytes_part(JPEG_B64, "image/jpeg"), _make_text_part("Describe")]
            req = CompletionRequest(model="claude-opus-4-7", messages=[
                Message(role="user", content=parts)
            ])
            provider.complete(req)
            call_args = client.messages.create.call_args
            msgs = call_args[1]["messages"]
            content = msgs[0]["content"]
            img_block = next(b for b in content if b.get("type") == "image")
            assert img_block["source"]["type"] == "base64"
            assert img_block["source"]["data"] == JPEG_B64


class TestMistralVision:
    def _make_response(self) -> MagicMock:
        raw = MagicMock()
        msg = MagicMock()
        msg.content = "A cat."
        msg.role = "assistant"
        msg.tool_calls = None
        c = MagicMock()
        c.index = 0
        c.message = msg
        c.finish_reason = "stop"
        raw.choices = [c]
        raw.id = "mistral-test"
        raw.usage.prompt_tokens = 5
        raw.usage.completion_tokens = 3
        raw.usage.total_tokens = 8
        return raw

    def test_image_url_is_string(self):
        import sys
        mock_mistralai = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.chat.complete.return_value = self._make_response()
        mock_mistralai.client.Mistral.return_value = mock_client_instance
        with patch.dict(sys.modules, {"mistralai": mock_mistralai, "mistralai.client": mock_mistralai.client}):
            from llmgate.providers.mistral import MistralProvider
            provider = MistralProvider.__new__(MistralProvider)
            provider._client = mock_client_instance
            provider.name = "mistral"

            from llmgate.types import CompletionRequest
            parts = [_make_image_url_part("https://x.com/img.jpg"), _make_text_part("Describe")]
            req = CompletionRequest(model="mistral/mistral-small-latest", messages=[
                Message(role="user", content=parts)
            ])
            provider.complete(req)
            call_args = mock_client_instance.chat.complete.call_args
            msgs = call_args[1]["messages"]
            content = msgs[0]["content"]
            img_part = next(p for p in content if p.get("type") == "image_url")
            # Mistral: image_url must be a plain string
            assert isinstance(img_part["image_url"], str)


class TestGroqVision:
    def test_no_detail_field(self):
        from llmgate.providers.groq import GroqProvider
        with patch("groq.Groq") as mock_cls, patch("groq.AsyncGroq"):
            client = mock_cls.return_value
            client.chat.completions.create.return_value = _make_openai_response()
            provider = GroqProvider.__new__(GroqProvider)
            provider._groq = MagicMock()
            provider._client = client
            provider._async_client = MagicMock()
            provider.name = "groq"

            from llmgate.types import CompletionRequest
            parts = [
                _make_image_url_part("https://x.com/img.jpg", detail="high"),
                _make_text_part("Describe")
            ]
            req = CompletionRequest(
                model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[Message(role="user", content=parts)]
            )
            provider.complete(req)
            call_args = client.chat.completions.create.call_args
            messages = call_args[1]["messages"] if call_args[1] else call_args[0][0]
            content = messages[0]["content"]
            img_part = next(p for p in content if p.get("type") == "image_url")
            assert "detail" not in img_part["image_url"]


class TestBedrockVision:
    def _make_response(self) -> dict:
        return {
            "output": {"message": {"content": [{"text": "An image."}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
            "ResponseMetadata": {"RequestId": "bedrock-test"},
        }

    def test_image_bytes_block(self):
        from llmgate.providers.bedrock import BedrockProvider
        provider = BedrockProvider.__new__(BedrockProvider)
        mock_brt = MagicMock()
        mock_brt.converse.return_value = self._make_response()
        provider._client = mock_brt
        provider._boto3 = MagicMock()
        provider._region = "us-east-1"
        provider.name = "bedrock"

        from llmgate.types import CompletionRequest
        parts = [_make_image_bytes_part(JPEG_B64, "image/jpeg"), _make_text_part("Describe")]
        req = CompletionRequest(
            model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[Message(role="user", content=parts)]
        )
        provider.complete(req)
        call_args = mock_brt.converse.call_args
        msgs = call_args[1]["messages"]
        content = msgs[0]["content"]
        img_block = next(b for b in content if "image" in b)
        assert img_block["image"]["format"] == "jpeg"
        assert isinstance(img_block["image"]["source"]["bytes"], bytes)


class TestOllamaVision:
    def _make_response(self) -> MagicMock:
        raw = MagicMock()
        raw.message.content = "An image."
        raw.message.role = "assistant"
        raw.message.tool_calls = None
        raw.done_reason = "stop"
        raw.prompt_eval_count = 10
        raw.eval_count = 5
        return raw

    def test_image_bytes_in_images_field(self):
        from llmgate.providers.ollama import OllamaProvider
        provider = OllamaProvider.__new__(OllamaProvider)
        mock_client = MagicMock()
        mock_client.chat.return_value = self._make_response()
        provider._client = mock_client
        provider._async_client = MagicMock()
        provider._ollama = MagicMock()
        provider.name = "ollama"

        from llmgate.types import CompletionRequest
        parts = [_make_text_part("What's here?"), _make_image_bytes_part(JPEG_B64, "image/jpeg")]
        req = CompletionRequest(
            model="ollama/llava",
            messages=[Message(role="user", content=parts)]
        )
        provider.complete(req)
        call_args = mock_client.chat.call_args
        msgs = call_args[1]["messages"]
        assert "images" in msgs[0]
        assert msgs[0]["images"] == [JPEG_B64]

    def test_str_content_unaffected(self):
        from llmgate.providers.ollama import OllamaProvider
        provider = OllamaProvider.__new__(OllamaProvider)
        mock_client = MagicMock()
        mock_client.chat.return_value = self._make_response()
        provider._client = mock_client
        provider._async_client = MagicMock()
        provider._ollama = MagicMock()
        provider.name = "ollama"

        from llmgate.types import CompletionRequest
        req = CompletionRequest(
            model="ollama/llava",
            messages=[Message(role="user", content="What's up?")]
        )
        provider.complete(req)
        call_args = mock_client.chat.call_args
        msgs = call_args[1]["messages"]
        assert "images" not in msgs[0]
        assert msgs[0]["content"] == "What's up?"


class TestCohereVisionNotSupported:
    def test_image_content_raises(self):
        from llmgate.providers.cohere import CohereProvider
        provider = CohereProvider.__new__(CohereProvider)
        provider._cohere = MagicMock()
        provider._client = MagicMock()
        provider._async_client = MagicMock()
        provider.name = "cohere"

        from llmgate.types import CompletionRequest
        parts = [_make_image_url_part("https://x.com/img.jpg"), _make_text_part("Describe")]
        req = CompletionRequest(
            model="cohere/command-r-plus",
            messages=[Message(role="user", content=parts)]
        )
        with pytest.raises(VisionNotSupported) as exc_info:
            provider.complete(req)
        assert "cohere" in str(exc_info.value)

    def test_str_content_still_works(self):
        """Text-only requests to Cohere should be unaffected."""
        from llmgate.providers.cohere import CohereProvider
        mock_response = MagicMock()
        mock_response.message.content = [MagicMock(text="hi")]
        mock_response.message.tool_calls = None
        mock_response.id = "cohere-test"
        mock_response.usage = None
        provider = CohereProvider.__new__(CohereProvider)
        provider._cohere = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.return_value = mock_response
        provider._client = mock_client
        provider._async_client = MagicMock()
        provider.name = "cohere"

        from llmgate.types import CompletionRequest
        req = CompletionRequest(
            model="cohere/command-r-plus",
            messages=[Message(role="user", content="Hello!")]
        )
        resp = provider.complete(req)
        assert resp.text == "hi"


# ---------------------------------------------------------------------------
# 4. Package-level exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_vision_types_exported(self):
        import llmgate
        assert hasattr(llmgate, "ImageURL")
        assert hasattr(llmgate, "ImageBytes")
        assert hasattr(llmgate, "TextPart")
        assert hasattr(llmgate, "ImagePart")
        assert hasattr(llmgate, "VisionNotSupported")

    def test_version_bumped(self):
        import llmgate
        # Check version is a valid semver string (not hardcoded, so this never breaks on bumps)
        parts = llmgate.__version__.split(".")
        assert len(parts) == 3, f"Expected semver X.Y.Z, got {llmgate.__version__!r}"
        assert all(p.isdigit() for p in parts), f"Non-numeric version parts: {llmgate.__version__!r}"
        assert tuple(int(p) for p in parts) >= (0, 7, 0), (
            f"Version {llmgate.__version__!r} is older than v0.7.0"
        )
