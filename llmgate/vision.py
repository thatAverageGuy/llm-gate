"""
llmgate.vision
~~~~~~~~~~~~~~
Provider-agnostic vision (multimodal) content normalizers.

Each function converts a ``Message.content`` value — which may be either a
plain ``str`` or a ``list[TextPart | ImagePart]`` — into the wire format
expected by a specific provider.  Provider files call these helpers rather
than duplicating serialization logic.

Supported providers and their expected formats:

* **OpenAI / Azure / Groq** — ``image_url`` content-part objects
* **Mistral** — same structure but ``image_url`` is a plain string
* **Anthropic** — ``image`` content blocks with ``source`` sub-object
* **Gemini** — ``google.genai.types.Part`` objects (inline bytes)
* **Bedrock** — Converse API ``image`` content blocks (raw bytes)
* **Ollama** — ``message["images"]`` list of raw base64 strings
"""

from __future__ import annotations

import base64
from typing import Any, Union

from llmgate.types import ImagePart, TextPart

# Type alias for the content field on a Message
ContentType = Union[str, list[Union[TextPart, ImagePart]]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_url_bytes(url: str) -> tuple[bytes, str]:
    """Fetch image bytes from an HTTPS URL and return ``(raw_bytes, mime_type)``.

    Used by providers that have no native URL-reference mode (Gemini, Ollama, Bedrock).
    The ``httpx`` package is already a transitive dependency of most LLM SDKs.
    """
    try:
        import httpx  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "httpx is required to fetch image URLs. Install it: pip install httpx"
        ) from exc

    resp = httpx.get(url, follow_redirects=True, timeout=30.0)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    return resp.content, content_type


def _is_data_uri(url: str) -> bool:
    return url.startswith("data:")


def _data_uri_to_bytes(url: str) -> tuple[bytes, str]:
    """Parse a ``data:image/jpeg;base64,...`` URI into ``(raw_bytes, mime_type)``."""
    # data:<mime>;base64,<data>
    header, encoded = url.split(",", 1)
    mime_type = header.split(";")[0].replace("data:", "").strip()
    raw = base64.b64decode(encoded)
    return raw, mime_type


# ---------------------------------------------------------------------------
# OpenAI / Azure / Groq
# ---------------------------------------------------------------------------


def to_openai_content(
    content: ContentType,
    *,
    include_detail: bool = True,
) -> str | list[dict[str, Any]]:
    """Serialize ``content`` for the OpenAI / Azure wire format.

    If ``content`` is a plain ``str`` it is returned unchanged.
    Otherwise, a list of OpenAI content-part dicts is returned.

    ``include_detail=False`` strips the ``detail`` field (used by Groq which
    does not support that parameter).
    """
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            if part.type == "image_url":
                assert part.image_url is not None
                img_obj: dict[str, Any] = {"url": part.image_url.url}
                if include_detail and part.image_url.detail is not None:
                    img_obj["detail"] = part.image_url.detail
                parts.append({"type": "image_url", "image_url": img_obj})
            else:  # image_bytes
                assert part.image_bytes is not None
                ib = part.image_bytes
                data_uri = f"data:{ib.mime_type};base64,{ib.data}"
                parts.append({"type": "image_url", "image_url": {"url": data_uri}})
        else:
            # Fallback: plain dict passed through by a lenient caller
            parts.append(dict(part))  # type: ignore[arg-type]
    return parts


def to_groq_content(content: ContentType) -> str | list[dict[str, Any]]:
    """Like :func:`to_openai_content` but strips the ``detail`` field."""
    return to_openai_content(content, include_detail=False)


# ---------------------------------------------------------------------------
# Mistral
# ---------------------------------------------------------------------------


def to_mistral_content(content: ContentType) -> str | list[dict[str, Any]]:
    """Serialize ``content`` for the Mistral wire format.

    Mistral's ``image_url`` field is a *plain string*, not an object.
    Text parts are identical to OpenAI.
    """
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            if part.type == "image_url":
                assert part.image_url is not None
                parts.append({"type": "image_url", "image_url": part.image_url.url})
            else:  # image_bytes
                assert part.image_bytes is not None
                ib = part.image_bytes
                data_uri = f"data:{ib.mime_type};base64,{ib.data}"
                parts.append({"type": "image_url", "image_url": data_uri})
        else:
            parts.append(dict(part))  # type: ignore[arg-type]
    return parts


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def to_anthropic_content(
    content: ContentType,
) -> str | list[dict[str, Any]]:
    """Serialize ``content`` for the Anthropic Messages API.

    Returns a list of Anthropic content blocks.  Plain-string content is
    returned unchanged.
    """
    if isinstance(content, str):
        return content
    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            if part.type == "image_url":
                assert part.image_url is not None
                url = part.image_url.url
                if _is_data_uri(url):
                    raw, mime = _data_uri_to_bytes(url)
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime,
                                "data": base64.b64encode(raw).decode(),
                            },
                        }
                    )
                else:
                    blocks.append(
                        {
                            "type": "image",
                            "source": {"type": "url", "url": url},
                        }
                    )
            else:  # image_bytes
                assert part.image_bytes is not None
                ib = part.image_bytes
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": ib.mime_type,
                            "data": ib.data,
                        },
                    }
                )
        else:
            blocks.append(dict(part))  # type: ignore[arg-type]
    return blocks


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


def to_gemini_parts(content: ContentType) -> list[Any]:
    """Return a list of ``google.genai.types.Part`` objects for Gemini.

    URL images are fetched client-side because Gemini has no native
    URL-reference mode for images.  Data URIs are decoded locally.
    Plain-string content returns a single ``Part.from_text(...)`` element.
    """
    try:
        from google.genai import types as genai_types  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "google-genai package is required: uv add google-genai"
        ) from exc

    if isinstance(content, str):
        return [genai_types.Part.from_text(text=content)]

    parts: list[Any] = []
    for part in content:
        if isinstance(part, TextPart):
            parts.append(genai_types.Part.from_text(text=part.text))
        elif isinstance(part, ImagePart):
            if part.type == "image_url":
                assert part.image_url is not None
                url = part.image_url.url
                if _is_data_uri(url):
                    raw, mime = _data_uri_to_bytes(url)
                else:
                    raw, mime = _fetch_url_bytes(url)
                parts.append(genai_types.Part.from_bytes(data=raw, mime_type=mime))
            else:  # image_bytes
                assert part.image_bytes is not None
                ib = part.image_bytes
                raw = base64.b64decode(ib.data)
                parts.append(
                    genai_types.Part.from_bytes(data=raw, mime_type=ib.mime_type)
                )
        else:
            parts.append(part)
    return parts


# ---------------------------------------------------------------------------
# Bedrock
# ---------------------------------------------------------------------------

# Bedrock Converse API "format" strings (no "image/" prefix)
_MIME_TO_BEDROCK_FORMAT = {
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}


def to_bedrock_content(content: ContentType) -> list[dict[str, Any]]:
    """Serialize ``content`` into Bedrock Converse API content blocks.

    Returns a list of blocks.  URL images are fetched client-side;
    base64 strings are decoded to raw bytes (boto3 handles re-encoding).
    Plain-string ``content`` returns a single ``{"text": ...}`` block.
    """
    if isinstance(content, str):
        return [{"text": content}]

    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextPart):
            blocks.append({"text": part.text})
        elif isinstance(part, ImagePart):
            if part.type == "image_url":
                assert part.image_url is not None
                url = part.image_url.url
                if _is_data_uri(url):
                    raw, mime = _data_uri_to_bytes(url)
                else:
                    raw, mime = _fetch_url_bytes(url)
                fmt = _MIME_TO_BEDROCK_FORMAT.get(mime, "jpeg")
                blocks.append(
                    {
                        "image": {
                            "format": fmt,
                            "source": {"bytes": raw},
                        }
                    }
                )
            else:  # image_bytes
                assert part.image_bytes is not None
                ib = part.image_bytes
                raw = base64.b64decode(ib.data)
                fmt = _MIME_TO_BEDROCK_FORMAT.get(ib.mime_type, "jpeg")
                blocks.append(
                    {
                        "image": {
                            "format": fmt,
                            "source": {"bytes": raw},
                        }
                    }
                )
        else:
            blocks.append(dict(part))  # type: ignore[arg-type]
    return blocks


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


def to_ollama_message(
    role: str,
    content: ContentType,
) -> dict[str, Any]:
    """Build an Ollama message dict from role + content.

    Ollama uses a top-level ``images`` field (list of raw base64 strings)
    alongside the text ``content`` field — it does NOT use a content-parts
    array.  URL images are fetched and re-encoded to base64.

    Returns a dict ready to pass as an Ollama chat message.
    """
    if isinstance(content, str):
        return {"role": role, "content": content}

    text_parts: list[str] = []
    images: list[str] = []

    for part in content:
        if isinstance(part, TextPart):
            text_parts.append(part.text)
        elif isinstance(part, ImagePart):
            if part.type == "image_url":
                assert part.image_url is not None
                url = part.image_url.url
                if _is_data_uri(url):
                    raw, _ = _data_uri_to_bytes(url)
                    images.append(base64.b64encode(raw).decode())
                else:
                    raw, _ = _fetch_url_bytes(url)
                    images.append(base64.b64encode(raw).decode())
            else:  # image_bytes (already base64)
                assert part.image_bytes is not None
                images.append(part.image_bytes.data)

    msg: dict[str, Any] = {"role": role, "content": " ".join(text_parts)}
    if images:
        msg["images"] = images
    return msg
