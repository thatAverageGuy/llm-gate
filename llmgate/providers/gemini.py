"""
llmgate.providers.gemini
~~~~~~~~~~~~~~~~~~~~~~~~
Google Gemini provider — wraps the official ``google-genai`` SDK (v1+).

Supported model prefixes: ``gemini-``

Uses the new ``google-genai`` package (google.genai), which replaced the
deprecated ``google-generativeai`` (google.generativeai).

Tool Calling:
  - Tools sent as ``function_declarations`` inside a ``tools`` list.
  - Model responses with function calls are in candidate content parts.
  - Tool results are sent back as ``function_response`` parts in a user turn.
"""
from __future__ import annotations

import os
import uuid
from typing import Any, AsyncIterator, ClassVar, Iterator

from llmgate.base import BaseProvider
from llmgate.exceptions import AuthError, ProviderAPIError, RateLimitError
from llmgate.types import (
    Choice, CompletionRequest, CompletionResponse, Message, StreamChunk, ToolCall, TokenUsage,
)


class GeminiProvider(BaseProvider):
    name: ClassVar[str] = "gemini"
    supported_model_prefixes: ClassVar[tuple[str, ...]] = ("gemini-",)

    def __init__(self, api_key: str | None = None, **_client_kwargs: Any) -> None:
        try:
            from google import genai  # noqa: PLC0415
            from google.genai import types as genai_types  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "google-genai package is required: uv add google-genai"
            ) from e

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise AuthError(
                "Gemini API key not found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) env var "
                "or pass api_key=...",
                provider=self.name,
            )
        self._client = genai.Client(api_key=resolved_key)
        self._genai_types = genai_types

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_genai_contents(messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert llmgate messages to google-genai ``contents`` format.

        Handles tool results (role="tool") by injecting function_response parts
        into a user turn.

        Returns:
            (system_instruction, contents_list)
        """
        system_instruction: str | None = None
        contents: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
                continue

            if msg.role == "user":
                contents.append({"role": "user", "parts": [{"text": msg.content or ""}]})
                continue

            if msg.role == "assistant":
                parts: list[dict[str, Any]] = []
                if msg.content:
                    parts.append({"text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        parts.append({
                            "function_call": {
                                "name": tc.function,
                                "args": tc.arguments,
                            }
                        })
                contents.append({"role": "model", "parts": parts})
                continue

            if msg.role == "tool":
                # Tool result — sent as a user turn with function_response part
                try:
                    import json as _json  # noqa: PLC0415
                    result_data = _json.loads(msg.content or "{}")
                    if not isinstance(result_data, dict):
                        result_data = {"result": result_data}
                except Exception:  # noqa: BLE001
                    result_data = {"result": msg.content or ""}
                contents.append({
                    "role": "user",
                    "parts": [{
                        "function_response": {
                            "name": msg.name or "",
                            "response": result_data,
                        }
                    }],
                })

        return system_instruction, contents

    def _build_config(self, request: CompletionRequest) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if request.max_tokens is not None:
            config["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.top_p is not None:
            config["top_p"] = request.top_p
        if request.tools:
            config["tools"] = [{
                "function_declarations": [
                    {
                        "name": t.function.name,
                        "description": t.function.description,
                        "parameters": t.function.parameters,
                    }
                    for t in request.tools
                ]
            }]
        config.update(request.extra_kwargs)
        return config

    def _map_response(self, raw: Any, model: str) -> CompletionResponse:
        text = raw.text if hasattr(raw, "text") and raw.text else None
        usage_meta = getattr(raw, "usage_metadata", None)
        usage = TokenUsage(
            prompt_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
            total_tokens=getattr(usage_meta, "total_token_count", 0) or 0,
        )
        finish_reason = None
        tool_calls: list[ToolCall] = []

        if raw.candidates:
            cand = raw.candidates[0]
            finish_reason = str(cand.finish_reason)
            for part in getattr(cand.content, "parts", []):
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    tool_calls.append(ToolCall(
                        id=str(uuid.uuid4()),
                        function=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                    ))

        return CompletionResponse(
            id=str(uuid.uuid4()),
            model=model,
            provider=self.name,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=text,
                        tool_calls=tool_calls or None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
            raw=raw,
        )

    def _handle_error(self, exc: Exception) -> None:
        msg = str(exc)
        err_type = type(exc).__name__
        if err_type in ("Unauthenticated", "PermissionDenied"):
            raise AuthError(msg, provider=self.name) from exc
        if err_type == "ResourceExhausted":
            raise RateLimitError(msg, provider=self.name) from exc
        raise ProviderAPIError(msg, provider=self.name) from exc

    def _make_full_config(self, request: CompletionRequest, system_instruction: str | None) -> dict[str, Any]:
        config = self._build_config(request)
        if system_instruction:
            config["system_instruction"] = system_instruction
        return config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        system_instruction, contents = self._to_genai_contents(request.messages)
        try:
            raw = self._client.models.generate_content(
                model=request.model,
                contents=contents,
                config=self._make_full_config(request, system_instruction),
            )
        except Exception as exc:  # noqa: BLE001
            self._handle_error(exc)
        return self._map_response(raw, request.model)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        system_instruction, contents = self._to_genai_contents(request.messages)
        try:
            raw = await self._client.aio.models.generate_content(
                model=request.model,
                contents=contents,
                config=self._make_full_config(request, system_instruction),
            )
        except Exception as exc:  # noqa: BLE001
            self._handle_error(exc)
        return self._map_response(raw, request.model)

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        system_instruction, contents = self._to_genai_contents(request.messages)
        chunk_id = str(uuid.uuid4())
        try:
            for chunk in self._client.models.generate_content_stream(
                model=request.model,
                contents=contents,
                config=self._make_full_config(request, system_instruction),
            ):
                text = getattr(chunk, "text", None) or ""
                if text:
                    finish_reason = None
                    if getattr(chunk, "candidates", None):
                        finish_reason = str(chunk.candidates[0].finish_reason)
                    yield StreamChunk(
                        id=chunk_id,
                        model=request.model,
                        provider=self.name,
                        delta=text,
                        finish_reason=finish_reason,
                    )
        except Exception as exc:  # noqa: BLE001
            self._handle_error(exc)

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        system_instruction, contents = self._to_genai_contents(request.messages)
        chunk_id = str(uuid.uuid4())
        try:
            async for chunk in await self._client.aio.models.generate_content_stream(
                model=request.model,
                contents=contents,
                config=self._make_full_config(request, system_instruction),
            ):
                text = getattr(chunk, "text", None) or ""
                if text:
                    finish_reason = None
                    if getattr(chunk, "candidates", None):
                        finish_reason = str(chunk.candidates[0].finish_reason)
                    yield StreamChunk(
                        id=chunk_id,
                        model=request.model,
                        provider=self.name,
                        delta=text,
                        finish_reason=finish_reason,
                    )
        except Exception as exc:  # noqa: BLE001
            self._handle_error(exc)
