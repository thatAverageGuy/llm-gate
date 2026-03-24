"""
llmgate.providers.groq
~~~~~~~~~~~~~~~~~~~~~~
Groq provider — wraps the official ``groq`` Python SDK (OpenAI-compatible).

**Routing**: Use the ``groq/`` prefix to explicitly target this provider:
    ``groq/llama-3.1-8b-instant``
    ``groq/mixtral-8x7b-32768``
    ``groq/gemma2-9b-it``

The prefix is stripped before the model name is sent to the Groq API.
Tool/function calling API is identical to OpenAI.
"""
from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, ClassVar, Iterator

from llmgate.base import BaseProvider
from llmgate.exceptions import AuthError, ProviderAPIError, RateLimitError
from llmgate.types import (
    Choice, CompletionRequest, CompletionResponse, Message, StreamChunk, ToolCall, TokenUsage,
)


class GroqProvider(BaseProvider):
    name: ClassVar[str] = "groq"
    supported_model_prefixes: ClassVar[tuple[str, ...]] = ("groq/",)

    def __init__(self, api_key: str | None = None, **client_kwargs: Any) -> None:
        try:
            import groq  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise ImportError("groq package is required: pip install groq") from e

        self._groq = groq
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise AuthError(
                "Groq API key not found. Set GROQ_API_KEY env var or pass api_key=...",
                provider=self.name,
            )
        self._client = groq.Groq(api_key=resolved_key, **client_kwargs)
        self._async_client = groq.AsyncGroq(api_key=resolved_key, **client_kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_params(self, request: CompletionRequest) -> dict[str, Any]:
        # Strip "groq/" routing prefix so the underlying model name is clean
        model = self._strip_prefix(request.model)
        params: dict[str, Any] = {
            "model": model,
            "messages": [m.to_dict() for m in request.messages],
            **request.extra_kwargs,
        }
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.function.name,
                        "description": t.function.description,
                        "parameters": t.function.parameters,
                    },
                }
                for t in request.tools
            ]
            if request.tool_choice is not None:
                params["tool_choice"] = request.tool_choice
        return params

    def _parse_tool_calls(self, raw_tool_calls: Any) -> list[ToolCall] | None:
        if not raw_tool_calls:
            return None
        result = []
        for tc in raw_tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (json.JSONDecodeError, AttributeError):
                args = {}
            result.append(ToolCall(id=tc.id, function=tc.function.name, arguments=args))
        return result or None

    def _map_response(self, raw: Any, model: str) -> CompletionResponse:
        choices = []
        for c in raw.choices:
            tool_calls = self._parse_tool_calls(getattr(c.message, "tool_calls", None))
            choices.append(Choice(
                index=c.index,
                message=Message(
                    role=c.message.role,
                    content=c.message.content or None,
                    tool_calls=tool_calls,
                ),
                finish_reason=c.finish_reason,
            ))
        usage = TokenUsage(
            prompt_tokens=raw.usage.prompt_tokens if raw.usage else 0,
            completion_tokens=raw.usage.completion_tokens if raw.usage else 0,
            total_tokens=raw.usage.total_tokens if raw.usage else 0,
        )
        return CompletionResponse(
            id=raw.id,
            model=model,
            provider=self.name,
            choices=choices,
            usage=usage,
            raw=raw,
        )

    def _handle_error(self, exc: Exception) -> None:
        status = getattr(exc, "status_code", None)
        msg = str(exc)
        err_type = type(exc).__name__
        if err_type == "AuthenticationError" or status == 401:
            raise AuthError(msg, provider=self.name, status_code=status) from exc
        if err_type == "RateLimitError" or status == 429:
            raise RateLimitError(msg, provider=self.name, status_code=status) from exc
        raise ProviderAPIError(msg, provider=self.name, status_code=status) from exc

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        try:
            raw = self._client.chat.completions.create(**self._build_params(request))
        except Exception as exc:  # noqa: BLE001
            self._handle_error(exc)
        return self._map_response(raw, request.model)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        try:
            raw = await self._async_client.chat.completions.create(
                **self._build_params(request)
            )
        except Exception as exc:  # noqa: BLE001
            self._handle_error(exc)
        return self._map_response(raw, request.model)

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        try:
            sdk_stream = self._client.chat.completions.create(
                **self._build_params(request), stream=True
            )
        except Exception as exc:  # noqa: BLE001
            self._handle_error(exc)
        for chunk in sdk_stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            finish_reason = chunk.choices[0].finish_reason
            if delta:
                yield StreamChunk(
                    id=chunk.id,
                    model=request.model,
                    provider=self.name,
                    delta=delta,
                    finish_reason=finish_reason,
                    index=chunk.choices[0].index,
                )

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        try:
            sdk_stream = await self._async_client.chat.completions.create(
                **self._build_params(request), stream=True
            )
        except Exception as exc:  # noqa: BLE001
            self._handle_error(exc)
        async for chunk in sdk_stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            finish_reason = chunk.choices[0].finish_reason
            if delta:
                yield StreamChunk(
                    id=chunk.id,
                    model=request.model,
                    provider=self.name,
                    delta=delta,
                    finish_reason=finish_reason,
                    index=chunk.choices[0].index,
                )
