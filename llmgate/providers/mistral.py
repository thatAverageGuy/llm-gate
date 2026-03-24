"""
llmgate.providers.mistral
~~~~~~~~~~~~~~~~~~~~~~~~~
Mistral AI provider — wraps the official ``mistralai`` Python SDK.

**Routing**: Use the ``mistral/`` prefix:
    ``mistral/mistral-large-latest``
    ``mistral/mistral-small-latest``
    ``mistral/open-mistral-7b``

Tool/function calling API compatible with OpenAI format.

**Install**: ``pip install llmgate[mistral]``
**Env var**: ``MISTRAL_API_KEY``
"""
from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, ClassVar, Iterator

from llmgate.base import BaseProvider
from llmgate.exceptions import AuthError, ProviderAPIError, RateLimitError
from llmgate.types import (
    Choice, CompletionRequest, CompletionResponse, Message,
    StreamChunk, ToolCall, TokenUsage,
)


class MistralProvider(BaseProvider):
    name: ClassVar[str] = "mistral"
    supported_model_prefixes: ClassVar[tuple[str, ...]] = ("mistral/",)

    def __init__(self, api_key: str | None = None, **client_kwargs: Any) -> None:
        try:
            from mistralai import Mistral  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "mistralai package is required: pip install llmgate[mistral]"
            ) from e

        resolved_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not resolved_key:
            raise AuthError(
                "Mistral API key not found. Set MISTRAL_API_KEY env var or pass api_key=...",
                provider=self.name,
            )
        self._client = Mistral(api_key=resolved_key, **client_kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_params(self, request: CompletionRequest) -> dict[str, Any]:
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
                        "description": t.function.description or "",
                        "parameters": t.function.parameters or {},
                    },
                }
                for t in request.tools
            ]
            if request.tool_choice:
                params["tool_choice"] = request.tool_choice
        return params

    def _map_response(self, raw: Any, model: str) -> CompletionResponse:
        choices = []
        for c in raw.choices:
            msg = c.message
            tool_calls: list[ToolCall] = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        function=tc.function.name,
                        arguments=args,
                    ))
            choices.append(Choice(
                index=c.index,
                message=Message(
                    role=msg.role,
                    content=msg.content,
                    tool_calls=tool_calls or None,
                ),
                finish_reason=c.finish_reason or "stop",
            ))
        usage = raw.usage
        return CompletionResponse(
            id=raw.id,
            model=model,
            provider=self.name,
            choices=choices,
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            raw=raw,
        )

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        params = self._build_params(request)
        try:
            raw = self._client.chat.complete(**params)
        except Exception as exc:
            self._wrap_exception(exc)
        return self._map_response(raw, request.model)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        params = self._build_params(request)
        try:
            raw = await self._client.chat.complete_async(**params)
        except Exception as exc:
            self._wrap_exception(exc)
        return self._map_response(raw, request.model)

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        params = self._build_params(request)
        try:
            with self._client.chat.stream(**params) as streamer:
                for event in streamer:
                    delta = event.data.choices[0].delta if event.data.choices else None
                    if delta and delta.content:
                        yield StreamChunk(delta=delta.content)
        except Exception as exc:
            self._wrap_exception(exc)

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        params = self._build_params(request)
        try:
            async with self._client.chat.stream_async(**params) as streamer:
                async for event in streamer:
                    delta = event.data.choices[0].delta if event.data.choices else None
                    if delta and delta.content:
                        yield StreamChunk(delta=delta.content)
        except Exception as exc:
            self._wrap_exception(exc)

    def _wrap_exception(self, exc: Exception) -> None:
        msg = str(exc)
        exc_type = type(exc).__name__
        if "401" in msg or "Unauthorized" in exc_type:
            raise AuthError(msg, provider=self.name) from exc
        if "429" in msg or "RateLimit" in exc_type:
            raise RateLimitError(msg, provider=self.name) from exc
        raise ProviderAPIError(msg, provider=self.name) from exc
