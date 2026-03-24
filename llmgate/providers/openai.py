"""
llmgate.providers.openai
~~~~~~~~~~~~~~~~~~~~~~~~
OpenAI provider — wraps the official ``openai`` Python SDK.

Supported model prefixes: ``gpt-``, ``o1-``, ``o3-``, ``chatgpt-``
"""
from __future__ import annotations

import os
from typing import Any, AsyncIterator, ClassVar, Iterator

from llmgate.base import BaseProvider
from llmgate.exceptions import AuthError, ProviderAPIError, RateLimitError
from llmgate.types import (
    Choice, CompletionRequest, CompletionResponse, Message, StreamChunk, TokenUsage,
)


class OpenAIProvider(BaseProvider):
    name: ClassVar[str] = "openai"
    supported_model_prefixes: ClassVar[tuple[str, ...]] = ("gpt-", "o1-", "o3-", "chatgpt-")

    def __init__(self, api_key: str | None = None, **client_kwargs: Any) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise ImportError("openai package is required: pip install openai") from e

        self._openai = openai
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise AuthError(
                "OpenAI API key not found. Set OPENAI_API_KEY env var or pass api_key=...",
                provider=self.name,
            )
        self._client = openai.OpenAI(api_key=resolved_key, **client_kwargs)
        self._async_client = openai.AsyncOpenAI(api_key=resolved_key, **client_kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_params(self, request: CompletionRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": request.model,
            "messages": [m.to_dict() for m in request.messages],
            **request.extra_kwargs,
        }
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.top_p is not None:
            params["top_p"] = request.top_p
        return params

    def _map_response(self, raw: Any, model: str) -> CompletionResponse:
        choices = [
            Choice(
                index=c.index,
                message=Message(role=c.message.role, content=c.message.content or ""),
                finish_reason=c.finish_reason,
            )
            for c in raw.choices
        ]
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

    def _handle_openai_error(self, exc: Exception) -> None:
        """Re-raise openai SDK errors as llmgate exceptions."""
        err_type = type(exc).__name__
        status = getattr(exc, "status_code", None)
        msg = str(exc)
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
            self._handle_openai_error(exc)
        return self._map_response(raw, request.model)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        try:
            raw = await self._async_client.chat.completions.create(
                **self._build_params(request)
            )
        except Exception as exc:  # noqa: BLE001
            self._handle_openai_error(exc)
        return self._map_response(raw, request.model)

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        try:
            sdk_stream = self._client.chat.completions.create(
                **self._build_params(request), stream=True
            )
        except Exception as exc:  # noqa: BLE001
            self._handle_openai_error(exc)
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
            self._handle_openai_error(exc)
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
