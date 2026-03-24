"""
llmgate.providers.anthropic
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Anthropic provider — wraps the official ``anthropic`` Python SDK.

Supported model prefixes: ``claude-``

Anthropic's API has several peculiarities we handle here:
1. The ``system`` prompt is a top-level param, not a message.
2. ``max_tokens`` is *required* by the Anthropic API (we default to 1024).
3. Tools use ``input_schema`` instead of ``parameters``.
4. Tool calls come back as ``tool_use`` content blocks.
5. Tool results go back as special ``tool_result`` content in user messages.
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


class AnthropicProvider(BaseProvider):
    name: ClassVar[str] = "anthropic"
    supported_model_prefixes: ClassVar[tuple[str, ...]] = ("claude-",)

    def __init__(self, api_key: str | None = None, **client_kwargs: Any) -> None:
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise ImportError("anthropic package is required: pip install anthropic") from e

        self._anthropic = anthropic
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise AuthError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY env var or pass api_key=...",
                provider=self.name,
            )
        self._client = anthropic.Anthropic(api_key=resolved_key, **client_kwargs)
        self._async_client = anthropic.AsyncAnthropic(api_key=resolved_key, **client_kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Split system prompt and convert messages to Anthropic format.

        Handles:
          - role="tool"  → user message with tool_result content block
          - role="assistant" with tool_calls → assistant message with tool_use blocks
          - All others  → standard text messages
        """
        system_parts: list[str] = []
        result: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content or "")
                continue

            if msg.role == "tool":
                # Tool result — sent as a user message with tool_result content block
                result.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id or "",
                        "content": msg.content or "",
                    }],
                })
                continue

            if msg.role == "assistant" and msg.tool_calls:
                # Assistant requesting tool calls — build mixed content blocks
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function,
                        "input": tc.arguments,
                    })
                result.append({"role": "assistant", "content": content_blocks})
                continue

            # Standard user/assistant text message
            result.append({"role": msg.role, "content": msg.content or ""})

        system = "\n".join(system_parts) if system_parts else None
        return system, result

    def _build_params(self, request: CompletionRequest) -> dict[str, Any]:
        system, messages = self._build_messages(request.messages)
        params: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,
            **request.extra_kwargs,
        }
        if system:
            params["system"] = system
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.tools:
            params["tools"] = [
                {
                    "name": t.function.name,
                    "description": t.function.description,
                    "input_schema": t.function.parameters,
                }
                for t in request.tools
            ]
            if request.tool_choice is not None:
                # Normalise: "auto"→{"type":"auto"}, "none"→{"type":"none"}, dict pass-through
                if isinstance(request.tool_choice, str):
                    # "auto" | "none" | function name
                    if request.tool_choice in ("auto", "none"):
                        params["tool_choice"] = {"type": request.tool_choice}
                    else:
                        params["tool_choice"] = {"type": "tool", "name": request.tool_choice}
                else:
                    params["tool_choice"] = request.tool_choice
        return params

    def _map_response(self, raw: Any, model: str) -> CompletionResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in raw.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    function=block.name,
                    arguments=dict(block.input) if block.input else {},
                ))

        content = "\n".join(text_parts) if text_parts else None
        usage = TokenUsage(
            prompt_tokens=raw.usage.input_tokens if raw.usage else 0,
            completion_tokens=raw.usage.output_tokens if raw.usage else 0,
            total_tokens=(raw.usage.input_tokens + raw.usage.output_tokens) if raw.usage else 0,
        )
        return CompletionResponse(
            id=raw.id,
            model=model,
            provider=self.name,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls or None,
                    ),
                    finish_reason=raw.stop_reason,
                )
            ],
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
            raw = self._client.messages.create(**self._build_params(request))
        except Exception as exc:  # noqa: BLE001
            self._handle_error(exc)
        return self._map_response(raw, request.model)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        try:
            raw = await self._async_client.messages.create(**self._build_params(request))
        except Exception as exc:  # noqa: BLE001
            self._handle_error(exc)
        return self._map_response(raw, request.model)

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        chunk_id = str(uuid.uuid4())
        try:
            with self._client.messages.stream(**self._build_params(request)) as s:
                for text in s.text_stream:
                    yield StreamChunk(
                        id=chunk_id,
                        model=request.model,
                        provider=self.name,
                        delta=text,
                    )
        except Exception as exc:  # noqa: BLE001
            self._handle_error(exc)

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        chunk_id = str(uuid.uuid4())
        try:
            async with self._async_client.messages.stream(**self._build_params(request)) as s:
                async for text in s.text_stream:
                    yield StreamChunk(
                        id=chunk_id,
                        model=request.model,
                        provider=self.name,
                        delta=text,
                    )
        except Exception as exc:  # noqa: BLE001
            self._handle_error(exc)
