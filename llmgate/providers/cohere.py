"""
llmgate.providers.cohere
~~~~~~~~~~~~~~~~~~~~~~~~
Cohere provider — wraps the official ``cohere`` Python SDK.

**Routing**: Use the ``cohere/`` prefix:
    ``cohere/command-r-plus``
    ``cohere/command-r``
    ``cohere/command``

Cohere uses a different chat history format (``chat_history`` + ``message``)
and its own tool calling schema. This provider handles the mapping.

**Install**: ``pip install llmgate[cohere]``
**Env var**: ``COHERE_API_KEY``
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


def _to_cohere_messages(
    messages: list[Message],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Cohere expects a separate ``message`` (current user turn) and
    ``chat_history`` (prior turns). System messages become a preamble
    prepended to the first user message.

    Returns (message, chat_history).
    """
    preamble_parts: list[str] = []
    history: list[dict[str, Any]] = []
    current_message = ""

    # Extract system messages first
    non_system = []
    for m in messages:
        if m.role == "system":
            preamble_parts.append(m.content or "")
        else:
            non_system.append(m)

    for i, m in enumerate(non_system):
        if i == len(non_system) - 1 and m.role == "user":
            current_message = m.content or ""
        elif m.role == "user":
            history.append({"role": "USER", "message": m.content or ""})
        elif m.role == "assistant":
            if m.tool_calls:
                # Assistant turn with tool calls — add as CHATBOT turn
                history.append({
                    "role": "CHATBOT",
                    "message": m.content or "",
                    "tool_calls": [
                        {
                            "name": tc.function,
                            "parameters": tc.arguments or {},
                        }
                        for tc in m.tool_calls
                    ],
                })
            else:
                history.append({"role": "CHATBOT", "message": m.content or ""})
        elif m.role == "tool":
            # Tool results follow CHATBOT turns with tool_calls
            history.append({
                "role": "TOOL",
                "tool_results": [
                    {
                        "call": {"name": m.name or "", "parameters": {}},
                        "outputs": [{"result": m.content or ""}],
                    }
                ],
            })

    return current_message, history, "\n\n".join(preamble_parts) if preamble_parts else None


class CohereProvider(BaseProvider):
    name: ClassVar[str] = "cohere"
    supported_model_prefixes: ClassVar[tuple[str, ...]] = ("cohere/",)

    def __init__(self, api_key: str | None = None, **client_kwargs: Any) -> None:
        try:
            import cohere  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "cohere package is required: pip install llmgate[cohere]"
            ) from e

        resolved_key = api_key or os.environ.get("COHERE_API_KEY")
        if not resolved_key:
            raise AuthError(
                "Cohere API key not found. Set COHERE_API_KEY env var or pass api_key=...",
                provider=self.name,
            )
        self._cohere = cohere
        self._client = cohere.ClientV2(api_key=resolved_key, **client_kwargs)
        self._async_client = cohere.AsyncClientV2(api_key=resolved_key, **client_kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_params(self, request: CompletionRequest) -> dict[str, Any]:
        model = self._strip_prefix(request.model)
        # For ClientV2 we use the messages array directly (OpenAI-compatible)
        msgs = []
        for m in request.messages:
            if m.role == "tool":
                msgs.append({
                    "role": "tool",
                    "tool_call_id": m.tool_call_id or "",
                    "content": m.content or "",
                })
            elif m.role == "assistant" and m.tool_calls:
                msgs.append({
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function,
                                "arguments": json.dumps(tc.arguments or {}),
                            },
                        }
                        for tc in m.tool_calls
                    ],
                    "content": m.content or "",
                })
            else:
                msgs.append({"role": m.role, "content": m.content or ""})

        params: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            **request.extra_kwargs,
        }
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.top_p is not None:
            params["p"] = request.top_p
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
                # Cohere V2 supports force_single_step: bool
                params["tool_choice"] = request.tool_choice
        return params

    def _map_response(self, raw: Any, model: str) -> CompletionResponse:
        choices = []
        for c in (raw.message.content if hasattr(raw, "message") else []):
            pass  # handled below
        
        # Cohere V2 response structure
        resp_msg = raw.message
        tool_calls: list[ToolCall] = []
        text_content: str | None = None

        if hasattr(resp_msg, "tool_calls") and resp_msg.tool_calls:
            for tc in resp_msg.tool_calls:
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
        if hasattr(resp_msg, "content") and resp_msg.content:
            for block in resp_msg.content:
                if hasattr(block, "text"):
                    text_content = block.text
                    break

        finish_reason = "tool_calls" if tool_calls else "stop"
        choices = [Choice(
            index=0,
            message=Message(
                role="assistant",
                content=text_content,
                tool_calls=tool_calls or None,
            ),
            finish_reason=finish_reason,
        )]

        usage = raw.usage if hasattr(raw, "usage") else None
        billed = getattr(usage, "billed_units", None) if usage else None
        return CompletionResponse(
            id=getattr(raw, "id", ""),
            model=model,
            provider=self.name,
            choices=choices,
            usage=TokenUsage(
                prompt_tokens=getattr(billed, "input_tokens", 0) if billed else 0,
                completion_tokens=getattr(billed, "output_tokens", 0) if billed else 0,
                total_tokens=(getattr(billed, "input_tokens", 0) + getattr(billed, "output_tokens", 0)) if billed else 0,
            ),
            raw=raw,
        )

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        params = self._build_params(request)
        try:
            raw = self._client.chat(**params)
        except Exception as exc:
            self._wrap_exception(exc)
        return self._map_response(raw, request.model)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        params = self._build_params(request)
        try:
            raw = await self._async_client.chat(**params)
        except Exception as exc:
            self._wrap_exception(exc)
        return self._map_response(raw, request.model)

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        params = self._build_params(request)
        try:
            for event in self._client.chat_stream(**params):
                if hasattr(event, "type") and event.type == "content-delta":
                    text = getattr(event.delta, "message", None)
                    if text and hasattr(text, "content"):
                        for block in (text.content or []):
                            if hasattr(block, "text") and block.text:
                                yield StreamChunk(delta=block.text)
        except Exception as exc:
            self._wrap_exception(exc)

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        params = self._build_params(request)
        try:
            async for event in self._async_client.chat_stream(**params):
                if hasattr(event, "type") and event.type == "content-delta":
                    text = getattr(event.delta, "message", None)
                    if text and hasattr(text, "content"):
                        for block in (text.content or []):
                            if hasattr(block, "text") and block.text:
                                yield StreamChunk(delta=block.text)
        except Exception as exc:
            self._wrap_exception(exc)

    def _wrap_exception(self, exc: Exception) -> None:
        msg = str(exc)
        exc_type = type(exc).__name__
        if "401" in msg or "Unauthorized" in exc_type or "InvalidTokenError" in exc_type:
            raise AuthError(msg, provider=self.name) from exc
        if "429" in msg or "TooManyRequests" in exc_type:
            raise RateLimitError(msg, provider=self.name) from exc
        raise ProviderAPIError(msg, provider=self.name) from exc
