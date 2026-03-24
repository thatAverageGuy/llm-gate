"""
llmgate.providers.ollama
~~~~~~~~~~~~~~~~~~~~~~~~
Ollama provider — local model inference via the ``ollama`` Python SDK.

**Routing**: Use the ``ollama/`` prefix:
    ``ollama/llama3.2``
    ``ollama/mistral``
    ``ollama/qwen2.5-coder``
    ``ollama/phi4``

No API key required.  Ollama must be running locally (or at a custom host).

**Install**: ``pip install llmgate[ollama]``

**Env vars**:
    ``OLLAMA_HOST``  — defaults to ``http://localhost:11434``
"""
from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, ClassVar, Iterator

from llmgate.base import BaseProvider
from llmgate.exceptions import ProviderAPIError
from llmgate.types import (
    Choice, CompletionRequest, CompletionResponse, Message,
    StreamChunk, ToolCall, TokenUsage,
)


class OllamaProvider(BaseProvider):
    name: ClassVar[str] = "ollama"
    supported_model_prefixes: ClassVar[tuple[str, ...]] = ("ollama/",)

    def __init__(
        self,
        host: str | None = None,
        api_key: str | None = None,  # ignored — Ollama needs no key
        **client_kwargs: Any,
    ) -> None:
        try:
            import ollama  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "ollama package is required: pip install llmgate[ollama]"
            ) from e

        resolved_host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._ollama = ollama
        self._client = ollama.Client(host=resolved_host, **client_kwargs)
        self._async_client = ollama.AsyncClient(host=resolved_host, **client_kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_params(self, request: CompletionRequest) -> dict[str, Any]:
        model = self._strip_prefix(request.model)
        messages = [m.to_dict() for m in request.messages]

        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **request.extra_kwargs,
        }
        # Options dict for generation params
        options: dict[str, Any] = {}
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if options:
            params["options"] = options

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

        return params

    def _map_response(self, raw: Any, model: str) -> CompletionResponse:
        msg = raw.message
        tool_calls: list[ToolCall] = []

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                fn = tc.function
                args = fn.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                tool_calls.append(ToolCall(
                    id=f"{fn.name}-{id(tc)}",  # Ollama doesn't provide IDs
                    function=fn.name,
                    arguments=args,
                ))

        finish_reason = "tool_calls" if tool_calls else (raw.done_reason or "stop")
        return CompletionResponse(
            id=f"ollama-{id(raw)}",
            model=model,
            provider=self.name,
            choices=[Choice(
                index=0,
                message=Message(
                    role=msg.role,
                    content=msg.content if not tool_calls else None,
                    tool_calls=tool_calls or None,
                ),
                finish_reason=finish_reason,
            )],
            usage=TokenUsage(
                prompt_tokens=getattr(raw, "prompt_eval_count", 0) or 0,
                completion_tokens=getattr(raw, "eval_count", 0) or 0,
                total_tokens=(getattr(raw, "prompt_eval_count", 0) or 0)
                             + (getattr(raw, "eval_count", 0) or 0),
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
            raise ProviderAPIError(
                f"Ollama error: {exc}. Is Ollama running at the configured host?",
                provider=self.name,
            ) from exc
        return self._map_response(raw, request.model)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        params = self._build_params(request)
        try:
            raw = await self._async_client.chat(**params)
        except Exception as exc:
            raise ProviderAPIError(
                f"Ollama error: {exc}. Is Ollama running at the configured host?",
                provider=self.name,
            ) from exc
        return self._map_response(raw, request.model)

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        params = self._build_params(request)
        try:
            for chunk in self._client.chat(stream=True, **params):
                delta = getattr(chunk.message, "content", None)
                if delta:
                    yield StreamChunk(delta=delta)
        except Exception as exc:
            raise ProviderAPIError(
                f"Ollama stream error: {exc}",
                provider=self.name,
            ) from exc

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        params = self._build_params(request)
        try:
            async for chunk in await self._async_client.chat(stream=True, **params):
                delta = getattr(chunk.message, "content", None)
                if delta:
                    yield StreamChunk(delta=delta)
        except Exception as exc:
            raise ProviderAPIError(
                f"Ollama async stream error: {exc}",
                provider=self.name,
            ) from exc
