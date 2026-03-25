"""
llmgate.providers.azure
~~~~~~~~~~~~~~~~~~~~~~~
Azure OpenAI provider — uses the standard ``openai`` SDK with ``AzureOpenAI`` client.

**Routing**: Use the ``azure/`` prefix:
    ``azure/<deployment-name>``

The deployment name after the prefix is sent directly to Azure (no further stripping).

**No extra install required** — uses the ``openai`` package already bundled.

**Env vars**:
    ``AZURE_OPENAI_API_KEY``
    ``AZURE_OPENAI_ENDPOINT``     — e.g. ``https://<name>.openai.azure.com``
    ``AZURE_OPENAI_API_VERSION``  — defaults to ``2024-02-01``

Or pass ``api_key``, ``azure_endpoint``, ``api_version`` kwargs directly.
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


class AzureOpenAIProvider(BaseProvider):
    name: ClassVar[str] = "azure"
    supported_model_prefixes: ClassVar[tuple[str, ...]] = ("azure/",)

    def __init__(
        self,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        **client_kwargs: Any,
    ) -> None:
        try:
            import openai  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "openai package is required: pip install openai"
            ) from e

        resolved_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not resolved_key:
            raise AuthError(
                "Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY env var "
                "or pass api_key=...",
                provider=self.name,
            )
        if not endpoint:
            raise AuthError(
                "Azure OpenAI endpoint not found. Set AZURE_OPENAI_ENDPOINT env var "
                "or pass azure_endpoint=...",
                provider=self.name,
            )

        self._openai = openai
        self._client = openai.AzureOpenAI(
            api_key=resolved_key,
            azure_endpoint=endpoint,
            api_version=version,
            **client_kwargs,
        )
        self._async_client = openai.AsyncAzureOpenAI(
            api_key=resolved_key,
            azure_endpoint=endpoint,
            api_version=version,
            **client_kwargs,
        )

    # ------------------------------------------------------------------
    # Helpers (identical to OpenAI provider)
    # ------------------------------------------------------------------

    def _build_params(self, request: CompletionRequest) -> dict[str, Any]:
        # Deployment name is everything after "azure/"
        deployment = self._strip_prefix(request.model)
        params: dict[str, Any] = {
            "model": deployment,
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
            if request.tool_choice is not None:
                params["tool_choice"] = request.tool_choice
        if request.response_format is not None:
            from llmgate.structured import get_json_schema  # noqa: PLC0415
            schema = get_json_schema(request.response_format)
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.response_format.__name__.lower(),
                    "strict": True,
                    "schema": schema,
                },
            }
        return params

    def _parse_tool_calls(self, raw_tool_calls: Any) -> list[ToolCall]:
        tool_calls = []
        for tc in raw_tool_calls:
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
        return tool_calls

    def _map_response(self, raw: Any, model: str, response_format: Any = None) -> CompletionResponse:
        choices = []
        for c in raw.choices:
            msg = c.message
            tool_calls = self._parse_tool_calls(msg.tool_calls) if msg.tool_calls else []
            choices.append(Choice(
                index=c.index,
                message=Message(
                    role=msg.role,
                    content=msg.content,
                    tool_calls=tool_calls or None,
                ),
                finish_reason=c.finish_reason or "stop",
            ))
        u = raw.usage
        parsed = None
        if response_format is not None and choices:
            from llmgate.structured import validate_parsed  # noqa: PLC0415
            parsed = validate_parsed(choices[0].message.content, response_format)
        return CompletionResponse(
            id=raw.id,
            model=model,
            provider=self.name,
            choices=choices,
            usage=TokenUsage(
                prompt_tokens=u.prompt_tokens,
                completion_tokens=u.completion_tokens,
                total_tokens=u.total_tokens,
            ),
            raw=raw,
            parsed=parsed,
        )

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        params = self._build_params(request)
        try:
            raw = self._client.chat.completions.create(**params)
        except Exception as exc:
            self._wrap_exception(exc)
        return self._map_response(raw, request.model, request.response_format)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        params = self._build_params(request)
        try:
            raw = await self._async_client.chat.completions.create(**params)
        except Exception as exc:
            self._wrap_exception(exc)
        return self._map_response(raw, request.model, request.response_format)

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        params = self._build_params(request)
        try:
            with self._client.chat.completions.create(stream=True, **params) as streamer:
                for chunk in streamer:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield StreamChunk(delta=chunk.choices[0].delta.content)
        except Exception as exc:
            self._wrap_exception(exc)

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        params = self._build_params(request)
        try:
            async with await self._async_client.chat.completions.create(
                stream=True, **params
            ) as streamer:
                async for chunk in streamer:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield StreamChunk(delta=chunk.choices[0].delta.content)
        except Exception as exc:
            self._wrap_exception(exc)

    def _wrap_exception(self, exc: Exception) -> None:
        msg = str(exc)
        if hasattr(self, "_openai"):
            if isinstance(exc, self._openai.AuthenticationError):
                raise AuthError(msg, provider=self.name) from exc
            if isinstance(exc, self._openai.RateLimitError):
                raise RateLimitError(msg, provider=self.name) from exc
        raise ProviderAPIError(msg, provider=self.name) from exc
