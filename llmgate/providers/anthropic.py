"""
llmgate.providers.anthropic
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Anthropic provider — wraps the official ``anthropic`` Python SDK.

Supported model prefixes: ``claude-``

Anthropic's API has two peculiarities we handle here:
1. The ``system`` prompt is a top-level param, not a message.
2. ``max_tokens`` is *required* by the Anthropic API (we default to 1024).
"""
from __future__ import annotations

import os
from typing import Any, ClassVar

from llmgate.base import BaseProvider
from llmgate.exceptions import AuthError, ProviderAPIError, RateLimitError
from llmgate.types import Choice, CompletionRequest, CompletionResponse, Message, TokenUsage


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

    def _build_params(self, request: CompletionRequest) -> dict[str, Any]:
        """Split out system messages and build Anthropic-compatible params."""
        system_parts = [m.content for m in request.messages if m.role == "system"]
        non_system = [m.to_dict() for m in request.messages if m.role != "system"]

        params: dict[str, Any] = {
            "model": request.model,
            "messages": non_system,
            "max_tokens": request.max_tokens or 1024,
            **request.extra_kwargs,
        }
        if system_parts:
            params["system"] = "\n".join(system_parts)
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.top_p is not None:
            params["top_p"] = request.top_p
        return params

    def _map_response(self, raw: Any, model: str) -> CompletionResponse:
        choices = [
            Choice(
                index=i,
                message=Message(role="assistant", content=block.text),
                finish_reason=raw.stop_reason,
            )
            for i, block in enumerate(raw.content)
            if hasattr(block, "text")
        ]
        usage = TokenUsage(
            prompt_tokens=raw.usage.input_tokens if raw.usage else 0,
            completion_tokens=raw.usage.output_tokens if raw.usage else 0,
            total_tokens=(raw.usage.input_tokens + raw.usage.output_tokens) if raw.usage else 0,
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
