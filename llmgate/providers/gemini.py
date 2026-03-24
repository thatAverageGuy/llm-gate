"""
llmgate.providers.gemini
~~~~~~~~~~~~~~~~~~~~~~~~
Google Gemini provider — wraps the official ``google-genai`` SDK (v1+).

Supported model prefixes: ``gemini-``

Uses the new ``google-genai`` package (google.genai), which replaced the
deprecated ``google-generativeai`` (google.generativeai).
"""
from __future__ import annotations

import os
import uuid
from typing import Any, ClassVar

from llmgate.base import BaseProvider
from llmgate.exceptions import AuthError, ProviderAPIError, RateLimitError
from llmgate.types import Choice, CompletionRequest, CompletionResponse, Message, TokenUsage


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

        Returns:
            (system_instruction, contents_list)
        """
        system_instruction: str | None = None
        contents: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg.content}]})

        return system_instruction, contents

    def _build_config(self, request: CompletionRequest) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if request.max_tokens is not None:
            config["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.top_p is not None:
            config["top_p"] = request.top_p
        config.update(request.extra_kwargs)
        return config

    def _map_response(self, raw: Any, model: str) -> CompletionResponse:
        text = raw.text if hasattr(raw, "text") else ""
        usage_meta = getattr(raw, "usage_metadata", None)
        usage = TokenUsage(
            prompt_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
            total_tokens=getattr(usage_meta, "total_token_count", 0) or 0,
        )
        finish_reason = None
        if raw.candidates:
            finish_reason = str(raw.candidates[0].finish_reason)

        return CompletionResponse(
            id=str(uuid.uuid4()),
            model=model,
            provider=self.name,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=text),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
            raw=raw,
        )

    def _handle_error(self, exc: Exception) -> None:
        msg = str(exc)
        err_type = type(exc).__name__
        # google.genai raises google.api_core.exceptions.*
        if err_type in ("Unauthenticated", "PermissionDenied"):
            raise AuthError(msg, provider=self.name) from exc
        if err_type == "ResourceExhausted":
            raise RateLimitError(msg, provider=self.name) from exc
        raise ProviderAPIError(msg, provider=self.name) from exc

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        system_instruction, contents = self._to_genai_contents(request.messages)
        config = self._build_config(request)

        try:
            raw = self._client.models.generate_content(
                model=request.model,
                contents=contents,
                config={
                    **config,
                    **({"system_instruction": system_instruction} if system_instruction else {}),
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._handle_error(exc)
        return self._map_response(raw, request.model)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        system_instruction, contents = self._to_genai_contents(request.messages)
        config = self._build_config(request)

        try:
            raw = await self._client.aio.models.generate_content(
                model=request.model,
                contents=contents,
                config={
                    **config,
                    **({"system_instruction": system_instruction} if system_instruction else {}),
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._handle_error(exc)
        return self._map_response(raw, request.model)
