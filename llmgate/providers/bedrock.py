"""
llmgate.providers.bedrock
~~~~~~~~~~~~~~~~~~~~~~~~~
AWS Bedrock provider — uses ``boto3`` with the ``bedrock-runtime`` Converse API.

The Converse API is a unified interface that works across all Bedrock models
(Claude, Llama, Titan, Mistral, Command, etc.) with a consistent format.

**Routing**: Use the ``bedrock/`` prefix with the full Bedrock model ID:
    ``bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0``
    ``bedrock/meta.llama3-8b-instruct-v1:0``
    ``bedrock/mistral.mistral-large-2402-v1:0``

**Install**: ``pip install llmgate[bedrock]``

**Auth**: Standard AWS credentials:
    ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, ``AWS_DEFAULT_REGION``
    — or configure via ``~/.aws/credentials`` / IAM role.
"""
from __future__ import annotations

import os
from typing import Any, AsyncIterator, ClassVar, Iterator

from llmgate.base import BaseProvider
from llmgate.exceptions import AuthError, ProviderAPIError, RateLimitError
from llmgate.types import (
    Choice, CompletionRequest, CompletionResponse, Message,
    StreamChunk, ToolCall, TokenUsage,
)


def _to_bedrock_messages(messages: list[Message]) -> tuple[list[dict], str | None]:
    """Convert llmgate messages to Bedrock Converse format.
    Returns (messages, system_prompt).
    """
    system_prompt: str | None = None
    bedrock_msgs: list[dict] = []

    for m in messages:
        if m.role == "system":
            system_prompt = m.content
            continue

        if m.role == "user":
            bedrock_msgs.append({
                "role": "user",
                "content": [{"text": m.content or ""}],
            })
        elif m.role == "assistant":
            content: list[dict] = []
            if m.content:
                content.append({"text": m.content})
            if m.tool_calls:
                for tc in m.tool_calls:
                    content.append({
                        "toolUse": {
                            "toolUseId": tc.id,
                            "name": tc.function,
                            "input": tc.arguments or {},
                        }
                    })
            bedrock_msgs.append({"role": "assistant", "content": content})
        elif m.role == "tool":
            bedrock_msgs.append({
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": m.tool_call_id or "",
                        "content": [{"text": m.content or ""}],
                        "status": "success",
                    }
                }],
            })

    return bedrock_msgs, system_prompt


class BedrockProvider(BaseProvider):
    name: ClassVar[str] = "bedrock"
    supported_model_prefixes: ClassVar[tuple[str, ...]] = ("bedrock/",)

    def __init__(
        self,
        region: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        api_key: str | None = None,  # ignored; AWS uses access keys
        **client_kwargs: Any,
    ) -> None:
        try:
            import boto3  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "boto3 package is required: pip install llmgate[bedrock]"
            ) from e

        resolved_region = (
            region
            or os.environ.get("AWS_DEFAULT_REGION")
            or os.environ.get("AWS_REGION")
            or "us-east-1"
        )
        session_kwargs: dict[str, Any] = {}
        if aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID"):
            session_kwargs["aws_access_key_id"] = aws_access_key_id or os.environ["AWS_ACCESS_KEY_ID"]
            session_kwargs["aws_secret_access_key"] = (
                aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
            )
            if aws_session_token or os.environ.get("AWS_SESSION_TOKEN"):
                session_kwargs["aws_session_token"] = (
                    aws_session_token or os.environ.get("AWS_SESSION_TOKEN")
                )

        self._client = boto3.client(
            "bedrock-runtime",
            region_name=resolved_region,
            **session_kwargs,
            **client_kwargs,
        )
        # boto3 is sync only; async calls run in an executor
        self._boto3 = boto3
        self._region = resolved_region

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_converse_params(self, request: CompletionRequest) -> dict[str, Any]:
        model_id = self._strip_prefix(request.model)
        messages = request.messages
        if request.response_format is not None:
            from llmgate.structured import inject_schema_prompt  # noqa: PLC0415
            messages = inject_schema_prompt(messages, request.response_format)
        bedrock_messages, system_prompt = _to_bedrock_messages(messages)

        params: dict[str, Any] = {
            "modelId": model_id,
            "messages": bedrock_messages,
        }
        if system_prompt:
            params["system"] = [{"text": system_prompt}]

        inference: dict[str, Any] = {}
        if request.max_tokens is not None:
            inference["maxTokens"] = request.max_tokens
        if request.temperature is not None:
            inference["temperature"] = request.temperature
        if request.top_p is not None:
            inference["topP"] = request.top_p
        if inference:
            params["inferenceConfig"] = inference

        if request.tools:
            params["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": t.function.name,
                            "description": t.function.description or "",
                            "inputSchema": {
                                "json": t.function.parameters or {},
                            },
                        }
                    }
                    for t in request.tools
                ]
            }
            if request.tool_choice and request.tool_choice != "auto":
                params["toolConfig"]["toolChoice"] = {"any": {}}

        return params

    def _map_response(self, raw: Any, model: str, response_format: Any = None) -> CompletionResponse:
        output = raw.get("output", {}).get("message", {})
        content_blocks = output.get("content", [])
        stop_reason = raw.get("stopReason", "end_turn")

        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []

        for block in content_blocks:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(ToolCall(
                    id=tu.get("toolUseId", ""),
                    function=tu.get("name", ""),
                    arguments=tu.get("input", {}),
                ))

        text = "".join(text_parts) or None
        finish_reason = "tool_calls" if tool_calls else stop_reason

        usage_raw = raw.get("usage", {})
        parsed = None
        if response_format is not None and text:
            from llmgate.structured import validate_parsed  # noqa: PLC0415
            parsed = validate_parsed(text, response_format)
        return CompletionResponse(
            id=raw.get("ResponseMetadata", {}).get("RequestId", ""),
            model=model,
            provider=self.name,
            choices=[Choice(
                index=0,
                message=Message(
                    role="assistant",
                    content=text,
                    tool_calls=tool_calls or None,
                ),
                finish_reason=finish_reason,
            )],
            usage=TokenUsage(
                prompt_tokens=usage_raw.get("inputTokens", 0),
                completion_tokens=usage_raw.get("outputTokens", 0),
                total_tokens=usage_raw.get("totalTokens", 0),
            ),
            raw=raw,
            parsed=parsed,
        )

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        params = self._build_converse_params(request)
        try:
            raw = self._client.converse(**params)
        except Exception as exc:
            self._wrap_exception(exc)
        return self._map_response(raw, request.model, request.response_format)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        import asyncio  # noqa: PLC0415
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.complete, request)

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        params = self._build_converse_params(request)
        try:
            response = self._client.converse_stream(**params)
            for event in response.get("stream", []):
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    if "text" in delta:
                        yield StreamChunk(delta=delta["text"])
        except Exception as exc:
            self._wrap_exception(exc)

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        # boto3 is sync; collect all chunks in executor and yield
        import asyncio  # noqa: PLC0415
        chunks: list[StreamChunk] = await asyncio.get_event_loop().run_in_executor(
            None, lambda: list(self.stream(request))
        )
        for chunk in chunks:
            yield chunk

    def _wrap_exception(self, exc: Exception) -> None:
        msg = str(exc)
        error_code = getattr(getattr(exc, "response", {}), "get", lambda k, d=None: d)(
            "Error", {}
        ).get("Code", "")
        if "ThrottlingException" in type(exc).__name__ or error_code == "ThrottlingException":
            raise RateLimitError(msg, provider=self.name) from exc
        if "AccessDenied" in type(exc).__name__ or "UnauthorizedClient" in msg:
            raise AuthError(msg, provider=self.name) from exc
        raise ProviderAPIError(msg, provider=self.name) from exc
