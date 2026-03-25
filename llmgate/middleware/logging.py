"""
llmgate.middleware.logging
~~~~~~~~~~~~~~~~~~~~~~~~~~
Structured request/response logging middleware.

Uses stdlib ``logging`` with logger ``"llmgate"``.

Usage::

    from llmgate.middleware import LoggingMiddleware

    gate = LLMGate(middleware=[LoggingMiddleware(level="DEBUG")])
"""
from __future__ import annotations

import logging
import time
from typing import Any

from llmgate.middleware.base import AsyncNext, BaseMiddleware, SyncNext
from llmgate.types import CompletionRequest, CompletionResponse

logger = logging.getLogger("llmgate")


class LoggingMiddleware(BaseMiddleware):
    """
    Log each LLM request and its response.

    Args:
        level:         Log level string (``"DEBUG"``, ``"INFO"``, etc.).
        mask_content:  If True, message content is not logged (for PII).
        log_usage:     If True, include token usage in the response log.
    """

    def __init__(
        self,
        level: str = "INFO",
        mask_content: bool = False,
        log_usage: bool = True,
    ) -> None:
        self._level = getattr(logging, level.upper(), logging.INFO)
        self._mask_content = mask_content
        self._log_usage = log_usage

    def _request_extra(self, request: CompletionRequest) -> dict[str, Any]:
        msgs = request.messages
        return {
            "model": request.model,
            "message_count": len(msgs),
            "has_tools": bool(request.tools),
            "stream": request.stream,
            "messages": (
                "[masked]" if self._mask_content
                else [{"role": m.role, "content": m.content} for m in msgs]
            ),
        }

    def _response_extra(
        self, request: CompletionRequest, resp: CompletionResponse, elapsed: float
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {
            "model": request.model,
            "provider": resp.provider,
            "finish_reason": resp.choices[0].finish_reason if resp.choices else None,
            "latency_s": round(elapsed, 3),
            "has_tool_calls": bool(resp.tool_calls),
        }
        if self._log_usage:
            extra["usage"] = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return extra

    def handle(
        self,
        request: CompletionRequest,
        call_next: SyncNext,
    ) -> CompletionResponse:
        logger.log(self._level, "llmgate request", extra=self._request_extra(request))
        t0 = time.perf_counter()
        try:
            resp = call_next(request)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error(
                "llmgate error",
                extra={"model": request.model, "latency_s": round(elapsed, 3),
                       "error": str(exc), "error_type": type(exc).__name__},
                exc_info=True,
            )
            raise
        elapsed = time.perf_counter() - t0
        logger.log(
            self._level, "llmgate response",
            extra=self._response_extra(request, resp, elapsed),
        )
        return resp

    async def ahandle(
        self,
        request: CompletionRequest,
        call_next: AsyncNext,
    ) -> CompletionResponse:
        import time as _time  # noqa: PLC0415
        logger.log(self._level, "llmgate request", extra=self._request_extra(request))
        t0 = _time.perf_counter()
        try:
            resp = await call_next(request)
        except Exception as exc:
            elapsed = _time.perf_counter() - t0
            logger.error(
                "llmgate error",
                extra={"model": request.model, "latency_s": round(elapsed, 3),
                       "error": str(exc), "error_type": type(exc).__name__},
                exc_info=True,
            )
            raise
        elapsed = _time.perf_counter() - t0
        logger.log(
            self._level, "llmgate response",
            extra=self._response_extra(request, resp, elapsed),
        )
        return resp
