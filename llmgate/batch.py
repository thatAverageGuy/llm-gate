"""
llmgate.batch
~~~~~~~~~~~~~
Parallel batch completion execution.

Public API:
    batch(requests, *, max_concurrency, fail_fast, middleware, timeout) -> BatchResult
    abatch(requests, *, max_concurrency, fail_fast, middleware, timeout) -> BatchResult

Both functions execute multiple :class:`~llmgate.types.CompletionRequest` objects
in parallel and return a :class:`~llmgate.types.BatchResult` that aggregates
responses, errors, and token usage.

Concurrency strategy:
- ``abatch`` uses ``asyncio.Semaphore`` to cap concurrent async coroutines.
- ``batch``  uses ``concurrent.futures.ThreadPoolExecutor`` to cap concurrent
  threads, each calling the synchronous provider path directly.  This avoids
  the ``asyncio.run()``-in-running-loop pitfall that would occur if ``batch``
  simply wrapped ``abatch``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llmgate.middleware.base import BaseMiddleware

from llmgate.completion import acompletion, completion
from llmgate.exceptions import BatchTimeoutError
from llmgate.types import BatchError, BatchResult, CompletionRequest, Message


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_requests(
    requests: list[CompletionRequest | dict[str, Any]],
) -> list[CompletionRequest]:
    """Coerce plain dicts to CompletionRequest objects."""
    normalised: list[CompletionRequest] = []
    for r in requests:
        if isinstance(r, dict):
            # Normalise message dicts inside the request dict
            raw_messages = r.get("messages", [])
            messages = [
                Message(**m) if isinstance(m, dict) else m for m in raw_messages
            ]
            normalised.append(CompletionRequest(**{**r, "messages": messages}))
        else:
            normalised.append(r)
    return normalised


def _make_batch_result(
    results: list[CompletionRequest | None],
    errors: list[BatchError],
    normalised: list[CompletionRequest],
) -> BatchResult:
    """Build a BatchResult from accumulated results and errors."""

    successful = sum(1 for r in results if r is not None)
    failed = len(errors)
    total_tokens = sum(
        r.usage.total_tokens  # type: ignore[union-attr]
        for r in results
        if r is not None
    )
    return BatchResult(
        results=results,  # type: ignore[arg-type]
        errors=errors,
        successful=successful,
        failed=failed,
        total_tokens=total_tokens,
    )


# ---------------------------------------------------------------------------
# Sync: batch()
# ---------------------------------------------------------------------------


def batch(
    requests: list[CompletionRequest | dict[str, Any]],
    *,
    max_concurrency: int = 5,
    fail_fast: bool = False,
    middleware: list[BaseMiddleware] | None = None,
    timeout: float | None = None,
) -> BatchResult:
    """Execute multiple completion requests in parallel (synchronous).

    Uses a ``ThreadPoolExecutor`` so it is safe to call from any context,
    including inside a running asyncio event loop.

    Args:
        requests:        List of :class:`~llmgate.types.CompletionRequest`
                         objects or plain dicts with the same fields.
        max_concurrency: Maximum number of requests in flight at once.
                         Defaults to ``5``.
        fail_fast:       If ``True``, raises the first exception immediately
                         and cancels pending work.  If ``False`` (default),
                         all requests are attempted and errors are collected
                         in :attr:`~llmgate.types.BatchResult.errors`.
        middleware:      Optional middleware list applied to every individual
                         request, same as passing ``middleware`` to
                         :func:`~llmgate.completion.completion`.
        timeout:         Per-request timeout in seconds.  ``None`` means no
                         timeout.  Timed-out requests raise
                         :class:`~llmgate.exceptions.BatchTimeoutError`.

    Returns:
        :class:`~llmgate.types.BatchResult`

    Raises:
        Any :class:`~llmgate.exceptions.LLMGateError` subclass when
        ``fail_fast=True`` and a request fails.
    """
    normalised = _normalise_requests(requests)
    results: list[Any] = [None] * len(normalised)
    errors: list[BatchError] = []

    def _run_one(idx: int, req: CompletionRequest) -> tuple[int, Any, Exception | None]:
        try:
            resp = completion(
                req.model,
                req.messages,
                middleware=middleware,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                tools=req.tools,
                tool_choice=req.tool_choice,
                response_format=req.response_format,
                **req.extra_kwargs,
            )
            return idx, resp, None
        except Exception as exc:  # noqa: BLE001
            return idx, None, exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        # Submit all tasks up-front so the executor can run them concurrently.
        # We then call future.result(timeout=timeout) on each future to enforce
        # a per-request deadline.  Futures that have already completed (common
        # when max_concurrency is sufficient) return immediately; futures that
        # are still running will raise TimeoutError after `timeout` seconds.
        futures: list[concurrent.futures.Future[tuple[int, Any, Exception | None]]] = [
            executor.submit(_run_one, i, req) for i, req in enumerate(normalised)
        ]

        for idx, future in enumerate(futures):
            try:
                _, resp, exc = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                future.cancel()
                exc = BatchTimeoutError(idx, timeout)  # type: ignore[arg-type]
                resp = None

            if exc is not None:
                if fail_fast:
                    for f in futures:
                        f.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise exc
                errors.append(
                    BatchError(
                        index=idx,
                        request=normalised[idx],
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                )
            else:
                results[idx] = resp

    return _make_batch_result(results, errors, normalised)


# ---------------------------------------------------------------------------
# Async: abatch()
# ---------------------------------------------------------------------------


async def abatch(
    requests: list[CompletionRequest | dict[str, Any]],
    *,
    max_concurrency: int = 5,
    fail_fast: bool = False,
    middleware: list[BaseMiddleware] | None = None,
    timeout: float | None = None,
) -> BatchResult:
    """Execute multiple completion requests in parallel (asynchronous).

    Uses ``asyncio.Semaphore`` to cap concurrent coroutines.

    Args:
        requests:        List of :class:`~llmgate.types.CompletionRequest`
                         objects or plain dicts with the same fields.
        max_concurrency: Maximum number of requests in flight at once.
                         Defaults to ``5``.
        fail_fast:       If ``True``, raises the first exception and cancels
                         remaining tasks.  If ``False`` (default), all
                         requests are attempted and errors collected.
        middleware:      Optional middleware list applied per request.
        timeout:         Per-request timeout in seconds.  ``None`` means no
                         timeout.  Timed-out requests raise
                         :class:`~llmgate.exceptions.BatchTimeoutError`.

    Returns:
        :class:`~llmgate.types.BatchResult`
    """
    normalised = _normalise_requests(requests)
    results: list[Any] = [None] * len(normalised)
    errors: list[BatchError] = []
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_one(
        idx: int, req: CompletionRequest
    ) -> tuple[int, Any, Exception | None]:
        async with semaphore:
            try:
                coro = acompletion(
                    req.model,
                    req.messages,
                    middleware=middleware,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    tools=req.tools,
                    tool_choice=req.tool_choice,
                    response_format=req.response_format,
                    **req.extra_kwargs,
                )
                if timeout is not None:
                    try:
                        resp = await asyncio.wait_for(coro, timeout=timeout)
                    except asyncio.TimeoutError:
                        raise BatchTimeoutError(idx, timeout)
                else:
                    resp = await coro
                return idx, resp, None
            except Exception as exc:  # noqa: BLE001
                return idx, None, exc

    tasks = [asyncio.create_task(_run_one(i, req)) for i, req in enumerate(normalised)]

    for coro in asyncio.as_completed(tasks):
        idx, resp, exc = await coro
        if exc is not None:
            if fail_fast:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise exc
            errors.append(
                BatchError(
                    index=idx,
                    request=normalised[idx],
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            )
        else:
            results[idx] = resp

    return _make_batch_result(results, errors, normalised)
