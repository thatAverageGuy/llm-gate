"""
llmgate.middleware
~~~~~~~~~~~~~~~~~~
Built-in middleware for logging, caching, retrying, and rate limiting.

Quick reference::

    from llmgate.middleware import (
        LoggingMiddleware,
        RetryMiddleware,
        CacheMiddleware,
        RateLimitMiddleware,
    )
    from llmgate import LLMGate

    gate = LLMGate(middleware=[
        RetryMiddleware(max_retries=3),
        LoggingMiddleware(level="INFO"),
        CacheMiddleware(ttl=300),
    ])
"""
from llmgate.middleware.base import BaseMiddleware
from llmgate.middleware.cache import CacheMiddleware
from llmgate.middleware.logging import LoggingMiddleware
from llmgate.middleware.ratelimit import RateLimitMiddleware
from llmgate.middleware.retry import RetryMiddleware

__all__ = [
    "BaseMiddleware",
    "LoggingMiddleware",
    "RetryMiddleware",
    "CacheMiddleware",
    "RateLimitMiddleware",
]
