"""
llmgate
~~~~~~~
Lightweight, provider-agnostic LLM calling library.

Supported providers:  OpenAI · Google Gemini · Anthropic · Groq

Quick start::

    from llmgate import completion

    resp = completion("gpt-4o-mini", [{"role": "user", "content": "Hello!"}])
    print(resp.text)

With middleware::

    from llmgate import LLMGate
    from llmgate.middleware import RetryMiddleware, LoggingMiddleware, CacheMiddleware

    gate = LLMGate(middleware=[
        RetryMiddleware(max_retries=3),
        LoggingMiddleware(level="INFO"),
        CacheMiddleware(ttl=300),
    ])
    resp = gate.completion("groq/llama-3.1-8b-instant", messages)

"""
from __future__ import annotations

from llmgate.completion import acompletion, completion
from llmgate.exceptions import (
    AuthError,
    ConfigError,
    LLMGateError,
    ModelNotFoundError,
    ProviderAPIError,
    ProviderError,
    RateLimitError,
    StreamingNotSupported,
)
from llmgate.gate import LLMGate
from llmgate.middleware import (
    BaseMiddleware,
    CacheMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
    RetryMiddleware,
)
from llmgate.types import (
    Choice,
    CompletionRequest,
    CompletionResponse,
    FunctionDefinition,
    Message,
    StreamChunk,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)

__version__ = "0.2.0"
__all__ = [
    # Core API
    "completion",
    "acompletion",
    # Client
    "LLMGate",
    # Middleware
    "BaseMiddleware",
    "LoggingMiddleware",
    "RetryMiddleware",
    "CacheMiddleware",
    "RateLimitMiddleware",
    # Types
    "Message",
    "CompletionRequest",
    "CompletionResponse",
    "Choice",
    "TokenUsage",
    "StreamChunk",
    "FunctionDefinition",
    "ToolCall",
    "ToolDefinition",
    # Exceptions
    "LLMGateError",
    "ProviderError",
    "AuthError",
    "RateLimitError",
    "ProviderAPIError",
    "ModelNotFoundError",
    "ConfigError",
    "StreamingNotSupported",
]
