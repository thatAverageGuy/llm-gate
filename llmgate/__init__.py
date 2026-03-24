"""
llmgate
~~~~~~~
Lightweight, provider-agnostic LLM calling library.

Supported providers:  OpenAI · Google Gemini · Anthropic · Groq

Quick start::

    from llmgate import completion

    resp = completion("gpt-4o-mini", [{"role": "user", "content": "Hello!"}])
    print(resp.text)

Async::

    from llmgate import acompletion

    resp = await acompletion("gemini-1.5-flash", [{"role": "user", "content": "Hi"}])
    print(resp.text)

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
from llmgate.types import (
    Choice,
    CompletionRequest,
    CompletionResponse,
    Message,
    StreamChunk,
    TokenUsage,
)

__version__ = "0.1.0"
__all__ = [
    # Core API
    "completion",
    "acompletion",
    # Types
    "Message",
    "CompletionRequest",
    "CompletionResponse",
    "Choice",
    "TokenUsage",
    "StreamChunk",
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
