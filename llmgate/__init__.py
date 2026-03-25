"""
llmgate
~~~~~~~
Lightweight, provider-agnostic LLM calling library.

Supported providers (core):   OpenAI · Google Gemini · Anthropic · Groq
Optional providers:            Mistral · Cohere · Azure OpenAI · AWS Bedrock · Ollama

Quick start::

    from llmgate import completion

    resp = completion("gpt-4o-mini", [{"role": "user", "content": "Hello!"}])
    print(resp.text)

Structured outputs::

    from pydantic import BaseModel
    from llmgate import parse

    class City(BaseModel):
        name: str
        population: int

    city = parse("groq/llama-3.1-8b-instant", messages, response_format=City)

Embeddings::

    from llmgate import embed

    resp = embed("text-embedding-3-small", "Hello world")
    vector = resp.embeddings[0]   # list[float]

With middleware::

    from llmgate import LLMGate
    from llmgate.middleware import RetryMiddleware, CacheMiddleware

    gate = LLMGate(middleware=[RetryMiddleware(max_retries=3), CacheMiddleware(ttl=300)])
    resp = gate.completion("groq/llama-3.1-8b-instant", messages)
"""
from __future__ import annotations

from llmgate.completion import acompletion, aparse, completion, parse
from llmgate.embeddings import aembed, embed
from llmgate.exceptions import (
    AuthError,
    ConfigError,
    EmbeddingsNotSupported,
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
    EmbeddingRequest,
    EmbeddingResponse,
    FunctionDefinition,
    Message,
    StreamChunk,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)

__version__ = "0.3.0"
__all__ = [
    # Completion API
    "completion",
    "acompletion",
    "parse",
    "aparse",
    # Embeddings API
    "embed",
    "aembed",
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
    "EmbeddingRequest",
    "EmbeddingResponse",
    # Exceptions
    "LLMGateError",
    "ProviderError",
    "AuthError",
    "RateLimitError",
    "ProviderAPIError",
    "ModelNotFoundError",
    "ConfigError",
    "StreamingNotSupported",
    "EmbeddingsNotSupported",
]
