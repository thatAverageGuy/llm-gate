# llmgate

> A lightweight, provider-agnostic Python library for calling LLMs — one API for every provider.

[![PyPI version](https://img.shields.io/pypi/v/llmgate)](https://pypi.org/project/llmgate/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

---

## Why llmgate?

Switch between OpenAI, Gemini, Anthropic, Groq, Mistral, Cohere, Azure, Bedrock, or Ollama **without changing your application code**. Same function, same response shape, same error types — every time.

```python
from llmgate import completion

# OpenAI
resp = completion("gpt-4o-mini", messages)

# Switch to Groq — literally one word changes
resp = completion("groq/llama-3.1-8b-instant", messages)

# Switch to Gemini
resp = completion("gemini-2.5-flash-lite", messages)

print(resp.text)  # always the same
```

---

## Install

```bash
pip install llmgate
```

**Optional provider extras:**

```bash
pip install llmgate[mistral]          # Mistral
pip install llmgate[cohere]           # Cohere
pip install llmgate[bedrock]          # AWS Bedrock (boto3)
pip install llmgate[ollama]           # Ollama (local models)
pip install llmgate[all]              # everything
```

---

## Quick Start

```python
import os
from llmgate import completion

# Set your key (or put it in a .env file)
os.environ["GROQ_API_KEY"] = "gsk_..."

response = completion(
    model="groq/llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.text)
# → "Hello! How can I help you today?"
```

---

## Supported Providers

| Provider | Core / Optional | Model prefix | Install |
|---|---|---|---|
| **OpenAI** | Core | `gpt-4o`, `o1-`, `o3-` | included |
| **Anthropic** | Core | `claude-` | included |
| **Google Gemini** | Core | `gemini-` | included |
| **Groq** | Core | `groq/` | included |
| **Mistral** | Optional | `mistral/` | `llmgate[mistral]` |
| **Cohere** | Optional | `cohere/` | `llmgate[cohere]` |
| **Azure OpenAI** | Optional | `azure/` | included (uses openai) |
| **AWS Bedrock** | Optional | `bedrock/` | `llmgate[bedrock]` |
| **Ollama** (local) | Optional | `ollama/` | `llmgate[ollama]` |

Provider is **auto-detected from the model string**. Use `provider=` to override.

---

## API Reference

### `completion()` / `acompletion()`

```python
from llmgate import completion, acompletion

# Sync
resp = completion(
    model="gpt-4o-mini",
    messages=[...],
    provider=None,          # auto-detected; override with "openai", "groq", etc.
    api_key=None,           # overrides env var for this call
    max_tokens=None,
    temperature=None,
    top_p=None,
    stream=False,           # True → returns Iterator[StreamChunk]
    tools=[...],            # tool / function definitions
    tool_choice=None,       # "auto" | "none" | specific tool name
    response_format=None,   # Pydantic model class → enables structured output
    middleware=[...],       # list of BaseMiddleware instances
)

# Async — identical signature
resp = await acompletion("gemini-2.5-flash-lite", messages)
```

### `CompletionResponse`

```python
resp.text                      # str   — first choice content
resp.parsed                    # BaseModel | None  — populated when response_format set
resp.id                        # str   — provider response ID
resp.model                     # str
resp.provider                  # str   — "openai" | "gemini" | "anthropic" | ...
resp.choices                   # list[Choice]
resp.choices[0].message.role   # "assistant"
resp.choices[0].message.content
resp.choices[0].message.tool_calls  # list[ToolCall] | None
resp.usage.prompt_tokens       # int
resp.usage.completion_tokens   # int
resp.usage.total_tokens        # int
resp.raw                       # raw SDK response (escape hatch)
```

---

## Streaming

```python
from llmgate import completion

for chunk in completion("gpt-4o-mini", messages, stream=True):
    print(chunk.delta, end="", flush=True)

# Async
async for chunk in await acompletion("groq/llama-3.1-8b-instant", messages, stream=True):
    print(chunk.delta, end="", flush=True)
```

---

## Tool / Function Calling

```python
from llmgate import completion

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
}]

resp = completion("gpt-4o-mini", messages, tools=tools, tool_choice="auto")

if resp.tool_calls:
    for tc in resp.tool_calls:
        print(tc.function, tc.arguments)
```

---

## Structured Outputs

Pass any Pydantic `BaseModel` as `response_format` to get a validated, typed instance in `resp.parsed`.

```python
from pydantic import BaseModel
from llmgate import completion, parse

class Movie(BaseModel):
    title: str
    year: int
    rating: float

# Full response
resp = completion(
    "groq/llama-3.1-8b-instant",
    [{"role": "user", "content": "Name a great sci-fi film."}],
    response_format=Movie,
)
movie: Movie = resp.parsed

# Shorthand — returns the Pydantic instance directly
movie = parse("gemini-2.5-flash-lite", messages, response_format=Movie)

# Async
movie = await aparse("claude-3-5-haiku-20241022", messages, response_format=Movie)
```

**Provider strategies:**

| Provider | Strategy |
|---|---|
| OpenAI / Azure | Native `json_schema` (schema-constrained) |
| Gemini | `response_schema` + `response_mime_type` (native) |
| Groq / Mistral / Cohere / Ollama | `json_object` mode + Pydantic validation |
| Anthropic / Bedrock | Schema injected into system prompt + extraction |

> **Note:** `stream=True` and `response_format` cannot be used together.

---

## Embeddings

```python
from llmgate import embed, aembed

# Single text → OpenAI (auto-detected)
resp = embed("text-embedding-3-small", "Hello world")
vector: list[float] = resp.embeddings[0]

# Batch
resp = embed("text-embedding-3-small", ["Hello", "world"])
vectors: list[list[float]] = resp.embeddings

# Other providers
resp = embed("gemini/text-embedding-004", "Hello")
resp = embed("cohere/embed-english-v3.0", "Hello")
resp = embed("mistral/mistral-embed", "Hello")
resp = embed("ollama/nomic-embed-text", "Hello")
resp = embed("bedrock/amazon.titan-embed-text-v2:0", "Hello")
resp = embed("azure/my-embedding-deployment", "Hello")

# Control dimensions (OpenAI / Gemini / Azure)
resp = embed("text-embedding-3-small", "Hello", dimensions=256)

# Async
resp = await aembed("text-embedding-3-small", "Hello")
```

**EmbeddingResponse:**

```python
resp.embeddings   # list[list[float]] — one vector per input
resp.model        # str
resp.provider     # str
resp.usage        # TokenUsage
```

> Anthropic and Groq do not offer embedding APIs — they raise `EmbeddingsNotSupported`.

---

## Batch Completions

Execute multiple completion requests **in parallel** with configurable concurrency control.

```python
from llmgate import batch, abatch
from llmgate.types import CompletionRequest, Message

# Simple usage with dicts
results = batch([
    {"model": "gpt-4o-mini",              "messages": [{"role": "user", "content": "What is 1+1?"}]},
    {"model": "groq/llama-3.1-8b-instant", "messages": [{"role": "user", "content": "What is 2+2?"}]},
    {"model": "gemini-2.5-flash-lite",     "messages": [{"role": "user", "content": "What is 3+3?"}]},
], max_concurrency=3)

# Iterate successful responses (same order as input)
for resp in results.results:
    if resp is not None:
        print(resp.text)

# Check aggregate stats
print(f"{results.successful}/{results.successful + results.failed} succeeded")
print(f"Total tokens: {results.total_tokens}")
print(f"Success rate: {results.success_rate:.1%}")

# Inspect failures
for err in results.errors:
    print(f"Request {err.index} failed ({err.error_type}): {err.error}")

# Async version
results = await abatch(requests, max_concurrency=10)

# Fail immediately on first error instead of collecting
results = batch(requests, fail_fast=True)

# Apply middleware per-request
from llmgate.middleware import RetryMiddleware
results = batch(requests, middleware=[RetryMiddleware(max_retries=2)])
```

**`BatchResult` fields:**

```python
results.results        # list[CompletionResponse | None] — same order as input
results.errors         # list[BatchError] — details for each failed request
results.successful     # int — count of successful requests
results.failed         # int — count of failed requests
results.total_tokens   # int — aggregate token usage across successes
results.success_rate   # float — fraction that succeeded (0.0 – 1.0)
```

**`LLMGate` batch methods** use the gate's configured middleware automatically:

```python
from llmgate import LLMGate
from llmgate.middleware import RetryMiddleware, LoggingMiddleware

gate = LLMGate(middleware=[RetryMiddleware(max_retries=3), LoggingMiddleware()])
results = gate.batch(requests, max_concurrency=5)
results = await gate.abatch(requests, max_concurrency=5)
```

---

## Middleware

Apply logging, retry, caching, and rate-limiting as composable middleware:

```python
from llmgate import LLMGate
from llmgate.middleware import (
    RetryMiddleware,
    LoggingMiddleware,
    CacheMiddleware,
    RateLimitMiddleware,
)

gate = LLMGate(middleware=[
    RetryMiddleware(max_retries=3, backoff_factor=0.5),
    LoggingMiddleware(level="INFO"),
    CacheMiddleware(ttl=300),
    RateLimitMiddleware(rpm=60),
])

resp = gate.completion("gpt-4o-mini", messages)
resp = await gate.acompletion("gemini-2.5-flash-lite", messages)

# Streaming through middleware
for chunk in gate.stream("groq/llama-3.1-8b-instant", messages):
    print(chunk.delta, end="", flush=True)

# Embeddings through middleware
resp = gate.embed("text-embedding-3-small", "Hello")
```

---

## Error Handling

```python
from llmgate.exceptions import (
    AuthError,              # 401 / bad API key
    RateLimitError,         # 429 / quota exceeded
    ProviderAPIError,       # other provider errors
    ModelNotFoundError,     # unknown model / no provider matched
    EmbeddingsNotSupported, # provider doesn't have an embeddings API
)

try:
    resp = completion("gpt-4o-mini", messages)
except AuthError as e:
    print(f"Bad API key for {e.provider}")
except RateLimitError as e:
    print(f"Rate limited by {e.provider} — back off and retry")
except ProviderAPIError as e:
    print(f"Provider error: {e}")
except ModelNotFoundError as e:
    print(f"Unknown model: {e.model}")
```

---

## Environment Variables

| Variable | Provider |
|---|---|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Gemini |
| `GROQ_API_KEY` | Groq |
| `MISTRAL_API_KEY` | Mistral |
| `COHERE_API_KEY` | Cohere |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI (default: `2024-02-01`) |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` | Bedrock |
| `OLLAMA_HOST` | Ollama (default: `http://localhost:11434`) |

---

## Roadmap

These features are shipped ✅ or planned 🗓️:

| Feature | Status |
|---|---|
| Multi-provider completion (`completion()`, `acompletion()`) | ✅ v0.1 |
| Streaming (`stream=True`) | ✅ v0.2 |
| Tool / function calling | ✅ v0.2 |
| Composable middleware (logging, cache, retry, rate-limit) | ✅ v0.2 |
| 5 additional providers (Mistral, Cohere, Azure, Bedrock, Ollama) | ✅ v0.2 |
| Structured outputs (Pydantic `response_format`) | ✅ v0.3 |
| Embeddings API (`embed()`, `aembed()`) | ✅ v0.3 |
| **Batch completions** — parallel requests with concurrency control | ✅ v0.4 |
| **Vision / multimodal** — image inputs (GPT-4V, Gemini Vision, Claude) | 🗓️ planned |
| **Automatic tool-call loop** — orchestrate multi-step tool use | 🗓️ planned |
| **Token counting** — local tokenisation before sending | 🗓️ planned |
| **Prompt templates** — reusable, parameterised prompt builders | 🗓️ planned |

---

## Development

```bash
git clone https://github.com/thatAverageGuy/llm-gate
cd llm-gate
uv sync             # install deps + dev deps
uv run pytest       # run full test suite (all mocked — no API keys needed)
uv build            # build wheel + sdist
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

---

## License

MIT — see [LICENSE](LICENSE).