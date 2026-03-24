<div align="center">

# llmgate

**A lightweight, provider-agnostic LLM calling library for Python.**

*Born from the need for a simple, secure, dependency-free alternative to heavyweight LLM SDKs.*

[![PyPI version](https://badge.fury.io/py/llmgate.svg)](https://badge.fury.io/py/llmgate)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-29%20passing-brightgreen)](tests/)

</div>

---

## Why llmgate?

**LiteLLM is powerful — but it's also complex, heavy, and recently had a security incident.** llmgate was built as a focused alternative: *one job, done right*.

- ✅ **No gateway server** — just a Python library
- ✅ **No proxy, no UI, no bloat** — pure LLM calling
- ✅ **4 providers out of the box** — OpenAI, Gemini, Anthropic, Groq
- ✅ **Pydantic v2 types** — validated I/O, great IDE support
- ✅ **Sync & async** — `completion()` and `acompletion()`
- ✅ **Extensible by design** — add providers via subclass, no core changes
- ✅ **Streaming-ready** — interface stubbed for v0.2

---

## Installation

```bash
pip install llmgate
```

With [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv add llmgate
```

**Requires Python 3.10+**

---

## Quick Start

### 1. Set your API keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AIza..."       # or GOOGLE_API_KEY
export GROQ_API_KEY="gsk_..."
```

Or use a `.env` file — llmgate reads standard environment variables.

### 2. Make your first call

```python
from llmgate import completion

response = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.text)
# → "Hello! How can I help you today?"
```

That's it. Same API for every provider.

---

## Supported Providers

| Provider | Model prefix | Example |
|---|---|---|
| **OpenAI** | `gpt-`, `o1-`, `o3-` | `gpt-4o`, `gpt-4o-mini`, `o3-mini` |
| **Anthropic** | `claude-` | `claude-3-5-sonnet-20241022`, `claude-3-haiku-20240307` |
| **Google Gemini** | `gemini-` | `gemini-2.5-flash-lite`, `gemini-2.0-flash` |
| **Groq** | `groq/` | `groq/llama-3.1-8b-instant`, `groq/gemma2-9b-it` |

Provider is **auto-detected from the model name**. Use `provider=` to override explicitly.

---

## API Reference

### `completion()`

```python
from llmgate import completion

response = completion(
    model="gpt-4o-mini",           # Required. Model name (determines provider).
    messages=[...],                # Required. List of message dicts or Message objects.
    provider=None,                 # Optional. Force provider: "openai"|"gemini"|"anthropic"|"groq"
    api_key=None,                  # Optional. Override the env var API key for this call.
    max_tokens=None,               # Optional. Maximum tokens in the response.
    temperature=None,              # Optional. 0.0–2.0. Controls randomness.
    top_p=None,                    # Optional. Nucleus sampling parameter.
    stream=False,                  # Not yet supported (raises NotImplementedError).
    **kwargs                       # Extra params forwarded to the provider SDK.
)
```

### `acompletion()`

Identical signature to `completion()` but returns a coroutine. Use in async contexts:

```python
import asyncio
from llmgate import acompletion

async def main():
    response = await acompletion("gemini-2.5-flash-lite", [
        {"role": "user", "content": "What is 2 + 2?"}
    ])
    print(response.text)

asyncio.run(main())
```

### `CompletionResponse`

```python
response.text                      # str   — content of the first choice (shortcut)
response.id                        # str   — unique response ID from the provider
response.model                     # str   — model name used
response.provider                  # str   — "openai" | "gemini" | "anthropic" | "groq"
response.choices                   # list[Choice]
response.choices[0].message.role   # str   — "assistant"
response.choices[0].message.content# str   — full response text
response.choices[0].finish_reason  # str | None
response.usage.prompt_tokens       # int
response.usage.completion_tokens   # int
response.usage.total_tokens        # int
response.raw                       # Any   — raw SDK response (escape hatch)
```

---

## Usage Examples

### System prompts

```python
from llmgate import completion

response = completion(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "system", "content": "You are a concise technical assistant."},
        {"role": "user", "content": "Explain asyncio in one sentence."},
    ],
    max_tokens=100,
    temperature=0.3,
)
print(response.text)
```

### Multi-turn conversation

```python
from llmgate import completion

history = [{"role": "user", "content": "My name is Alex."}]
response = completion("gpt-4o-mini", history)

history.append({"role": "assistant", "content": response.text})
history.append({"role": "user", "content": "What is my name?"})

response = completion("gpt-4o-mini", history)
print(response.text)  # → "Your name is Alex."
```

### Forcing a specific provider

```python
# Route explicitly to Groq even without the "groq/" prefix
response = completion(
    "llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Hello"}],
    provider="groq",
)
```

### Passing an API key inline

```python
response = completion(
    "gpt-4o-mini",
    messages=[{"role": "user", "content": "Hi"}],
    api_key="sk-my-special-key",
)
```

### Async (e.g. FastAPI)

```python
from fastapi import FastAPI
from llmgate import acompletion

app = FastAPI()

@app.post("/chat")
async def chat(prompt: str):
    response = await acompletion(
        "groq/llama-3.1-8b-instant",
        [{"role": "user", "content": prompt}],
    )
    return {"reply": response.text, "tokens": response.usage.total_tokens}
```

### Using `Message` objects directly

```python
from llmgate import completion
from llmgate.types import Message

messages = [
    Message(role="system", content="Be concise."),
    Message(role="user", content="What is Python?"),
]
response = completion("gemini-2.5-flash-lite", messages)
```

---

## Error Handling

```python
from llmgate import completion
from llmgate.exceptions import (
    AuthError,          # 401 / bad API key
    RateLimitError,     # 429 / quota exceeded
    ProviderAPIError,   # other provider errors
    ModelNotFoundError, # no provider matched the model string
    StreamingNotSupported,
)

try:
    response = completion("gpt-4o-mini", [{"role": "user", "content": "Hi"}])
except AuthError as e:
    print(f"Bad API key for provider: {e.provider}")
except RateLimitError as e:
    print(f"Rate limited by {e.provider} (HTTP {e.status_code}). Back off and retry.")
except ProviderAPIError as e:
    print(f"Provider error: {e}")
except ModelNotFoundError as e:
    print(f"Unknown model: {e.model}")
    # Tip: for Groq models, use the groq/ prefix, e.g. "groq/llama-3.1-8b-instant"
```

---

## Adding a Custom Provider

llmgate is built to be extended. Subclass `BaseProvider` and register it:

```python
from llmgate.base import BaseProvider
from llmgate.types import CompletionRequest, CompletionResponse, Choice, Message, TokenUsage

class MistralProvider(BaseProvider):
    name = "mistral"
    supported_model_prefixes = ("mistral-", "open-mistral-")

    def __init__(self, api_key=None, **kwargs):
        # set up your SDK client here
        ...

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        # call your SDK, map to CompletionResponse
        raw = ...
        return CompletionResponse(
            id=raw.id,
            model=request.model,
            provider=self.name,
            choices=[Choice(index=0, message=Message(role="assistant", content=raw.text))],
            usage=TokenUsage(...),
            raw=raw,
        )

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        ...
```

Register it once at app startup:

```python
import llmgate.completion as _registry
from myproviders import MistralProvider

_registry._PROVIDER_CLASSES.insert(0, MistralProvider)
_registry._PROVIDER_NAME_MAP["mistral"] = MistralProvider
```

Then use it like any built-in provider:

```python
response = completion("mistral-large-latest", messages)
```

---

## Environment Variables

| Variable | Provider | Description |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI | Your OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic | Your Anthropic API key |
| `GEMINI_API_KEY` | Gemini | Your Google AI Studio API key |
| `GOOGLE_API_KEY` | Gemini | Fallback if `GEMINI_API_KEY` is not set |
| `GROQ_API_KEY` | Groq | Your Groq API key |

---

## Roadmap

- [ ] **Streaming** — `stream=True` via async generators
- [ ] **Tool / function calling** — unified tool schema across providers
- [ ] **Middleware hooks** — logging, caching, retry, rate-limit backoff
- [ ] **More providers** — Mistral, Cohere, Azure OpenAI, AWS Bedrock, Ollama (local)
- [ ] **Structured outputs** — Pydantic model as `response_format`
- [ ] **Embeddings** — `embed()` and `aembed()`

---

## Development

```bash
git clone https://github.com/thatAverageGuy/llm-gate
cd llm-gate
uv sync          # install deps + dev deps
uv run pytest    # run test suite (29 tests, all mocked)
uv build         # build wheel + sdist
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

---

## License

MIT — see [LICENSE](LICENSE).