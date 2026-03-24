# llmgate User Guide

A practical, end-to-end guide to using llmgate in your projects.

---

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Core Concepts](#core-concepts)
4. [Making Completions](#making-completions)
5. [Provider-Specific Notes](#provider-specific-notes)
6. [Async Usage](#async-usage)
7. [Error Handling](#error-handling)
8. [Working with Responses](#working-with-responses)
9. [Building a Chatbot](#building-a-chatbot)
10. [Using with FastAPI](#using-with-fastapi)
11. [Adding a Custom Provider](#adding-a-custom-provider)
12. [Migrating from LiteLLM](#migrating-from-litellm)

---

## Installation

```bash
# pip
pip install llmgate

# uv (recommended)
uv add llmgate
```

Requires **Python 3.10+**.

---

## Configuration

llmgate reads API keys from environment variables. Set them once and forget.

**Option 1 — export in shell:**

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AIza..."
export GROQ_API_KEY="gsk_..."
```

**Option 2 — `.env` file + python-dotenv:**

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
```

```python
from dotenv import load_dotenv
load_dotenv()

from llmgate import completion
```

**Option 3 — inline per call** (useful for multi-tenant apps):

```python
response = completion("gpt-4o-mini", messages, api_key="sk-user-specific-key")
```

---

## Core Concepts

### Messages

llmgate uses the same message format as OpenAI — a list of dicts with `role` and `content`:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is the capital of France?"},
]
```

Valid roles: `system`, `user`, `assistant`, `tool`.

You can also use the typed `Message` class:

```python
from llmgate.types import Message

messages = [
    Message(role="system",    content="You are helpful."),
    Message(role="user",      content="Hello!"),
]
```

Both forms work interchangeably anywhere in the API.

### Model routing

llmgate determines the provider from the model name:

| Model string starts with... | Provider |
|---|---|
| `gpt-`, `o1-`, `o3-`, `chatgpt-` | OpenAI |
| `gemini-` | Google Gemini |
| `claude-` | Anthropic |
| `groq/` | Groq |

For Groq, always prefix with `groq/`:

```python
# ✅ Correct
completion("groq/llama-3.1-8b-instant", messages)

# ❌ Will raise ModelNotFoundError
completion("llama-3.1-8b-instant", messages)
```

You can bypass auto-routing with the `provider=` argument:

```python
completion("llama-3.1-8b-instant", messages, provider="groq")
```

---

## Making Completions

### Minimal call

```python
from llmgate import completion

response = completion("gpt-4o-mini", [{"role": "user", "content": "Hi!"}])
print(response.text)
```

### With all common parameters

```python
response = completion(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "system", "content": "You are concise."},
        {"role": "user",   "content": "Explain recursion."},
    ],
    max_tokens=200,
    temperature=0.5,
    top_p=0.95,
)
print(response.text)
print(f"Used {response.usage.total_tokens} tokens")
```

### Passing provider-specific parameters

Extra keyword arguments are forwarded verbatim to the underlying SDK:

```python
# OpenAI: pass frequency_penalty
response = completion(
    "gpt-4o",
    messages,
    frequency_penalty=0.5,
    presence_penalty=0.2,
)

# Anthropic: pass top_k
response = completion(
    "claude-3-haiku-20240307",
    messages,
    top_k=40,
)
```

---

## Provider-Specific Notes

### OpenAI

- Env var: `OPENAI_API_KEY`
- Model prefixes: `gpt-`, `o1-`, `o3-`, `chatgpt-`
- `o1-*` and `o3-*` models don't support `system` messages or `temperature` — pass those via `extra_kwargs` if needed.

### Anthropic

- Env var: `ANTHROPIC_API_KEY`
- Model prefix: `claude-`
- `max_tokens` is **required** by the Anthropic API. llmgate defaults to `1024` if you don't provide it.
- `system` messages are extracted from your messages list automatically.

```python
# System message is handled transparently:
response = completion("claude-3-5-sonnet-20241022", [
    {"role": "system", "content": "Be brief."},
    {"role": "user",   "content": "What is Python?"},
])
```

### Google Gemini

- Env var: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
- Model prefix: `gemini-`
- Uses the new `google-genai` SDK.
- Multi-turn: alternating `user`/`model` messages expected. llmgate converts `assistant` → `model` automatically.

### Groq

- Env var: `GROQ_API_KEY`
- Always use the `groq/` prefix: `groq/llama-3.1-8b-instant`
- The Groq SDK is OpenAI-compatible, so the interface is identical to OpenAI.
- Popular models: `groq/llama-3.1-8b-instant`, `groq/llama-3.3-70b-versatile`, `groq/gemma2-9b-it`

---

## Async Usage

Every call has an async counterpart — `acompletion()` — with an identical signature.

```python
import asyncio
from llmgate import acompletion

async def chat(message: str) -> str:
    response = await acompletion(
        "groq/llama-3.1-8b-instant",
        [{"role": "user", "content": message}],
        max_tokens=512,
    )
    return response.text

print(asyncio.run(chat("Hello!")))
```

### Parallel calls

```python
import asyncio
from llmgate import acompletion

async def compare_models(prompt: str):
    tasks = [
        acompletion("gpt-4o-mini",               [{"role": "user", "content": prompt}]),
        acompletion("claude-3-haiku-20240307",    [{"role": "user", "content": prompt}]),
        acompletion("gemini-2.5-flash-lite",      [{"role": "user", "content": prompt}]),
        acompletion("groq/llama-3.1-8b-instant",  [{"role": "user", "content": prompt}]),
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        print(f"[{r.provider}] {r.text[:80]}")

asyncio.run(compare_models("What is 42?"))
```

---

## Error Handling

```python
from llmgate import completion
from llmgate.exceptions import (
    LLMGateError,           # catch-all base class
    AuthError,              # bad/missing API key
    RateLimitError,         # 429, quota exceeded
    ProviderAPIError,       # other 4xx/5xx
    ModelNotFoundError,     # unknown model string
    ConfigError,            # missing/bad config
    StreamingNotSupported,  # stream=True (not yet implemented)
)

try:
    response = completion("gpt-4o-mini", messages)
except AuthError as e:
    # Retry with a different key, or alert the user
    print(f"Auth failed for {e.provider}. Check your API key.")
except RateLimitError as e:
    # Implement exponential back-off
    import time; time.sleep(5)
except ProviderAPIError as e:
    print(f"Provider error (HTTP {e.status_code}): {e}")
except ModelNotFoundError as e:
    print(f"'{e.model}' is not recognised. Check the model name.")
except LLMGateError as e:
    # Catch anything else from llmgate
    print(f"llmgate error: {e}")
```

### Retry pattern

```python
import time
from llmgate import completion
from llmgate.exceptions import RateLimitError

def completion_with_retry(model, messages, retries=3, backoff=2.0, **kwargs):
    for attempt in range(retries):
        try:
            return completion(model, messages, **kwargs)
        except RateLimitError:
            if attempt == retries - 1:
                raise
            time.sleep(backoff ** attempt)
```

---

## Working with Responses

### Access the response text

```python
response = completion("gpt-4o-mini", messages)

# Shortcut
print(response.text)

# Full path
print(response.choices[0].message.content)
```

### Token usage

```python
print(f"Prompt tokens:     {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens:      {response.usage.total_tokens}")
```

### Access provider-specific fields (escape hatch)

```python
response = completion("gpt-4o-mini", messages)

# Access the raw OpenAI response object
raw = response.raw
print(raw.system_fingerprint)  # OpenAI-specific field
```

### Serialise to dict / JSON

```python
import json

d = response.model_dump()            # dict (excludes `raw` by default)
j = response.model_dump_json()       # JSON string
print(json.loads(j)["usage"])
```

---

## Building a Chatbot

```python
from llmgate import completion
from llmgate.types import Message

class Chatbot:
    def __init__(self, model: str = "gpt-4o-mini", system: str | None = None):
        self.model = model
        self.history: list[Message] = []
        if system:
            self.history.append(Message(role="system", content=system))

    def chat(self, user_input: str) -> str:
        self.history.append(Message(role="user", content=user_input))
        response = completion(self.model, self.history)
        reply = response.text
        self.history.append(Message(role="assistant", content=reply))
        return reply

    def reset(self):
        self.history = [m for m in self.history if m.role == "system"]


# Usage
bot = Chatbot("groq/llama-3.1-8b-instant", system="You are a helpful assistant.")
print(bot.chat("Hi! What's your name?"))
print(bot.chat("What did I just ask you?"))
```

---

## Using with FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llmgate import acompletion
from llmgate.exceptions import AuthError, RateLimitError

app = FastAPI()

class ChatRequest(BaseModel):
    model: str = "groq/llama-3.1-8b-instant"
    message: str
    system: str | None = None

class ChatResponse(BaseModel):
    reply: str
    provider: str
    total_tokens: int

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.message})

    try:
        response = await acompletion(req.model, messages)
    except AuthError:
        raise HTTPException(status_code=401, detail="Invalid API key")
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limited. Try again later.")

    return ChatResponse(
        reply=response.text,
        provider=response.provider,
        total_tokens=response.usage.total_tokens,
    )
```

---

## Adding a Custom Provider

1. **Subclass `BaseProvider`**

```python
# my_provider.py
from llmgate.base import BaseProvider
from llmgate.types import CompletionRequest, CompletionResponse, Choice, Message, TokenUsage

class MyProvider(BaseProvider):
    name = "myprovider"
    supported_model_prefixes = ("my-model-",)   # or use "myprovider/" prefix style

    def __init__(self, api_key=None, **kwargs):
        import my_sdk
        self._client = my_sdk.Client(api_key=api_key or os.environ["MY_API_KEY"])

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        raw = self._client.generate(
            model=request.model,
            messages=[m.to_dict() for m in request.messages],
        )
        return CompletionResponse(
            id=raw.id,
            model=request.model,
            provider=self.name,
            choices=[Choice(
                index=0,
                message=Message(role="assistant", content=raw.text),
                finish_reason=raw.finish_reason,
            )],
            usage=TokenUsage(
                prompt_tokens=raw.usage.input,
                completion_tokens=raw.usage.output,
                total_tokens=raw.usage.total,
            ),
            raw=raw,
        )

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        raw = await self._client.agenerate(...)
        return ...
```

2. **Register it once at startup**

```python
import llmgate.completion as _registry
from my_provider import MyProvider

_registry._PROVIDER_CLASSES.insert(0, MyProvider)
_registry._PROVIDER_NAME_MAP["myprovider"] = MyProvider
```

3. **Use it**

```python
response = completion("my-model-v1", messages)
```

---

## Migrating from LiteLLM

| LiteLLM | llmgate | Notes |
|---|---|---|
| `litellm.completion(model, messages)` | `llmgate.completion(model, messages)` | Same shape |
| `litellm.acompletion(...)` | `llmgate.acompletion(...)` | Same shape |
| `response["choices"][0]["message"]["content"]` | `response.text` | Pydantic model, not dict |
| `response["usage"]["total_tokens"]` | `response.usage.total_tokens` | Attribute access |
| `groq/llama-3.1-8b` | `groq/llama-3.1-8b` | Same prefix convention |
| `litellm.exceptions.AuthenticationError` | `llmgate.exceptions.AuthError` | Renamed, slimmer hierarchy |
| Gateway / proxy mode | ❌ Not supported | By design — use a dedicated gateway |
| Streaming | ❌ Not yet (v0.2) | Interface stubbed |
