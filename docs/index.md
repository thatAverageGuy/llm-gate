# llmgate

<div class="grid cards" markdown>

-   :zap: **One API, every provider**

    Switch between OpenAI, Gemini, Anthropic, Groq, and six more providers by changing a single word.

-   :eyes: **Vision & Multimodal**

    Pass images alongside text — URL or base64 — with automatic per-provider serialization.

-   :arrows_counterclockwise: **Fallback & Routing**

    Pass a list of models — llmgate tries each in order and falls back automatically on rate limits or errors.

-   :package: **Zero boilerplate**

    Consistent response shape, unified error types, and Pydantic v2 models throughout.

-   :arrows_counterclockwise: **Async-first**

    Every call has a `sync` and `async` variant. Batch completions with concurrency control built in.

</div>

```python
from llmgate import completion

# OpenAI
resp = completion("gpt-4o-mini", [{"role": "user", "content": "Hello!"}])

# Switch to Gemini — one word changes
resp = completion("gemini-2.5-flash-lite", [{"role": "user", "content": "Hello!"}])

# Switch to Groq
resp = completion("groq/llama-3.3-70b-versatile", [{"role": "user", "content": "Hello!"}])

print(resp.text)  # always the same shape
```

---

## Why llmgate?

Every LLM provider has a different SDK, different message formats, different error types, and different response shapes. Switching providers in production means touching dozens of files.

llmgate solves this with a **thin, stable abstraction**: one function, one response model, one error hierarchy — regardless of which provider is under the hood.

| What you get | Detail |
|---|---|
| **9 providers** | OpenAI · Anthropic · Gemini · Groq · Mistral · Cohere · Azure · Bedrock · Ollama |
| **Vision** | URL + base64 images across 8 providers |
| **Streaming** | `stream=True` returns `Iterator[StreamChunk]` |
| **Tools** | Function calling with a unified `ToolCall` type |
| **Structured outputs** | Pass any Pydantic model, get back a validated instance |
| **Embeddings** | 7 providers, batched, async |
| **Batch** | Parallel completions with configurable concurrency |
| **Fallback & Routing** | `model=[...]` list — automatic multi-provider failover with `AllProvidersFailedError` |
| **Middleware** | Retry, cache, logging, rate-limit, fallback — composable |

---

## Install

```bash
pip install llmgate
```

!!! tip "Optional providers"
    Some providers require an extra package:
    ```bash
    pip install llmgate[mistral]   # Mistral
    pip install llmgate[cohere]    # Cohere
    pip install llmgate[bedrock]   # AWS Bedrock
    pip install llmgate[ollama]    # Ollama (local)
    pip install llmgate[all]       # everything
    ```

---

## Quick Example

```python
from llmgate import completion
import os

os.environ["OPENAI_API_KEY"] = "sk-..."

resp = completion(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user",   "content": "What is a large language model?"},
    ],
    max_tokens=100,
)

print(resp.text)
print(f"Tokens used: {resp.usage.total_tokens}")
```

[Get started →](getting-started/installation.md){ .md-button .md-button--primary }
[View on GitHub →](https://github.com/thatAverageGuy/llm-gate){ .md-button }
