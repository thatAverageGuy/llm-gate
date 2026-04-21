# Quick Start

Five minutes from install to first response.

---

## 1. Set your API key

=== "Shell"

    ```bash
    export OPENAI_API_KEY="sk-..."
    ```

=== ".env file"

    ```bash
    # .env
    OPENAI_API_KEY=sk-...
    ```

    ```python
    from dotenv import load_dotenv
    load_dotenv()
    ```

=== "Inline (per call)"

    ```python
    resp = completion("gpt-4o-mini", messages, api_key="sk-...")
    ```

---

## 2. Make your first call

```python
from llmgate import completion

resp = completion(
    "gpt-4o-mini",
    [{"role": "user", "content": "What is the capital of France?"}],
)

print(resp.text)
# → "The capital of France is Paris."
```

---

## 3. Switch providers — literally one word

```python
from llmgate import completion

messages = [{"role": "user", "content": "Explain recursion in one sentence."}]

# OpenAI
resp = completion("gpt-4o-mini", messages)

# Anthropic
resp = completion("claude-3-5-haiku-20241022", messages)

# Gemini
resp = completion("gemini-2.5-flash-lite", messages)

# Groq (fastest)
resp = completion("groq/llama-3.1-8b-instant", messages)

print(resp.text)   # always the same
print(resp.provider)   # "openai" | "anthropic" | "gemini" | "groq"
```

---

## 4. Async

```python
import asyncio
from llmgate import acompletion

async def main():
    resp = await acompletion(
        "groq/llama-3.3-70b-versatile",
        [{"role": "user", "content": "Hello!"}],
    )
    print(resp.text)

asyncio.run(main())
```

---

## What's next?

- [Configuration](configuration.md) — API keys, env vars, per-call overrides
- [Completions guide](../guide/completions.md) — parameters, responses, provider-specific options
- [Vision guide](../guide/vision.md) — image inputs across all providers
- [Providers](../providers/index.md) — per-provider notes and model lists
