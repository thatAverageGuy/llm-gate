# Fallback & Routing

Fallback routing lets you pass **multiple model strings** instead of one. llmgate tries each model in order and returns the first successful response — automatically, transparently, with zero extra code.

This is the single most powerful reliability feature for production LLM applications: no more hand-rolling retry loops across providers.

---

## Quickstart

```python
from llmgate import completion

resp = completion(
    model=["gpt-4o-mini", "groq/llama-3.1-8b-instant", "gemini-2.0-flash"],
    messages=[{"role": "user", "content": "Hello!"}],
)

print(resp.text)
print(resp.provider)           # → whichever model succeeded
print(resp.fallback_attempts)  # → ["gpt-4o-mini"] if first model failed
```

That's it. When `gpt-4o-mini` hits a rate limit, llmgate silently tries `groq/llama-3.1-8b-instant`, then `gemini-2.0-flash`. Your application code never changes.

---

## How it works

1. llmgate tries each model in list order.
2. If a model fails with a **triggering error** (`RateLimitError`, `ProviderAPIError`, or `AuthError`), it's recorded in `fallback_attempts` and the next model is tried.
3. If a model fails with a **non-triggering error** (e.g. `ModelNotFoundError`), the error propagates immediately — no fallback.
4. The first successful response is returned with `fallback_attempts` populated.
5. If **all models fail**, `AllProvidersFailedError` is raised with the full `(model, exception)` list.

```
model=["gpt-4o-mini", "groq/llama-3.1-8b-instant", "gemini-2.0-flash"]
         ↓ RateLimitError
              ↓ ProviderAPIError
                        ↓ ✓ success → resp.fallback_attempts = ["gpt-4o-mini", "groq/llama-3.1-8b-instant"]
```

---

## Observability

`CompletionResponse` has a new field:

```python
resp.fallback_attempts  # list[str] — models tried before this one
```

- **Empty list** (`[]`) — first model succeeded, no fallback occurred
- **Non-empty** (`["gpt-4o-mini"]`) — that model failed, the current `resp.provider` is the one that worked

Use this for logging, alerting, or dashboards:

```python
resp = completion(model=["gpt-4o-mini", "groq/llama-3.1-8b-instant"], messages=messages)

if resp.fallback_attempts:
    logger.warning(
        "Provider fallback: %s failed, used %s",
        resp.fallback_attempts,
        resp.provider,
    )
```

---

## Three API surfaces

### 1. Top-level `completion()` / `acompletion()`

The simplest path — just swap a `str` for a `list[str]`:

```python
from llmgate import completion, acompletion

# Sync
resp = completion(
    model=["gpt-4o-mini", "groq/llama-3.1-8b-instant"],
    messages=messages,
)

# Async
resp = await acompletion(
    model=["gpt-4o-mini", "groq/llama-3.1-8b-instant"],
    messages=messages,
)
```

### 2. `LLMGate(fallback_chain=[...])` — app-level config

Configure the chain once at startup. All middleware (retry, logging, cache) applies to **each candidate** before falling back:

```python
from llmgate import LLMGate
from llmgate.middleware import RetryMiddleware, LoggingMiddleware

gate = LLMGate(
    fallback_chain=["gpt-4o-mini", "groq/llama-3.1-8b-instant", "gemini-2.0-flash"],
    middleware=[
        RetryMiddleware(max_retries=2),   # retries each model before fallback
        LoggingMiddleware(level="INFO"),
    ],
)

# model arg is optional when fallback_chain is configured
resp = gate.completion(messages=messages)
resp = await gate.acompletion(messages=messages)
```

!!! tip "Retry then fall back"
    With `LLMGate(fallback_chain=[...])`, `RetryMiddleware` wraps **each individual model attempt**. So the sequence is:

    1. Try `gpt-4o-mini` → fail → retry up to 2× → still fails
    2. Try `groq/llama-3.1-8b-instant` → fail → retry up to 2× → still fails
    3. Try `gemini-2.0-flash` → ✓ success

### 3. `FallbackMiddleware` — composable middleware

Drop `FallbackMiddleware` into any existing middleware stack:

```python
from llmgate import LLMGate
from llmgate.middleware import RetryMiddleware, FallbackMiddleware

gate = LLMGate(middleware=[
    RetryMiddleware(max_retries=2),
    FallbackMiddleware(
        models=["groq/llama-3.1-8b-instant", "gemini-2.0-flash"],
    ),
])

resp = gate.completion("gpt-4o-mini", messages)
```

!!! note "Middleware coverage on fallback models"
    With `FallbackMiddleware`, the primary model goes through the full middleware stack. Fallback models are called directly (bypassing middleware) to avoid recursive chains. Use `LLMGate(fallback_chain=[...])` if you need full middleware coverage on every candidate.

---

## Customising `fallback_on`

By default, fallback triggers on:

```python
(RateLimitError, ProviderAPIError, AuthError)
```

Override this with the `fallback_on` parameter:

```python
from llmgate import completion
from llmgate.exceptions import RateLimitError

# Only fall back on rate limits — auth errors propagate immediately
resp = completion(
    model=["gpt-4o-mini", "groq/llama-3.1-8b-instant"],
    messages=messages,
    fallback_on=(RateLimitError,),
)
```

Or configure it on the gate:

```python
gate = LLMGate(
    fallback_chain=["gpt-4o-mini", "groq/llama-3.1-8b-instant"],
    fallback_on=(RateLimitError,),
)
```

---

## Handling total failure

When every model in the chain fails, `AllProvidersFailedError` is raised. It contains the full `(model, exception)` list for diagnostics:

```python
from llmgate import completion
from llmgate.exceptions import AllProvidersFailedError

try:
    resp = completion(
        model=["gpt-4o-mini", "groq/llama-3.1-8b-instant"],
        messages=messages,
    )
except AllProvidersFailedError as e:
    print(f"All {len(e.errors)} providers failed:")
    for model, exc in e.errors:
        print(f"  {model}: {type(exc).__name__}: {exc}")
```

---

## Streaming

!!! warning "Not supported with model lists"
    `stream=True` cannot be combined with a model list. Streaming fallback is planned for v0.7.

    ```python
    # ❌ raises ValueError
    completion(model=["gpt-4o-mini", "groq/llama-3.1-8b-instant"], messages=messages, stream=True)

    # ✓ streaming works normally with a single model
    for chunk in completion("gpt-4o-mini", messages, stream=True):
        print(chunk.delta, end="")
    ```

---

## Reference

| Parameter | Where | Description |
|---|---|---|
| `model: list[str]` | `completion()`, `acompletion()` | Ordered fallback chain |
| `fallback_on` | `completion()`, `acompletion()`, `LLMGate()` | Exception types that trigger fallback. Default: `(RateLimitError, ProviderAPIError, AuthError)` |
| `fallback_chain` | `LLMGate()` | App-level fallback chain with full middleware per candidate |
| `FallbackMiddleware(models=[...])` | `LLMGate(middleware=[...])` | Composable middleware fallback |
| `resp.fallback_attempts` | `CompletionResponse` | Models tried before this response |
| `AllProvidersFailedError` | exception | Raised when all models fail; `.errors` is `list[tuple[str, Exception]]` |
