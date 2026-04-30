# Streaming

Pass `stream=True` to receive response tokens as they are generated rather than waiting for the full completion.

---

## Sync Streaming

```python
from llmgate import completion

for chunk in completion("gpt-4o-mini", messages, stream=True):
    print(chunk.delta, end="", flush=True)
print()  # newline at end
```

`chunk` is a `StreamChunk` with a single `delta: str` field containing the incremental text.

---

## Async Streaming

```python
import asyncio
from llmgate import acompletion

async def stream_response():
    async for chunk in await acompletion(
        "groq/llama-3.3-70b-versatile",
        messages,
        stream=True,
    ):
        print(chunk.delta, end="", flush=True)
    print()

asyncio.run(stream_response())
```

---

## Collecting the full response

```python
chunks = []
for chunk in completion("gpt-4o-mini", messages, stream=True):
    chunks.append(chunk.delta)
    print(chunk.delta, end="", flush=True)

full_text = "".join(chunks)
```

---

## Streaming through middleware

Streaming works seamlessly with `LLMGate` middleware:

```python
from llmgate import LLMGate
from llmgate.middleware import LoggingMiddleware, RetryMiddleware

gate = LLMGate(middleware=[
    RetryMiddleware(max_retries=3),
    LoggingMiddleware(),
])

for chunk in gate.stream("claude-3-5-haiku-20241022", messages):
    print(chunk.delta, end="", flush=True)
```

---

## Streaming Fallback

Streaming works seamlessly with model lists, providing mid-stream resilience out of the box:

```python
resp = completion(
    model=["gpt-4o-mini", "groq/llama-3.1-8b-instant", "gemini-2.0-flash"],
    messages=messages,
    stream=True,
    stream_fallback_mode="prefill", # Strategies: "restart", "prefill", "user_turn"
)

for chunk in resp:
    print(chunk.delta, end="", flush=True)
```

See the [Fallback Routing](./fallback.md#streaming) guide for details on the different mid-stream recovery strategies (`restart`, `prefill`, and `user_turn`) and observability metadata.

---

!!! warning "Incompatibility"
    `stream=True` and `response_format=` (structured outputs) cannot be used together.
    Streaming returns raw incremental text; structured outputs require the complete response for Pydantic validation.
