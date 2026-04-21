# Batch Completions

Run many completion requests in parallel with built-in concurrency control and error isolation.

---

## Basic usage

```python
from llmgate import batch

requests = [
    {"model": "gpt-4o-mini",              "messages": [{"role": "user", "content": "What is 1+1?"}]},
    {"model": "groq/llama-3.1-8b-instant","messages": [{"role": "user", "content": "What is 2+2?"}]},
    {"model": "gemini-2.5-flash-lite",    "messages": [{"role": "user", "content": "What is 3+3?"}]},
]

results = batch(requests, max_concurrency=3)

for resp in results.results:
    if resp is not None:
        print(f"[{resp.provider}] {resp.text}")
```

---

## Async

```python
from llmgate import abatch

results = await abatch(requests, max_concurrency=10)
```

---

## The `BatchResult`

```python
results.results         # list[CompletionResponse | None] — same order as input
results.errors          # list[BatchError] — one per failed request
results.successful      # int
results.failed          # int
results.total_tokens    # int — aggregate across all successes
results.success_rate    # float — 0.0 to 1.0
```

---

## Handling failures

Each request fails independently — one bad call doesn't abort the rest:

```python
results = batch(requests, max_concurrency=5)

for i, resp in enumerate(results.results):
    if resp is None:
        err = next(e for e in results.errors if e.index == i)
        print(f"Request {i} failed: {err.error_type} — {err.error}")
    else:
        print(f"Request {i}: {resp.text[:50]}")
```

---

## Fail fast

Stop the entire batch on the first error:

```python
results = batch(requests, fail_fast=True)
```

---

## With middleware

```python
from llmgate.middleware import RetryMiddleware, LoggingMiddleware

results = batch(
    requests,
    max_concurrency=5,
    middleware=[RetryMiddleware(max_retries=2), LoggingMiddleware()],
)
```

---

## Via `LLMGate` (inherits configured middleware)

```python
from llmgate import LLMGate
from llmgate.middleware import RetryMiddleware

gate = LLMGate(middleware=[RetryMiddleware(max_retries=3)])

results = gate.batch(requests, max_concurrency=5)
results = await gate.abatch(requests, max_concurrency=5)
```

---

## Using `CompletionRequest` objects

```python
from llmgate.types import CompletionRequest, Message

requests = [
    CompletionRequest(
        model="gpt-4o-mini",
        messages=[Message(role="user", content="Hello!")],
        max_tokens=50,
        temperature=0.7,
    ),
    CompletionRequest(
        model="claude-3-5-haiku-20241022",
        messages=[Message(role="user", content="Hi there!")],
    ),
]

results = batch(requests, max_concurrency=2)
```
