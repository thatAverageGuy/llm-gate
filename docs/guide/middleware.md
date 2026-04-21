# Middleware

Middleware wraps every completion call with cross-cutting concerns — retry, caching, logging, rate-limiting — without touching your application logic.

---

## Using middleware

Pass a list of middleware to any `completion()` call:

```python
from llmgate import completion
from llmgate.middleware import RetryMiddleware, LoggingMiddleware

resp = completion(
    "gpt-4o-mini",
    messages,
    middleware=[RetryMiddleware(max_retries=3), LoggingMiddleware()],
)
```

Or configure it once on a `LLMGate` instance:

```python
from llmgate import LLMGate
from llmgate.middleware import RetryMiddleware, CacheMiddleware, LoggingMiddleware, RateLimitMiddleware

gate = LLMGate(middleware=[
    RetryMiddleware(max_retries=3, backoff_factor=0.5),
    CacheMiddleware(ttl=300),
    LoggingMiddleware(level="INFO"),
    RateLimitMiddleware(rpm=60),
])

resp = gate.completion("gpt-4o-mini", messages)
resp = await gate.acompletion("gemini-2.5-flash-lite", messages)

for chunk in gate.stream("groq/llama-3.1-8b-instant", messages):
    print(chunk.delta, end="")

resp = gate.embed("text-embedding-3-small", "Hello")
results = gate.batch(requests, max_concurrency=5)
```

Middleware is applied in **list order** (first in, first out for pre-hooks; last in, first out for post-hooks).

---

## Built-in middleware

### `RetryMiddleware`

Retries on `RateLimitError` and `ProviderAPIError` with exponential backoff.

```python
RetryMiddleware(
    max_retries=3,       # number of retry attempts
    backoff_factor=0.5,  # sleep = backoff_factor * 2^attempt
)
```

### `CacheMiddleware`

In-memory LRU cache keyed on (model, messages, parameters). Skips the provider entirely on cache hits.

```python
CacheMiddleware(
    ttl=300,    # seconds before cached responses expire
)
```

### `LoggingMiddleware`

Logs each request (model, provider, token usage) using the standard Python `logging` module.

```python
LoggingMiddleware(
    level="INFO",   # "DEBUG" | "INFO" | "WARNING"
)
```

### `RateLimitMiddleware`

Client-side rate limit — sleeps before sending if the per-minute request budget is exhausted.

```python
RateLimitMiddleware(
    rpm=60,   # max requests per minute
)
```

---

## Writing a custom middleware

Subclass `BaseMiddleware` and override `before_call` and/or `after_call`:

```python
import time
import logging
from llmgate.middleware import BaseMiddleware
from llmgate.types import CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)

class TimingMiddleware(BaseMiddleware):
    def before_call(self, request: CompletionRequest) -> CompletionRequest:
        request._start = time.perf_counter()   # stash timestamp
        return request

    def after_call(
        self,
        request: CompletionRequest,
        response: CompletionResponse,
    ) -> CompletionResponse:
        elapsed = time.perf_counter() - request._start
        logger.info(
            "%(provider)s/%(model)s → %(tokens)d tokens in %(elapsed).2fs",
            dict(
                provider=response.provider,
                model=response.model,
                tokens=response.usage.total_tokens,
                elapsed=elapsed,
            ),
        )
        return response
```

Then use it:

```python
gate = LLMGate(middleware=[TimingMiddleware()])
resp = gate.completion("gpt-4o-mini", messages)
```
