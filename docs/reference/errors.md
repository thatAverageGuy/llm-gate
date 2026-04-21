# Error Handling

All llmgate exceptions inherit from `LLMGateError`, so you can catch broadly or narrow down to specific cases.

## Exception hierarchy

```
LLMGateError
├── ProviderError           — upstream provider rejected the call
│   ├── AuthError           — 401 / invalid API key
│   ├── RateLimitError      — 429 / quota exceeded
│   └── ProviderAPIError    — other 4xx/5xx
├── ModelNotFoundError      — unknown model string
├── ConfigError             — missing env var / bad config
├── StreamingNotSupported   — provider doesn't support streaming
├── EmbeddingsNotSupported  — provider doesn't have an embeddings API
├── VisionNotSupported      — provider doesn't support image inputs
└── BatchTimeoutError       — a single batch request timed out
```

---

## Usage

```python
from llmgate import completion
from llmgate.exceptions import (
    LLMGateError,
    AuthError,
    RateLimitError,
    ProviderAPIError,
    ModelNotFoundError,
    ConfigError,
    EmbeddingsNotSupported,
    VisionNotSupported,
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
except VisionNotSupported as e:
    print(f"{e.provider} doesn't support images — use a different provider")
except LLMGateError as e:
    # Catch-all for anything else from llmgate
    print(f"llmgate error: {e}")
```

---

## Exception fields

| Field | Available on | Description |
|---|---|---|
| `e.provider` | `ProviderError` subclasses, `VisionNotSupported`, `EmbeddingsNotSupported` | Provider name string |
| `e.model` | `ModelNotFoundError` | The unrecognised model string |
| `e.index` | `BatchTimeoutError` | Index of the failing request in the batch |
| `e.timeout` | `BatchTimeoutError` | Timeout value in seconds |

---

## Retry pattern

```python
import time
from llmgate import completion
from llmgate.exceptions import RateLimitError, ProviderAPIError

def completion_with_retry(model, messages, retries=3, **kwargs):
    for attempt in range(retries):
        try:
            return completion(model, messages, **kwargs)
        except (RateLimitError, ProviderAPIError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)   # 1s, 2s, 4s
```

Or use the built-in `RetryMiddleware` — it handles this automatically.

---

## Vision fallback pattern

```python
from llmgate import completion
from llmgate.exceptions import VisionNotSupported

def vision_completion(preferred_model, messages_with_images, fallback="gpt-4o-mini"):
    try:
        return completion(preferred_model, messages_with_images)
    except VisionNotSupported:
        return completion(fallback, messages_with_images)
```
