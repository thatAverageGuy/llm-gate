# Migrating from LiteLLM

llmgate is a lighter, more opinionated alternative to LiteLLM. Here's how to migrate.

## API comparison

| LiteLLM | llmgate | Notes |
|---|---|---|
| `litellm.completion(model, messages)` | `llmgate.completion(model, messages)` | Identical call shape |
| `litellm.acompletion(...)` | `llmgate.acompletion(...)` | Identical |
| `response["choices"][0]["message"]["content"]` | `response.text` | Pydantic model, not dict |
| `response["usage"]["total_tokens"]` | `response.usage.total_tokens` | Attribute access |
| `groq/llama-3.1-8b` | `groq/llama-3.1-8b` | Same prefix convention |
| `litellm.exceptions.AuthenticationError` | `llmgate.exceptions.AuthError` | Renamed, slimmer hierarchy |
| `litellm.embedding(model, input)` | `llmgate.embed(model, input)` | Similar shape |

---

## Find and replace

```python
# Before
import litellm
response = litellm.completion(model, messages)
text = response["choices"][0]["message"]["content"]
tokens = response["usage"]["total_tokens"]

# After
from llmgate import completion
response = completion(model, messages)
text = response.text
tokens = response.usage.total_tokens
```

---

## Error handling

```python
# Before (LiteLLM)
from litellm.exceptions import AuthenticationError, RateLimitError
try:
    ...
except AuthenticationError:
    ...
except RateLimitError:
    ...

# After (llmgate)
from llmgate.exceptions import AuthError, RateLimitError
try:
    ...
except AuthError:
    ...
except RateLimitError:
    ...
```

---

## What llmgate doesn't do

| Feature | Status |
|---|---|
| Gateway / proxy server mode | ❌ Out of scope — use a dedicated gateway |
| `LITELLM_LOG` environment variable | ❌ Use `LoggingMiddleware` instead |
| Fallbacks via `litellm.completion(fallbacks=[...])` | ❌ Implement via try/except or a wrapper |
| Cost tracking | ❌ Not yet — planned |
| 100+ providers | ❌ 9 providers, all production-quality |

llmgate trades breadth for depth — fewer providers, all tested, all typed.
