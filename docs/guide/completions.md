# Completions

The `completion()` / `acompletion()` functions are the heart of llmgate. They accept messages in the standard OpenAI format and return a consistent `CompletionResponse` regardless of provider.

---

## Signature

```python
from llmgate import completion, acompletion

resp = completion(
    model: str,
    messages: list[dict | Message],
    *,
    provider: str | None = None,
    api_key: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stream: bool = False,
    tools: list[ToolDefinition] | None = None,
    tool_choice: str | dict | None = None,
    response_format: type[BaseModel] | None = None,
    middleware: list[BaseMiddleware] | None = None,
    **extra_kwargs,
)
```

All parameters after `messages` are keyword-only.

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `model` | `str` | Model name — provider is auto-detected from the prefix |
| `messages` | `list` | List of message dicts or `Message` objects |
| `provider` | `str` | Override auto-detection: `"openai"`, `"anthropic"`, `"gemini"`, etc. |
| `api_key` | `str` | Override the env-var API key for this call only |
| `max_tokens` | `int` | Maximum tokens to generate |
| `temperature` | `float` | Sampling temperature `0.0–2.0` |
| `top_p` | `float` | Nucleus sampling threshold |
| `stream` | `bool` | Return streaming chunks instead of a full response |
| `tools` | `list` | Tool/function definitions for function calling |
| `tool_choice` | `str\|dict` | `"auto"`, `"none"`, or a specific tool name |
| `response_format` | `type[BaseModel]` | Pydantic class — enables structured output |
| `middleware` | `list` | Per-call middleware stack |
| `**extra_kwargs` | any | Forwarded verbatim to the underlying SDK |

---

## The Response Object

```python
resp = completion("gpt-4o-mini", messages)

resp.text                       # str — first choice content (shortcut)
resp.id                         # str — provider response ID
resp.model                      # str — model name as sent
resp.provider                   # str — "openai" | "anthropic" | ...
resp.choices                    # list[Choice]
resp.choices[0].message.role    # "assistant"
resp.choices[0].message.content # str | list[TextPart | ImagePart]
resp.choices[0].finish_reason   # "stop" | "tool_calls" | "length" | ...
resp.usage.prompt_tokens        # int
resp.usage.completion_tokens    # int
resp.usage.total_tokens         # int
resp.parsed                     # BaseModel | None (when response_format set)
resp.raw                        # raw SDK response (escape hatch)
```

---

## Examples

### Basic call

```python
from llmgate import completion

resp = completion(
    "gpt-4o-mini",
    [{"role": "user", "content": "Explain gravity in one sentence."}],
)
print(resp.text)
```

### With system message and parameters

```python
resp = completion(
    "claude-3-5-sonnet-20241022",
    messages=[
        {"role": "system", "content": "You are a concise technical writer."},
        {"role": "user",   "content": "What is a neural network?"},
    ],
    max_tokens=150,
    temperature=0.3,
)
print(resp.text)
print(f"Used {resp.usage.total_tokens} tokens")
```

### Multi-turn conversation

```python
from llmgate.types import Message

history = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user",   content="My name is Alice."),
]

resp = completion("gemini-2.5-flash-lite", history)
history.append(Message(role="assistant", content=resp.text))

history.append(Message(role="user", content="What is my name?"))
resp = completion("gemini-2.5-flash-lite", history)
print(resp.text)  # "Your name is Alice."
```

### Provider-specific parameters

Extra kwargs are forwarded verbatim to the underlying SDK:

```python
# OpenAI: frequency and presence penalties
resp = completion("gpt-4o", messages, frequency_penalty=0.5, presence_penalty=0.2)

# Anthropic: top_k
resp = completion("claude-3-haiku-20240307", messages, top_k=40)

# Groq: stop sequences
resp = completion("groq/llama-3.1-8b-instant", messages, stop=["END"])
```

### Token usage monitoring

```python
resp = completion("gpt-4o-mini", messages)
print(f"Prompt:     {resp.usage.prompt_tokens}")
print(f"Completion: {resp.usage.completion_tokens}")
print(f"Total:      {resp.usage.total_tokens}")
```

### Raw response access

```python
resp = completion("gpt-4o-mini", messages)
raw = resp.raw   # the raw openai.ChatCompletion object
print(raw.system_fingerprint)
```
