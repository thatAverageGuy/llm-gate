# OpenAI

## Setup

```bash
export OPENAI_API_KEY="sk-..."
```

No extra install — included with `pip install llmgate`.

---

## Model prefixes

Models starting with `gpt-`, `o1-`, `o3-`, or `chatgpt-` are routed to OpenAI automatically.

```python
completion("gpt-4o", messages)
completion("gpt-4o-mini", messages)
completion("o3-mini", messages)
completion("o1-preview", messages)
```

---

## Vision

OpenAI supports both URL and base64 images with an optional `detail` hint:

```python
completion("gpt-4o", messages=[{
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "https://...", "detail": "high"}},
    ],
}])
```

`detail` values: `"auto"` (default) · `"low"` (85 tokens) · `"high"` (full resolution, more tokens)

---

## Structured outputs

Uses native `json_schema` with `strict: true` — the most reliable structured output mode available:

```python
from llmgate import parse

movie = parse("gpt-4o-mini", messages, response_format=Movie)
```

---

## Notes

- `o1-*` and `o3-*` reasoning models do not support `system` messages or `temperature`. Pass as `extra_kwargs` or omit.
- `frequency_penalty` and `presence_penalty` are passed through via `**extra_kwargs`.
