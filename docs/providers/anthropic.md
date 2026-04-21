# Anthropic

## Setup

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

No extra install — included with `pip install llmgate`.

---

## Model prefix

Models starting with `claude-` are routed to Anthropic automatically.

```python
completion("claude-opus-4-7", messages)
completion("claude-3-5-sonnet-20241022", messages)
completion("claude-3-5-haiku-20241022", messages)
completion("claude-3-haiku-20240307", messages)
```

---

## System messages

System messages are automatically extracted from the messages list and sent via Anthropic's `system` parameter:

```python
completion("claude-3-5-sonnet-20241022", [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user",   "content": "Explain gravity."},
])
# ↑ "system" is extracted and sent correctly — no special handling needed
```

---

## Vision

Anthropic supports URL and base64 images. Up to 20MB per image, 100 images per request:

```python
completion("claude-opus-4-7", messages=[{
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
    ],
}])
```

---

## Notes

- `max_tokens` is **required** by the Anthropic API. llmgate defaults to `1024` if not provided.
- `top_k` is supported as an extra kwarg.
- `stream=True` works with `astream()`.
