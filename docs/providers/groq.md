# Groq

## Setup

```bash
export GROQ_API_KEY="gsk_..."
```

No extra install — included with `pip install llmgate`.

---

## Model prefix

Always prefix with `groq/`:

```python
completion("groq/llama-3.3-70b-versatile", messages)
completion("groq/llama-3.1-8b-instant", messages)
completion("groq/gemma2-9b-it", messages)
completion("groq/meta-llama/llama-4-scout-17b-16e-instruct", messages)  # vision
```

---

## Vision

Groq supports vision on specific models (e.g. `llama-4-scout-17b`). The `detail` param is not supported and is stripped automatically:

```python
completion("groq/meta-llama/llama-4-scout-17b-16e-instruct", messages=[{
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://..."}},
    ],
}])
```

---

## Notes

- Groq is OpenAI-compatible — the fastest inference available for open-weight models.
- `EmbeddingsNotSupported` is raised if you call `embed()` on a Groq model.
