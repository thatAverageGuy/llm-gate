# Ollama

## Setup

```bash
pip install llmgate[ollama]
# Start Ollama locally
ollama pull llama3.2
ollama pull llava   # vision model
```

Default host: `http://localhost:11434`. Override with:

```bash
export OLLAMA_HOST="http://my-ollama-server:11434"
```

---

## Model prefix

Always prefix with `ollama/`:

```python
completion("ollama/llama3.2", messages)
completion("ollama/gemma3", messages)
completion("ollama/llava", messages)          # vision
completion("ollama/deepseek-r1:7b", messages)
```

---

## Vision

Ollama uses a top-level `images` field on the message (not content parts). llmgate handles this automatically. URL images are fetched client-side:

```python
completion("ollama/llava", messages=[{
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
    ],
}])
```

!!! tip "Prefer base64 for local models"
    For best performance with Ollama, use `image_bytes` with base64 to avoid network latency from URL fetching.

---

## Embeddings

```python
from llmgate import embed

resp = embed("ollama/nomic-embed-text", "Hello world")
resp = embed("ollama/mxbai-embed-large", "Hello world")
```

---

## Notes

- No API key required — Ollama runs locally.
- Temperature, max tokens etc. are sent via Ollama's `options` dict — handled automatically.
