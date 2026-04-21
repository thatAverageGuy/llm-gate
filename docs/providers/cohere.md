# Cohere

## Setup

```bash
pip install llmgate[cohere]
export COHERE_API_KEY="..."
```

---

## Model prefix

Always prefix with `cohere/`:

```python
completion("cohere/command-r-plus", messages)
completion("cohere/command-r", messages)
completion("cohere/command-a-03-2025", messages)
```

---

## Embeddings

Cohere has excellent embedding models:

```python
from llmgate import embed

resp = embed("cohere/embed-english-v3.0", "Hello world")
resp = embed("cohere/embed-multilingual-v3.0", "Bonjour monde")
```

---

## Vision

!!! warning "Not supported"
    Cohere's vision API is not yet stable. Passing image content to a Cohere model raises `VisionNotSupported`.

    ```python
    from llmgate.exceptions import VisionNotSupported

    try:
        resp = completion("cohere/command-r-plus", messages_with_images)
    except VisionNotSupported:
        resp = completion("gpt-4o-mini", messages_with_images)  # fallback
    ```
