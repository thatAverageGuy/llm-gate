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

Cohere has excellent embedding models with task-aware vectors.

!!! important "Always set `input_type`"
    The default (`search_document`) is wrong for query embedding. Use the correct type for production RAG.

```python
from llmgate import embed

# Embedding corpus documents:
resp = embed("cohere/embed-english-v3.0", chunks,
             input_type="search_document")

# Embedding a search query:
resp = embed("cohere/embed-english-v3.0", query,
             input_type="search_query")

# Multilingual:
resp = embed("cohere/embed-multilingual-v3.0", texts,
             input_type="search_document")

# Truncation strategy for long inputs:
resp = embed("cohere/embed-english-v3.0", long_text,
             input_type="search_document",
             truncate="END")   # "NONE" (error) | "START" | "END"
```

Supported `input_type` values: `"search_document"`, `"search_query"`, `"classification"`, `"clustering"`.

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
