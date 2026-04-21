# Mistral

## Setup

```bash
pip install llmgate[mistral]
export MISTRAL_API_KEY="..."
```

---

## Model prefix

Always prefix with `mistral/`:

```python
completion("mistral/mistral-large-latest", messages)
completion("mistral/mistral-small-latest", messages)
completion("mistral/open-mistral-7b", messages)
completion("mistral/pixtral-12b-2409", messages)  # vision
```

---

## Vision

Mistral's `image_url` is a plain string (not an object). llmgate handles this wire format difference automatically:

```python
completion("mistral/pixtral-12b-2409", messages=[{
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this."},
        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
    ],
}])
```
