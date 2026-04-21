# Azure OpenAI

## Setup

No extra install — uses the `openai` SDK already bundled.

```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://myinstance.openai.azure.com"
export AZURE_OPENAI_API_VERSION="2024-02-01"   # optional default
```

---

## Model prefix

Always prefix with `azure/` followed by your **deployment name**:

```python
completion("azure/my-gpt4-deployment", messages)
completion("azure/my-gpt4o-mini", messages)
```

The deployment name (everything after `azure/`) is sent directly to Azure.

---

## Vision

Identical to OpenAI — URL and base64 images both work, `detail` param supported:

```python
completion("azure/my-gpt4o-deployment", messages=[{
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "https://...", "detail": "high"}},
    ],
}])
```

---

## Notes

- `AZURE_OPENAI_API_VERSION` defaults to `2024-02-01` if not set.
- Both `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` are required — authentication fails with a clear `AuthError` if either is missing.
