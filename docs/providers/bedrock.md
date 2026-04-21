# AWS Bedrock

## Setup

```bash
pip install llmgate[bedrock]
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
```

Standard AWS credential resolution is used — IAM roles, instance profiles, and `~/.aws/credentials` all work.

---

## Model prefix

Always prefix with `bedrock/` followed by the full Bedrock model ID:

```python
completion("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0", messages)
completion("bedrock/amazon.nova-pro-v1:0", messages)
completion("bedrock/meta.llama3-70b-instruct-v1:0", messages)
```

---

## Vision

Bedrock's Converse API uses raw bytes for images (not base64 strings). llmgate handles conversion automatically — URL images are fetched client-side:

```python
completion("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0", messages=[{
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
    ],
}])
```

---

## Notes

- Uses the Converse API — consistent across all Bedrock models.
- Tool calls work via `toolUse` / `toolResult` Converse blocks, handled automatically.
- Embeddings supported via Titan models: `bedrock/amazon.titan-embed-text-v2:0`.
