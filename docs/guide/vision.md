# Vision & Images

llmgate v0.5.0 adds first-class multimodal support. Pass images alongside text using a unified content-parts API — the correct wire format for each provider is handled automatically.

!!! tip "Backward compatible"
    All existing text-only code continues to work unchanged. Vision is purely additive.

---

## How it works

`Message.content` accepts either:

- A plain `str` — the existing text-only interface
- A `list` of **`TextPart`** and **`ImagePart`** objects — the new multimodal interface

```python
from llmgate.types import TextPart, ImagePart, ImageURL, ImageBytes, Message
```

---

## Image Types

### `ImageURL` — URL-based image

```python
ImageURL(
    url="https://example.com/photo.jpg",
    detail="auto",   # optional: "auto" | "low" | "high" (OpenAI/Azure only)
)
```

Accepts:

- **`https://`** public URLs
- **`data:image/jpeg;base64,...`** data URIs

### `ImageBytes` — Inline base64

```python
ImageBytes(
    data="<base64-encoded-bytes>",   # no data-URI prefix
    mime_type="image/jpeg",          # "image/jpeg" | "image/png" | "image/webp" | "image/gif"
)
```

---

## Usage

### URL image

```python
from llmgate import completion

resp = completion(
    "gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text",      "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
        ],
    }],
)
print(resp.text)
```

!!! note "Dict format"
    You can pass content parts as plain dicts (OpenAI format) or as typed `TextPart`/`ImagePart` objects — both work.

### Base64 image

```python
import base64
from llmgate import completion
from llmgate.types import TextPart, ImagePart, ImageBytes, Message

with open("photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = completion(
    "claude-opus-4-7",
    messages=[Message(
        role="user",
        content=[
            ImagePart(type="image_bytes", image_bytes=ImageBytes(data=b64, mime_type="image/jpeg")),
            TextPart(text="Describe this image in detail."),
        ],
    )],
)
print(resp.text)
```

### Multiple images

```python
resp = completion(
    "gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text",      "text": "What's the difference between these two images?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}},
        ],
    }],
)
```

### With detail hint (OpenAI / Azure)

```python
resp = completion(
    "gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text",      "text": "Read the text in this image."},
            {"type": "image_url", "image_url": {"url": "https://...", "detail": "high"}},
        ],
    }],
)
```

The `detail` field is silently ignored by providers that don't support it.

---

## Provider Support Matrix

| Provider | URL images | Base64 / bytes | Notes |
|---|---|---|---|
| **OpenAI** | ✅ | ✅ | `detail` param supported |
| **Azure OpenAI** | ✅ | ✅ | Identical to OpenAI |
| **Anthropic** | ✅ | ✅ | Up to 100 images/request |
| **Gemini** | ✅ | ✅ | URLs fetched client-side → inline bytes |
| **Groq** | ✅ | ✅ | Vision model required (e.g. `llama-4-scout-17b`) |
| **Mistral** | ✅ | ✅ | Wire format difference handled automatically |
| **Bedrock** | ✅ | ✅ | URLs fetched client-side; raw bytes to Converse API |
| **Ollama** | ⚠️ | ✅ | URL images are fetched client-side automatically |
| **Cohere** | ❌ | ❌ | Raises `VisionNotSupported` |

---

## Provider-Specific Notes

### Gemini

Gemini does not accept image URLs natively. llmgate fetches the image bytes via `httpx` and sends them inline — this happens automatically and is transparent to you.

```python
# Just works — llmgate fetches the URL for you
resp = completion(
    "gemini-2.0-flash",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text",      "text": "Describe this."},
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
        ],
    }],
)
```

### Ollama

Ollama uses a top-level `images` list on the message, not a content-parts array. llmgate handles this translation automatically.

```python
resp = completion(
    "ollama/llava",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text",      "text": "What's this?"},
            {"type": "image_url", "image_url": {"url": "https://..."}},
        ],
    }],
)
```

### Cohere

Cohere's vision API is not yet stable. llmgate raises `VisionNotSupported` if you attempt to send image content to a Cohere model.

```python
from llmgate.exceptions import VisionNotSupported

try:
    resp = completion("cohere/command-r-plus", messages_with_image)
except VisionNotSupported:
    # Fall back to a different provider
    resp = completion("gpt-4o-mini", messages_with_image)
```

---

## Full Typed Example

```python
import base64
from pathlib import Path
from llmgate import completion
from llmgate.types import TextPart, ImagePart, ImageURL, ImageBytes, Message

# Load a local image
image_data = base64.b64encode(Path("chart.png").read_bytes()).decode()

resp = completion(
    "gpt-4o",
    messages=[
        Message(role="system", content="You are a data analyst."),
        Message(
            role="user",
            content=[
                TextPart(text="Summarize the trend shown in this chart."),
                ImagePart(
                    type="image_bytes",
                    image_bytes=ImageBytes(data=image_data, mime_type="image/png"),
                ),
            ],
        ),
    ],
    max_tokens=300,
)

print(resp.text)
```
