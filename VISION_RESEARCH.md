# Vision / Multimodal API Research

> Researched: 2026-04-21  
> Sources: Official provider docs, verified against live documentation.

---

## Summary Table

| Provider | Vision Support | URL images | Base64 images | Multi-image | Wire format |
|---|---|---|---|---|---|
| **OpenAI** | ✅ All GPT-4o, o-series | ✅ | ✅ | ✅ up to context limit | `image_url` content part |
| **Anthropic** | ✅ All claude-3+ models | ✅ | ✅ | ✅ up to 100/req | `image` content block |
| **Gemini** | ✅ All gemini-1.5+ models | ✅ (inline fetch) | ✅ | ✅ | `inline_data` / File API |
| **Groq** | ✅ llama-4-scout-17b only | ✅ | ✅ data URL | ✅ up to 5 | `image_url` (OpenAI compat) |
| **Mistral** | ✅ mistral-small, medium, large + Ministral | ✅ | ✅ | ✅ | `image_url` (OpenAI compat) |
| **Cohere** | ⚠️ Command A Vision only (`command-a-vision-07-2025`) | ❓ | ✅ base64 | ✅ up to 20 | Custom format |
| **Azure OpenAI** | ✅ Same as OpenAI (uses openai SDK) | ✅ | ✅ | ✅ | Identical to OpenAI |
| **Bedrock** | ✅ Claude 3+ via Converse API | ✅ (S3 URLs) | ✅ bytes | ✅ | `image` content block (Converse) |
| **Ollama** | ✅ llava, bakllava, llama3.2-vision, etc. | ❌ | ✅ base64 list | ✅ | `images` field in message |

---

## 1. OpenAI

**Source:** https://platform.openai.com/docs/guides/vision  
**Models:** All `gpt-4o`, `gpt-4o-mini`, `o1`, `o3`, `gpt-4-turbo` series

### Wire format

```json
{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "What's in this image?"
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "https://example.com/image.jpg",
        "detail": "auto"
      }
    }
  ]
}
```

**Base64 variant:**
```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/jpeg;base64,<base64_string>",
    "detail": "low"
  }
}
```

### Key facts
- `detail` param: `"auto"` (default), `"low"` (512×512, ~85 tokens), `"high"` (tiled crops, more tokens)
- Supported MIME: `image/jpeg`, `image/png`, `image/webp`, `image/gif` (non-animated)
- Max image size: 20MB per image URL; base64 limit depends on request size
- Multiple images: Include multiple `image_url` parts in the `content` array
- SDK call: `client.chat.completions.create(...)` — content array replaces plain string

### SDK snippet (Python)
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": "https://...", "detail": "auto"}},
        ],
    }],
)
```

---

## 2. Anthropic (Claude)

**Source:** https://docs.anthropic.com/en/docs/build-with-claude/vision  
**Models:** All `claude-3-*`, `claude-3-5-*`, `claude-opus-4-7`, `claude-sonnet-4-*`, etc.

### Wire format (base64)

```json
{
  "role": "user",
  "content": [
    {
      "type": "image",
      "source": {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "<base64_string>"
      }
    },
    {
      "type": "text",
      "text": "Describe this image."
    }
  ]
}
```

**URL variant:**
```json
{
  "type": "image",
  "source": {
    "type": "url",
    "url": "https://example.com/image.jpg"
  }
}
```

**Files API variant (beta):**
```json
{
  "type": "image",
  "source": {
    "type": "file",
    "file_id": "file_abc123"
  }
}
```
> Files API requires `betas=["files-api-2025-04-14"]` header.

### Key facts
- Supported MIME: `image/jpeg`, `image/png`, `image/gif`, `image/webp`
- Max images per request: **100** (200k context models) or **600** (others), but 20 per message on claude.ai
- Max dimensions: 8000×8000px (reduced to 2000×2000 if >20 images)
- Token cost: `width * height / 750` tokens approximately
- Claude Opus 4.7: supports high-res up to 2576px long edge (~4784 tokens max)
- Best practice: put images **before** text in the content array
- `message.content` must be a list (not a string) when including images

### SDK snippet (Python)
```python
import base64, httpx

image_data = base64.standard_b64encode(httpx.get(url).content).decode("utf-8")

message = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data},
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }],
)
```

---

## 3. Google Gemini

**Source:** https://ai.google.dev/gemini-api/docs/vision  
**Models:** `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3-flash-preview`, etc.

### Wire format (inline, google-genai SDK)

```python
from google import genai
from google.genai import types

client = genai.Client()

# From bytes (inline)
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        "Describe this image.",
    ],
)
```

### REST wire format (inline_data)
```json
{
  "contents": [{
    "parts": [
      {
        "inline_data": {
          "mime_type": "image/jpeg",
          "data": "<base64_string>"
        }
      },
      {"text": "Caption this image."}
    ]
  }]
}
```

### File API variant (for large/reused images)
```python
my_file = client.files.upload(file="path/to/sample.jpg")
response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=[my_file, "Caption this image."],
)
```

### Key facts
- Supported MIME: `image/png`, `image/jpeg`, `image/webp`, `image/heic`, `image/heif`
- Inline limit: total request < 20MB
- File API: for files > 20MB or reuse across requests
- Multiple images: list multiple `types.Part` objects in `contents`
- The SDK accepts both inline bytes and File API URIs interchangeably
- URL images: must fetch bytes first and pass inline (Gemini has no native URL-reference mode like OpenAI)

---

## 4. Groq

**Source:** https://console.groq.com/docs/vision  
**Models:** `meta-llama/llama-4-scout-17b-16e-instruct` (currently in preview; only vision model on Groq)

### Wire format (OpenAI-compatible)

```python
from groq import Groq

client = Groq()

# URL-based
completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        ],
    }],
)

# Base64-based (data URI)
completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    }],
)
```

### Key facts
- Wire format: **identical to OpenAI** (`image_url` content part)
- Only one vision model available currently: `llama-4-scout-17b-16e-instruct` (preview)
- Max images: **5 per request**
- Max URL image size: **20MB**
- Max base64 image size: **4MB** per image
- Max resolution: **33 megapixels** (33,177,600 total pixels) per image
- Supports multi-turn conversations, tool use, and JSON mode with images
- Implementation note: since Groq uses the OpenAI SDK under the hood with a custom base URL, the existing Groq provider just needs its message serializer updated

---

## 5. Mistral

**Source:** https://docs.mistral.ai/capabilities/vision/  
**Models:** `mistral-large-2512`, `mistral-medium-2508`, `mistral-small-2506`, `mistral-small-latest`, `ministral-14b-2512`, `ministral-8b-2512`, `ministral-3b-2512`

### Wire format (OpenAI-compatible)

```python
from mistralai import Mistral

client = Mistral(api_key=api_key)

# URL-based
chat_response = client.chat.complete(
    model="mistral-small-latest",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": "https://example.com/img.jpg"},
        ],
    }],
)
```

> **Note:** Mistral's `image_url` field is a plain string, NOT an object like `{"url": "..."}`. This is a subtle difference from OpenAI.

### Key facts
- Wire format: OpenAI-like but `image_url` is a **string** (not `{"url": ...}`)
- Also supports base64 (as a data URL string in the same `image_url` field)
- All current Mistral and Ministral models support vision
- For OCR and document parsing, there is a separate Document Processing API

---

## 6. Cohere

**Source:** Web research (official docs 404'd)  
**Models:** `command-a-vision-07-2025` (Command A Vision — the only Cohere vision model)

### Key facts
- Vision support is **limited to Command A Vision** specifically — Command R, Command R+, etc. do NOT support images
- Format: base64-encoded images in message content
- Max images: 20 per request, 20MB total
- Supported MIME: PNG, JPEG, WEBP, GIF (non-animated)
- **Decision for v0.5:** Given that Cohere vision is limited to one specialized model and the API format is not well-documented in their public v2 docs, we should raise `VisionNotSupported` for Cohere initially, with a note in docs. We can revisit when they stabilize the API.

---

## 7. Azure OpenAI

**Source:** https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/gpt-with-vision  
**Models:** Same as OpenAI vision models, accessed via Azure deployment

### Key facts
- **Identical wire format to OpenAI** — same `image_url` content part structure
- Uses the `openai` Python SDK with `AzureOpenAI` client (custom base URL + `api-version`)
- The existing `azure.py` provider already uses the OpenAI SDK — vision support requires zero extra SDK work
- Same `detail` param (`auto`/`low`/`high`) and same size/format limits as OpenAI
- Implementation: same message serialization code as OpenAI — can share a helper

---

## 8. AWS Bedrock

**Source:** Web research + AWS docs  
**Models:** Any Claude 3+ model via Bedrock (e.g., `anthropic.claude-3-sonnet-20240229-v1:0`, `anthropic.claude-3-5-sonnet-20241022-v2:0`)

### Wire format (Converse API via boto3)

```python
response = client.converse(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    messages=[{
        "role": "user",
        "content": [
            {
                "image": {
                    "format": "jpeg",  # "jpeg" | "png" | "gif" | "webp"
                    "source": {
                        "bytes": image_bytes  # raw bytes, NOT base64 string
                    }
                }
            },
            {"text": "Describe this image."}
        ]
    }]
)
```

### Key facts
- Uses the **Converse API** (boto3 `client.converse()`)
- Image data goes in as **raw bytes** in the `source.bytes` field — boto3 handles base64 encoding automatically for the wire
- Format field: `"jpeg"`, `"png"`, `"gif"`, `"webp"` (no `image/` prefix)
- Also supports S3 URIs: `"source": {"s3Location": {"uri": "s3://...", "bucketOwner": "..."}}`
- Only supported on Claude 3+ Bedrock models (Claude 2 and earlier don't support vision via Converse)

---

## 9. Ollama

**Source:** https://github.com/ollama/ollama/blob/main/docs/api.md  
**Models:** Any multimodal model: `llava`, `bakllava`, `llama3.2-vision`, `minicpm-v`, etc.

### Wire format (Chat API)

```json
{
  "model": "llava",
  "messages": [
    {
      "role": "user",
      "content": "What's in this image?",
      "images": ["<base64_string>", "<base64_string>"]
    }
  ]
}
```

> The `images` field is a **top-level field on the message object**, NOT a nested content part.

### Generate API (older endpoint)
```json
{
  "model": "llava",
  "prompt": "What is in this picture?",
  "images": ["<base64_string>"]
}
```

### Key facts
- Image data: **list of raw base64 strings** (no data URI prefix, no mime type field)
- Placed in `message.images` for the Chat API (not inside `content`)
- Only base64 is supported — no URL references
- Multiple images: just add more strings to the `images` list
- Vision models must be pulled first: `ollama pull llava`
- The `ollama` Python SDK wraps this and accepts the same format

---

## Cross-Provider Pattern Analysis

### The "OpenAI-compatible" group (OpenAI, Azure, Groq, Mistral)
These all use the same basic `image_url` content part structure in the `messages` array. The llmgate abstraction should serialize to this format for all four. Minor differences:

| Provider | `image_url` value | Notes |
|---|---|---|
| OpenAI | `{"url": "...", "detail": "auto"}` | Object with optional `detail` |
| Azure | Same as OpenAI | Identical |
| Groq | `{"url": "..."}` | Object, no `detail` supported |
| Mistral | `"..."` (plain string) | String, NOT object |

### The "native" group (Anthropic, Bedrock, Gemini)
Each has its own format but they're all well-documented and used in the existing providers already (just not wired for images).

### The "images field" group (Ollama)
Completely different — uses a top-level `images` field on the message, not a content array.

---

## llmgate Public API Design

### New Types (in `types.py`)

```python
class ImageURL(BaseModel):
    """A URL-based image reference."""
    url: str                          # https:// or data: URI
    detail: str | None = None         # "auto" | "low" | "high" (OpenAI only)

class ImageBytes(BaseModel):
    """A raw-bytes or base64 image."""
    data: str                         # base64-encoded bytes
    mime_type: str                    # "image/jpeg" | "image/png" | "image/webp" | "image/gif"

class ImagePart(BaseModel):
    """One image in a multipart message."""
    type: Literal["image_url", "image_bytes"] = "image_url"
    image_url: ImageURL | None = None
    image_bytes: ImageBytes | None = None

class TextPart(BaseModel):
    """One text segment in a multipart message."""
    type: Literal["text"] = "text"
    text: str

# Message.content becomes str | list[TextPart | ImagePart]
```

### User-facing API

```python
from llmgate import completion
from llmgate.types import ImageURL, ImageBytes

# Simple: URL image (all OpenAI-compat providers)
resp = completion("gpt-4o-mini", [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        ],
    }
])

# Base64 image
import base64
with open("photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = completion("claude-opus-4-7", [
    {
        "role": "user",
        "content": [
            {"type": "image_bytes", "image_bytes": {"data": b64, "mime_type": "image/jpeg"}},
            {"type": "text", "text": "Describe this."},
        ],
    }
])
```

---

## Provider Support Matrix (final decision)

| Provider | Vision in v0.5 | Notes |
|---|---|---|
| OpenAI | ✅ Full | URL + base64, `detail` param |
| Azure | ✅ Full | Identical to OpenAI |
| Anthropic | ✅ Full | URL + base64, multi-image |
| Gemini | ✅ Full | Inline bytes (URL images fetched client-side) |
| Groq | ✅ Full | URL + base64, model must be llama-4-scout |
| Mistral | ✅ Full | URL + base64 |
| Ollama | ✅ Full | base64 only, `images` field |
| Bedrock | ✅ Full | bytes via Converse API |
| Cohere | 🚫 Raise `VisionNotSupported` | Only one model, unstable API docs |
