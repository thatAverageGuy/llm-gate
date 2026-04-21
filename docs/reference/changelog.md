# Changelog

All notable changes to llmgate are documented here.

---

## v0.5.0 — 2026-04-21

### ✨ Added — Vision / Multimodal Support

- **New types** in `llmgate.types`:
  - `ImageURL` — URL or data-URI image reference with optional `detail` hint
  - `ImageBytes` — inline base64-encoded image with explicit MIME type
  - `TextPart` — text segment within a multipart message
  - `ImagePart` — image segment (`image_url` or `image_bytes` variant)
- **`Message.content`** widened from `str` to `str | list[TextPart | ImagePart]` — fully backward compatible
- **`llmgate/vision.py`** — central normalizer module with per-provider serializers:
  - `to_openai_content()` — OpenAI / Azure
  - `to_groq_content()` — Groq (strips unsupported `detail` param)
  - `to_mistral_content()` — Mistral (plain-string `image_url`)
  - `to_anthropic_content()` — Anthropic image source blocks
  - `to_gemini_parts()` — Gemini `Part` objects (URLs fetched client-side)
  - `to_bedrock_content()` — Bedrock Converse image blocks (raw bytes)
  - `to_ollama_message()` — Ollama top-level `images` list
- **New exception** `VisionNotSupported` — raised by Cohere
- **53 new tests** in `tests/test_vision.py`
- **Package exports**: `ImageURL`, `ImageBytes`, `TextPart`, `ImagePart`, `VisionNotSupported`

---

## v0.4.0 — 2026-04-11

### ✨ Added — Batch Completions

- `batch()` / `abatch()` — parallel completions with configurable concurrency
- `BatchResult` type with aggregate stats (`successful`, `failed`, `total_tokens`, `success_rate`)
- `BatchError` type with per-request failure details
- `fail_fast` mode to abort the batch on first error
- `BatchTimeoutError` exception for per-request timeouts
- `gate.batch()` / `gate.abatch()` on `LLMGate`

---

## v0.3.0 — 2026-03-31

### ✨ Added — Structured Outputs & Embeddings

- `parse()` / `aparse()` — shorthand returning a typed Pydantic instance
- `response_format` parameter on `completion()` / `acompletion()`
- `embed()` / `aembed()` — embeddings across 7 providers
- `EmbeddingRequest` / `EmbeddingResponse` types
- `EmbeddingsNotSupported` exception
- Per-provider embedding adapters: OpenAI, Gemini, Azure, Cohere, Mistral, Bedrock, Ollama

---

## v0.2.0 — 2026-03-25

### ✨ Added

- Streaming (`stream=True`) returning `Iterator[StreamChunk]` / `AsyncIterator[StreamChunk]`
- Tool / function calling with `ToolDefinition`, `FunctionDefinition`, `ToolCall`
- Composable middleware: `RetryMiddleware`, `CacheMiddleware`, `LoggingMiddleware`, `RateLimitMiddleware`
- `LLMGate` client class for gate-level middleware configuration
- 5 new optional providers: Mistral, Cohere, Azure OpenAI, AWS Bedrock, Ollama

---

## v0.1.0 — 2026-03-24

### 🎉 Initial release

- `completion()` / `acompletion()` supporting OpenAI, Anthropic, Gemini, Groq
- Unified `CompletionResponse` shape across all providers
- `AuthError`, `RateLimitError`, `ProviderAPIError`, `ModelNotFoundError` exceptions
- Provider auto-detection from model string prefix
- Full test suite (all mocked — no API keys needed in CI)
- GitHub Actions CI + PyPI Trusted Publishing
