# Changelog

All notable changes to llmgate are documented here.

---

## v0.8.1 — 2026-04-30

### ✨ Added — Embedding Middleware

- **Embedding Middleware Support** — Middleware chains (`RetryMiddleware`, `LoggingMiddleware`, etc.) are now fully supported for embedding calls.
- `BaseMiddleware` extended with `embed_handle` and `aembed_handle` methods for synchronous and asynchronous embedding hooks.
- `LLMGate.embed` and `LLMGate.aembed` now properly pass requests through the middleware stack before hitting the providers.
- `RetryMiddleware` updated to natively catch transient errors on embeddings and retry with exponential backoff.

## v0.8.0 — 2026-04-30

### ✨ Added — Streaming Fallback Resilience

- **`stream=True` + model list** — Seamless mid-stream fallback recovery is now fully supported. You can pass a list of models alongside `stream=True` and if one fails mid-stream, llmgate will automatically failover to the next model in the chain.
- **`stream_fallback_mode`** — New parameter to configure mid-stream recovery strategy:
  - `"restart"` *(Default)*: Next model starts fresh. Safe and universally supported.
  - `"prefill"`: Buffers partial text and injects it as an `assistant` prefill, allowing the fallback model to pick up exactly where the previous model left off. Supported natively by Gemini, Groq, Mistral, Cohere, and Ollama.
  - `"user_turn"`: Wraps partial text in an assistant turn, followed by a user prompt to continue. Works universally.
- **Auto-Downgrade logic**: Providers that do not support prefilling (OpenAI, Anthropic, Azure, Bedrock) are automatically detected and downgraded to `"user_turn"` to avoid API schema rejections mid-stream.
- **Observability fields** added to `StreamChunk`:
  - `chunk.fallback_attempts`: A list of models that were tried before the current chunk's model.
  - `chunk.resumed_from_partial`: True if the stream resumed mid-way via prefill or user_turn.

---

## v0.7.0 — 2026-04-29

### ⚡ Performance — Embedding Batching

- **Gemini**: fixed N-call loop → single `embed_content(contents=[list])` call (was making one API call per chunk)
- **Ollama**: fixed N-call loop → single `embed(input=[list])` call
- **Bedrock**: parallel `invoke_model` calls via `ThreadPoolExecutor` (real-time API has no true batch endpoint); results returned in original order

### ✨ Added — Provider-Specific Embedding Params

New first-class parameters on `embed()` / `aembed()` and `EmbeddingRequest`:

| Param | Providers | Description |
|---|---|---|
| `task_type` | Gemini | Optimisation hint: `RETRIEVAL_DOCUMENT`, `RETRIEVAL_QUERY`, `SEMANTIC_SIMILARITY`, `CLASSIFICATION`, `CLUSTERING`, `QUESTION_ANSWERING`, `FACT_VERIFICATION` |
| `title` | Gemini | Document title — improves quality when `task_type="RETRIEVAL_DOCUMENT"` |
| `input_type` | Cohere, Bedrock-Cohere | Purpose hint: `search_document`, `search_query`, `classification`, `clustering` |
| `truncate` | Cohere, Ollama | Overflow strategy — Cohere: `NONE`/`START`/`END`; Ollama: `true`/`false` |
| `encoding_format` | OpenAI, Azure, Mistral | Output encoding: `float` or `base64` |
| `user` | OpenAI, Azure | End-user identifier for abuse monitoring |

Additional provider improvements:
- **Gemini**: `task_type` + `title` + `dimensions` now pass through `EmbedContentConfig`; response parsing cleaned up
- **Cohere**: `input_type` was hardcoded to `"search_document"` — now user-controlled; `extra_kwargs.pop()` mutation bug fixed
- **Bedrock**: `normalize` (default `True`) and `dimensions` forwarded into Titan V2 request body; Cohere-on-Bedrock gets `input_type` + `truncate`
- **Mistral**: fixed import path (`mistralai` not `mistralai.client`); `dimensions` maps to `output_dimension` for Matryoshka reduction; `encoding_format` forwarded
- **OpenAI / Azure**: `encoding_format` and `user` now explicit params; `kwargs.pop()` side-effect fixed → `kwargs.get()`

### 🧪 Tests

- Embedding test suite expanded: 23 → 53 tests (all mocked, no API keys needed in CI)
- Live-tested against Gemini (`gemini-embedding-001`): batch of 5, `task_type`, `title`, `dimensions=128`

---

## v0.6.0 — 2026-04-25

### ✨ Added — Fallback / Routing

- **`completion(model=[...])` / `acompletion(model=[...])`** — pass a list of model strings for automatic multi-provider fallback routing. First successful response wins.
- **`LLMGate(fallback_chain=[...], fallback_on=(...))`** — app-level fallback config; all middleware (retry, logging, etc.) applies per-candidate before advancing to the next model.
- **`FallbackMiddleware`** — composable middleware for drop-in fallback on existing middleware stacks.
- **`AllProvidersFailedError`** — raised when all models in the chain fail; carries `errors: list[tuple[str, Exception]]` for per-model diagnostics.
- **`CompletionResponse.fallback_attempts`** — new `list[str]` field indicating which models were tried (and failed) before this response. Empty on first-try success.
- Default `fallback_on`: `(RateLimitError, ProviderAPIError, AuthError)` — all three trigger fallback; configurable per-call or per-gate.
- `stream=True` + model list raises `ValueError` (streaming fallback planned for v0.7).
- 29 new mocked unit tests; live-tested against Groq, Anthropic, and Gemini.

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
