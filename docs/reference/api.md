# API Reference

## `completion()`

```python
from llmgate import completion

resp = completion(
    model,          # str
    messages,       # list[dict | Message]
    *,
    provider=None,          # str | None
    api_key=None,           # str | None
    max_tokens=None,        # int | None
    temperature=None,       # float | None
    top_p=None,             # float | None
    stream=False,           # bool
    tools=None,             # list[ToolDefinition | dict] | None
    tool_choice=None,       # str | dict | None
    response_format=None,   # type[BaseModel] | None
    middleware=None,        # list[BaseMiddleware] | None
    **extra_kwargs,
) -> CompletionResponse | Iterator[StreamChunk]
```

## `acompletion()`

Async variant — identical signature, returns `Coroutine`.

---

## `parse()` / `aparse()`

```python
from llmgate import parse

instance = parse(model, messages, *, response_format: type[T], **kwargs) -> T
```

Shorthand for `completion(..., response_format=Model).parsed`.

---

## `embed()` / `aembed()`

```python
from llmgate import embed

resp = embed(
    model,                  # str — model with optional provider prefix
    input,                  # str | list[str] — batch in one call
    *,
    api_key=None,           # str | None
    dimensions=None,        # int | None  — OpenAI, Azure, Gemini, Mistral, Bedrock Titan V2
    task_type=None,         # str | None  — Gemini: "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY" | ...
    title=None,             # str | None  — Gemini: document title (task_type="RETRIEVAL_DOCUMENT")
    input_type=None,        # str | None  — Cohere/Bedrock-Cohere: "search_document" | "search_query" | ...
    truncate=None,          # str | None  — Cohere: "NONE"|"START"|"END"; Ollama: "true"|"false"
    encoding_format=None,   # str | None  — OpenAI/Azure/Mistral: "float" | "base64"
    user=None,              # str | None  — OpenAI/Azure end-user identifier
    **extra_kwargs,
) -> EmbeddingResponse
```

`aembed()` is the async variant — identical signature, returns `Coroutine`.

---

## `batch()` / `abatch()`

```python
from llmgate import batch

results = batch(
    requests,               # list[CompletionRequest | dict]
    *,
    max_concurrency=10,     # int
    fail_fast=False,        # bool
    middleware=None,        # list[BaseMiddleware] | None
    **extra_kwargs,
) -> BatchResult
```

---

## Types

### `Message`

```python
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[TextPart | ImagePart] | None
    tool_calls: list[ToolCall] | None
    tool_call_id: str | None
    name: str | None
```

### `TextPart`

```python
class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str
```

### `ImagePart`

```python
class ImagePart(BaseModel):
    type: Literal["image_url", "image_bytes"]
    image_url: ImageURL | None
    image_bytes: ImageBytes | None
```

### `ImageURL`

```python
class ImageURL(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] | None
```

### `ImageBytes`

```python
class ImageBytes(BaseModel):
    data: str       # base64-encoded, no data-URI prefix
    mime_type: str  # "image/jpeg" | "image/png" | "image/webp" | "image/gif"
```

### `CompletionResponse`

```python
class CompletionResponse(BaseModel):
    id: str
    model: str
    provider: str
    choices: list[Choice]
    usage: TokenUsage
    parsed: BaseModel | None
    raw: Any   # raw SDK response
    text: str  # property → choices[0].message.content
    tool_calls: list[ToolCall] | None  # property
```

### `EmbeddingResponse`

```python
class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    provider: str
    usage: TokenUsage
```

### `BatchResult`

```python
class BatchResult(BaseModel):
    results: list[CompletionResponse | None]
    errors: list[BatchError]
    successful: int
    failed: int
    total_tokens: int
    success_rate: float
```

### `TokenUsage`

```python
class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

### `ToolCall`

```python
class ToolCall(BaseModel):
    id: str
    function: str
    arguments: dict[str, Any]
```
