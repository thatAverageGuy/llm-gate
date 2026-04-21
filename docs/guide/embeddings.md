# Embeddings

Generate vector embeddings from text using `embed()` and `aembed()`.

---

## Basic usage

```python
from llmgate import embed

# Single string
resp = embed("text-embedding-3-small", "Hello world")
vector: list[float] = resp.embeddings[0]
print(len(vector))   # 1536

# Batch — list of strings
resp = embed("text-embedding-3-small", ["Hello", "world", "foo"])
vectors: list[list[float]] = resp.embeddings   # one per input
```

---

## Async

```python
from llmgate import aembed

resp = await aembed("text-embedding-3-small", "Hello world")
```

---

## Provider examples

=== "OpenAI"

    ```python
    resp = embed("text-embedding-3-small", text)
    resp = embed("text-embedding-3-large", text, dimensions=256)  # truncate
    ```

=== "Gemini"

    ```python
    resp = embed("gemini/text-embedding-004", text)
    ```

=== "Cohere"

    ```python
    resp = embed("cohere/embed-english-v3.0", text)
    resp = embed("cohere/embed-multilingual-v3.0", text)
    ```

=== "Mistral"

    ```python
    resp = embed("mistral/mistral-embed", text)
    ```

=== "Bedrock"

    ```python
    resp = embed("bedrock/amazon.titan-embed-text-v2:0", text)
    ```

=== "Ollama"

    ```python
    resp = embed("ollama/nomic-embed-text", text)
    ```

=== "Azure"

    ```python
    resp = embed("azure/my-embedding-deployment", text)
    ```

---

## The `EmbeddingResponse`

```python
resp.embeddings    # list[list[float]] — one vector per input string
resp.model         # str — model name
resp.provider      # str — "openai" | "gemini" | ...
resp.usage         # TokenUsage
resp.usage.prompt_tokens
resp.usage.total_tokens
```

---

## Provider support

| Provider | Supported | Notes |
|---|---|---|
| OpenAI | ✅ | `dimensions` param to truncate |
| Google Gemini | ✅ | Uses `gemini/` prefix |
| Azure OpenAI | ✅ | Uses `azure/` prefix |
| Cohere | ✅ | Batch supported |
| Mistral | ✅ | |
| Bedrock | ✅ | Titan models |
| Ollama | ✅ | Local models |
| Anthropic | ❌ | Raises `EmbeddingsNotSupported` |
| Groq | ❌ | Raises `EmbeddingsNotSupported` |
