# Embeddings

Generate vector embeddings from text using `embed()` and `aembed()`.

All inputs are sent in a **single API call** where the provider supports true batch endpoints (OpenAI, Azure, Gemini, Cohere, Mistral, Ollama). Bedrock is parallelised via a thread pool since its real-time endpoint is single-text only.

---

## Basic usage

```python
from llmgate import embed

# Single string → one vector
resp = embed("text-embedding-3-small", "Hello world")
vector: list[float] = resp.embeddings[0]
print(len(vector))   # 1536

# Batch — ONE API call for all inputs
resp = embed("text-embedding-3-small", ["Hello", "world", "foo"])
vectors: list[list[float]] = resp.embeddings   # one per input, same order
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

    # Reduce dimensions (text-embedding-3 models only)
    resp = embed("text-embedding-3-large", text, dimensions=256)

    # encoding_format and user tracking
    resp = embed("text-embedding-3-small", text,
                 encoding_format="float",
                 user="user-abc123")
    ```

=== "Gemini"

    ```python
    # Use gemini-embedding-001 (latest, 3072-dim)
    resp = embed("gemini/gemini-embedding-001", text)

    # ⚡ RAG — always set task_type for best retrieval quality
    # Embed your corpus documents:
    resp = embed("gemini/gemini-embedding-001", chunks,
                 task_type="RETRIEVAL_DOCUMENT")

    # Embed search queries:
    resp = embed("gemini/gemini-embedding-001", query,
                 task_type="RETRIEVAL_QUERY")

    # Add a title to improve document embedding quality:
    resp = embed("gemini/gemini-embedding-001", text,
                 task_type="RETRIEVAL_DOCUMENT",
                 title="My Document Title")

    # Reduce dimensions (Matryoshka — any size 1–3072):
    resp = embed("gemini/gemini-embedding-001", chunks, dimensions=128)
    ```

=== "Cohere"

    ```python
    # ⚠️ Always set input_type — wrong type degrades retrieval quality
    # Embed documents for your corpus:
    resp = embed("cohere/embed-english-v3.0", chunks,
                 input_type="search_document")

    # Embed a search query:
    resp = embed("cohere/embed-english-v3.0", query,
                 input_type="search_query")

    # Other supported values: "classification", "clustering"
    resp = embed("cohere/embed-multilingual-v3.0", texts,
                 input_type="search_document",
                 truncate="END")          # "NONE" | "START" | "END"
    ```

=== "Mistral"

    ```python
    resp = embed("mistral/mistral-embed", text)

    # Matryoshka dimension reduction:
    resp = embed("mistral/mistral-embed", text, dimensions=512)
    ```

=== "Bedrock"

    ```python
    # Titan Text Embeddings V2 (normalize=True by default — recommended for cosine sim)
    resp = embed("bedrock/amazon.titan-embed-text-v2:0", text)

    # Custom dimensions (Titan V2 supports 256 / 512 / 1024):
    resp = embed("bedrock/amazon.titan-embed-text-v2:0", chunks, dimensions=512)

    # Disable L2 normalisation:
    resp = embed("bedrock/amazon.titan-embed-text-v2:0", text, normalize=False)

    # Cohere on Bedrock — set input_type correctly:
    resp = embed("bedrock/cohere.embed-english-v3", chunks,
                 input_type="search_document")
    ```

=== "Ollama"

    ```python
    resp = embed("ollama/nomic-embed-text", text)
    resp = embed("ollama/mxbai-embed-large", texts)  # batch — one call

    # Control truncation behaviour (default: truncate silently):
    resp = embed("ollama/nomic-embed-text", text, truncate="false")  # error on overflow

    # Keep model loaded in memory:
    resp = embed("ollama/nomic-embed-text", text, keep_alive="1h")
    ```

=== "Azure"

    ```python
    resp = embed("azure/my-embedding-deployment", text)
    resp = embed("azure/my-embedding-deployment", text, dimensions=256)
    ```

---

## task_type — why it matters for RAG

!!! important "Set `task_type` for Gemini in production"
    Gemini's embedding model applies different transformations depending on the intended use.
    Using the wrong `task_type` (or none at all) will produce **lower quality retrieval**.

    | `task_type` | When to use |
    |---|---|
    | `RETRIEVAL_DOCUMENT` | Embedding corpus/knowledge-base chunks |
    | `RETRIEVAL_QUERY` | Embedding search/user queries |
    | `SEMANTIC_SIMILARITY` | Comparing sentence pairs |
    | `CLASSIFICATION` | Text classification labels |
    | `CLUSTERING` | Grouping similar texts |
    | `QUESTION_ANSWERING` | Q&A answer retrieval |
    | `FACT_VERIFICATION` | Fact-checking tasks |

```python
# Correct RAG setup:
doc_vecs = embed("gemini/gemini-embedding-001", chunks,
                 task_type="RETRIEVAL_DOCUMENT").embeddings

query_vec = embed("gemini/gemini-embedding-001", user_query,
                  task_type="RETRIEVAL_QUERY").embeddings[0]
```

---

## input_type — why it matters for Cohere

!!! important "Set `input_type` for Cohere in production"
    Cohere's embedding model is also task-aware. The default (`search_document`) is **wrong** for query vectors.

```python
# Index your knowledge base:
doc_vecs = embed("cohere/embed-english-v3.0", chunks,
                 input_type="search_document").embeddings

# At query time:
query_vec = embed("cohere/embed-english-v3.0", user_query,
                  input_type="search_query").embeddings[0]
```

---

## The `EmbeddingResponse`

```python
resp.embeddings    # list[list[float]] — one vector per input string, same order
resp.model         # str — model name as passed in
resp.provider      # str — "openai" | "gemini" | "cohere" | ...
resp.usage.prompt_tokens
resp.usage.total_tokens
```

---

## Full parameter reference

```python
embed(
    model,                  # str — model with provider prefix
    input,                  # str | list[str]
    *,
    api_key=None,           # str | None — override env var
    dimensions=None,        # int | None — OpenAI, Azure, Gemini, Mistral, Bedrock Titan V2
    task_type=None,         # str | None — Gemini only (see table above)
    title=None,             # str | None — Gemini, with task_type="RETRIEVAL_DOCUMENT"
    input_type=None,        # str | None — Cohere / Bedrock-Cohere
    truncate=None,          # str | None — Cohere: "NONE"|"START"|"END"; Ollama: "true"|"false"
    encoding_format=None,   # str | None — OpenAI/Azure/Mistral: "float" | "base64"
    user=None,              # str | None — OpenAI/Azure end-user identifier
    **extra_kwargs,         # any other provider-specific params
)
```

---

## Provider support

| Provider | Batching | Notes |
|---|---|---|
| OpenAI | ✅ Native | `dimensions`, `encoding_format`, `user` |
| Google Gemini | ✅ Native | `task_type`, `title`, `dimensions` — critical for RAG |
| Azure OpenAI | ✅ Native | `dimensions`, `encoding_format`, `user` |
| Cohere | ✅ Native | `input_type` (required for correct RAG), `truncate` |
| Mistral | ✅ Native | `dimensions` → `output_dimension`, `encoding_format` |
| Bedrock | ⚡ Parallel | Thread-pool parallelisation; `normalize`, `dimensions`, `input_type` |
| Ollama | ✅ Native | `truncate`, `keep_alive`, `options` via `extra_kwargs` |
| Anthropic | ❌ — | Raises `EmbeddingsNotSupported` |
| Groq | ❌ — | Raises `EmbeddingsNotSupported` |
