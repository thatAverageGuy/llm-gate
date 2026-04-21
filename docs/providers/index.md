# Providers

llmgate supports 9 providers across two tiers:

| Provider | Tier | Prefix | Install |
|---|---|---|---|
| [OpenAI](openai.md) | Core | `gpt-`, `o1-`, `o3-` | Included |
| [Anthropic](anthropic.md) | Core | `claude-` | Included |
| [Google Gemini](gemini.md) | Core | `gemini-` | Included |
| [Groq](groq.md) | Core | `groq/` | Included |
| [Mistral](mistral.md) | Optional | `mistral/` | `pip install llmgate[mistral]` |
| [Cohere](cohere.md) | Optional | `cohere/` | `pip install llmgate[cohere]` |
| [Azure OpenAI](azure.md) | Optional | `azure/` | Included (uses openai SDK) |
| [AWS Bedrock](bedrock.md) | Optional | `bedrock/` | `pip install llmgate[bedrock]` |
| [Ollama](ollama.md) | Optional | `ollama/` | `pip install llmgate[ollama]` |

---

## Feature matrix

| Provider | Completions | Streaming | Tools | Structured | Embeddings | Vision |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| OpenAI | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Anthropic | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Gemini | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Groq | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Mistral | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cohere | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Azure | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Bedrock | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Ollama | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Auto-detection

Provider is auto-detected from the model string. No configuration needed:

```python
completion("gpt-4o-mini", messages)           # → OpenAI
completion("claude-3-5-haiku-20241022", ...)  # → Anthropic
completion("gemini-2.5-flash-lite", ...)      # → Gemini
completion("groq/llama-3.1-8b-instant", ...)  # → Groq
completion("mistral/mistral-large", ...)       # → Mistral
```

Override with `provider=` if needed:

```python
completion("my-finetuned-model", messages, provider="openai")
```
