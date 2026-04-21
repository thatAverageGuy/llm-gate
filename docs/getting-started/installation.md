# Installation

## Requirements

- **Python 3.10+**
- An API key for at least one provider

---

## Install via pip

```bash
pip install llmgate
```

## Install via uv (recommended)

```bash
uv add llmgate
```

---

## Optional Provider Extras

The four core providers (OpenAI, Anthropic, Gemini, Groq) are included by default. The remaining five require their own SDK:

| Provider | Extra | Command |
|---|---|---|
| Mistral | `mistral` | `pip install llmgate[mistral]` |
| Cohere | `cohere` | `pip install llmgate[cohere]` |
| AWS Bedrock | `bedrock` | `pip install llmgate[bedrock]` |
| Ollama | `ollama` | `pip install llmgate[ollama]` |
| Azure OpenAI | *(none)* | Uses the `openai` package already bundled |
| **Everything** | `all` | `pip install llmgate[all]` |

---

## Verify the installation

```python
import llmgate
print(llmgate.__version__)   # 0.5.0
```
